import os
import yaml
import textwrap
import glob
import fnmatch
import argparse
import libcst as cst
from libcst._exceptions import ParserSyntaxError
from libcst.metadata import PositionProvider, MetadataWrapper
from datetime import datetime
import json

class MetaVarInjector(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, target_config):
        # Configuration for this target
        self.target_class = target_config.get("class") or target_config.get("cls")
        self.target_func = target_config.get("function") or target_config.get("method")
        # List of variable names to instrument
        self.var_names = list(target_config.get("var_name") or target_config.get("tag_names") or [])
        # Marker name to inject
        self.marker_name = target_config.get("name")
        self.marker_value = target_config.get("value")

        # State flags
        self.in_class = False
        self.in_func = False
        self.func_args = set()
        self.assigned_vars = set()
        self.injected = False  # track if any injection happened

    def visit_ClassDef(self, node: cst.ClassDef):
        # Use wildcard pattern matching for class names
        if self.target_class and fnmatch.fnmatchcase(node.name.value, self.target_class):
            self.in_class = True
        return node

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
        self.in_class = False
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef):
        # Use wildcard pattern matching for function/method names
        if (self.in_class or not self.target_class) and fnmatch.fnmatchcase(node.name.value, self.target_func):
            self.in_func = True
            self.func_args = {param.name.value for param in node.params.params}
            return node

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if not self.in_func:
            return updated_node

        body_stmts = list(updated_node.body.body)
        injections = []

        # -------------------
        # 1) Insert update_meta(var=var) calls
        # -------------------
        for var in self.var_names:
            stmt = cst.SimpleStatementLine([cst.Expr(cst.Call(
                func=cst.Name("update_meta"),
                args=[cst.Arg(keyword=cst.Name(var), value=cst.Name(var))]
            ))])
            # place at start or after assignment
            if var in self.func_args and var not in self.assigned_vars:
                injections.append((0, stmt))
            else:
                placed = False
                for idx, line in enumerate(body_stmts):
                    if isinstance(line, cst.SimpleStatementLine):
                        for sub in line.body:
                            if isinstance(sub, cst.Assign):
                                targets = {t.target.value for t in sub.targets if isinstance(t.target, cst.Name)}
                                if var in targets:
                                    injections.append((idx+1, stmt))
                                    placed = True
                                    break
                        if placed: break
                if not placed:
                    injections.append((0, stmt))
        if injections:
            self.injected = True
            injections.sort(key=lambda x: x[0])
            offset = 0
            for idx, stmt in injections:
                body_stmts.insert(idx+offset, stmt)
                offset += 1

        # 2) Inject marker if provided
        if self.marker_name and self.marker_value:
            marker = cst.SimpleStatementLine([cst.Expr(cst.Call(
                func=cst.Name("update_meta"),
                args=[cst.Arg(keyword=cst.Name(self.marker_name), value=cst.SimpleString(f'"{self.marker_value}"'))]
            ))])
            body_stmts.insert(0, marker)
            self.injected = True

        # 3) Only in forward: snapshot and hook
        if original_node.name.value == 'forward':
            for idx, line in reversed(list(enumerate(body_stmts))):
                if isinstance(line, cst.SimpleStatementLine):
                    for expr in line.body:
                        if isinstance(expr, cst.Return):
                            # snapshot vars
                            snap_mb = cst.SimpleStatementLine([cst.Assign(
                                targets=[cst.AssignTarget(cst.Name("micro_batch_idx"))],
                                value=cst.parse_expression(
                                    'get_meta(_opguard_global_rank())'
                                    ".get('micro_batch_idx', -1)"
                                )
                            )])
                            snap_gs = cst.SimpleStatementLine([cst.Assign(
                                targets=[cst.AssignTarget(cst.Name("global_step"))],
                                value=cst.parse_expression(
                                    'get_meta(_opguard_global_rank())'
                                    ".get('global_step', -1)"
                                )
                            )])
                            
                            # hook guard
                            hook = cst.If(
                                # check if hidden_states is defined
                                test=cst.Comparison(
                                    left=cst.SimpleString("'hidden_states'"),
                                    comparisons=[
                                        cst.ComparisonTarget(
                                            operator=cst.In(),
                                            comparator=cst.Call(func=cst.Name("locals"))
                                        )
                                    ]
                                ),
                                # if hidden_states is defined, check requires_grad
                                body=cst.IndentedBlock(body=[
                                    cst.If(
                                        test=cst.Attribute(cst.Name("hidden_states"), cst.Name("requires_grad")),
                                        body=cst.IndentedBlock(body=[
                                            cst.SimpleStatementLine([cst.Expr(cst.Call(
                                                func=cst.Attribute(cst.Name("hidden_states"), cst.Name("register_hook")),
                                                args=[cst.Arg(cst.Lambda(
                                                    params=cst.Parameters([cst.Param(cst.Name("grad"))]),
                                                    body=cst.parse_expression(
                                                        "update_meta(micro_batch_idx=micro_batch_idx, global_step=global_step) or grad"
                                                    )
                                                ))]
                                            ))])
                                        ]),
                                        
                                    )
                                ]),
                                
                            )
                            
                            # insert before return
                            body_stmts.insert(idx, hook)
                            body_stmts.insert(idx, snap_gs)
                            body_stmts.insert(idx, snap_mb)
                            self.injected = True
                            break

        self.in_func = False
        return updated_node.with_changes(body=updated_node.body.with_changes(body=body_stmts))

    def visit_Assign(self, node: cst.Assign):
        if self.in_func:
            for t in node.targets:
                if isinstance(t.target, cst.Name):
                    self.assigned_vars.add(t.target.value)
        return node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if self.injected:
            found = False
            for stmt in updated_node.body:
                if isinstance(stmt, cst.SimpleStatementLine):
                    body0 = stmt.body[0]
                    if isinstance(body0, cst.ImportFrom) and isinstance(body0.module, cst.Attribute):
                        mod = body0.module
                        if (isinstance(mod.value, cst.Attribute) and isinstance(mod.value.value, cst.Name)
                            and mod.value.value.value == "opguard" and mod.value.attr.value == "tracer"
                            and mod.attr.value == "meta_vars"):
                            names = {alias.name.value for alias in body0.names}
                            if {"update_meta","get_meta"}.issubset(names):
                                found = True
                                break
            imports = [cst.parse_statement(
                "from opguard.distributed import get_global_rank as _opguard_global_rank\n"
            )]
            if not found:
                imports += [
                    cst.SimpleStatementLine([cst.ImportFrom(
                        module=cst.Attribute(cst.Attribute(cst.Name("opguard"), cst.Name("tracer")), cst.Name("meta_vars")),
                        names=[cst.ImportAlias(cst.Name("update_meta")), cst.ImportAlias(cst.Name("get_meta"))]
                    )]),

                ]
            return updated_node.with_changes(body=imports + list(updated_node.body))
        return updated_node


def instrument_all_targets(config, log_dir: str = None):
    log_entries = []

    for target in config.get("meta_vars", []):
        file_pattern = target["file_path"]
        
        # Handle both absolute and relative paths
        pattern_path = file_pattern
        
                # Use glob with recursive=True for pattern matching
        matched_files = glob.glob(pattern_path, recursive=True)
        
        print(f"DEBUG: Pattern '{file_pattern}' -> '{pattern_path}' -> {len(matched_files)} files")
        
        if not matched_files:
            log_entries.append({
                "file_pattern": file_pattern,
                "status": "no_match",
                "timestamp": datetime.now().isoformat()
            })
            print(f"No files match pattern: {file_pattern}")
            continue

        for file_path in matched_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
                try:
                    module = cst.parse_module(code)
                except ParserSyntaxError:
                    dedented = textwrap.dedent(code)
                    module = cst.parse_module(dedented)
            except Exception as e:
                print(f"Skipping {file_path}: parse error: {e}")
                log_entries.append({
                    "file": file_path,
                    "status": "parse_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                continue

            wrapper = MetadataWrapper(module)
            transformed = wrapper.visit(MetaVarInjector(target))

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(transformed.code)
                print(f"Transformed {file_path}")
                log_entries.append({
                    "file": file_path,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"Failed to write {file_path}: {e}")
                log_entries.append({
                    "file": file_path,
                    "status": "write_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "meta_var_injection_log.json")
        with open(log_path, "w") as lf:
            json.dump(log_entries, lf, indent=2)
        print(f"[LOG] Meta-var injection log saved to {log_path}")




def main():
    parser = argparse.ArgumentParser(
        description="Instrument code based on config patterns."
    )

    parser.add_argument(
        "--config", "-c",
        dest="config_file",
        help="Path to the config file, default is opguard.yaml."
    )
    parser.add_argument(
        "--log-dir", "-l",
        dest="log_dir",
        help="Path to the log directory, default is ./logs."
    )
    args = parser.parse_args()

    config_path = args.config_file if args.config_file else "opguard.yaml"
    instrument_all_targets(config_path, args.log_dir)


if __name__ == "__main__":
    main()
