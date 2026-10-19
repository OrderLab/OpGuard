#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
import subprocess
import libcst as cst
from datetime import datetime
from libcst import matchers as m
import json
from libcst import parse_expression


class CallsiteTracerInjector(cst.CSTTransformer):
    """
    Injects CallsiteTracer context manager around specified function calls.
    """
    def __init__(self, config):
        self.config = config
        self.seen_import = False
        self.in_target_function = False
        self.target_function_name = None
        self.traced_functions = set()
        
        # Extract configuration
        for entry in config.get('callsite_tracer', []):
            if 'traced_functions' in entry:
                self.traced_functions.update(entry['traced_functions'])

    def _get_full_func_name(self, node):
        """Recursively extract dotted function name like `trainer.fit` or `a.b.c`"""
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            left = self._get_full_func_name(node.value)
            right = node.attr.value
            return f"{left}.{right}"
        return ""


    def visit_If(self, node):
        if m.matches(
            node.test,
            m.Comparison(
                left=m.Name(value="__name__"),  # or m.Name("__name__")
                comparisons=[
                    m.ComparisonTarget(
                        operator=m.Equal(),
                        comparator=m.SimpleString(
                            value=lambda v: v in ('"__main__"', "'__main__'")
                        ),
                    )
                ],
            ),
        ):
            for entry in self.config.get("callsite_tracer", []):
                if entry.get("function") == "__main__":
                    self.in_target_function = True
                    break
        # visit_* should return Optional[bool], not a node
        return True

    def leave_If(self, original_node, updated_node):
        self.in_target_function = False
        return updated_node

    def leave_ImportFrom(self, original_node, updated_node):
        if (
            m.matches(
                original_node.module,
                m.Attribute(
                    value=m.Attribute(
                        value=m.Name("opguard"),
                        attr=m.Name("callsite_tracer")
                    ),
                    attr=m.Name("trace")
                )
            )
            and any(alias.name.value == "CallsiteTracer" for alias in original_node.names)
        ):
            self.seen_import = True
        return updated_node


    def visit_FunctionDef(self, node):
        # Check if we're in a target function
        for entry in self.config.get('callsite_tracer', []):
            if (entry.get('function') == node.name.value):
                self.in_target_function = True
                self.target_function_name = node.name.value
                break

    def leave_FunctionDef(self, original_node, updated_node):
        self.in_target_function = False
        self.target_function_name = None
        return updated_node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        if not self.in_target_function:
            return updated_node
        # only single simple statement lines
        if len(updated_node.body) != 1:
            return updated_node

        stmt = updated_node.body[0]

        # helper to extract a Call from different statement shapes
        def _extract_top_call(node):
            # Expr:   foo(...)
            if isinstance(node, cst.Expr) and isinstance(node.value, cst.Call):
                return node.value
            # Await:  await foo(...)
            if isinstance(node, cst.Expr) and isinstance(node.value, cst.Await):
                if isinstance(node.value.expression, cst.Call):
                    return node.value.expression
            # Assign: x = foo(...)
            if isinstance(node, cst.Assign) and isinstance(node.value, cst.Call):
                return node.value
            # AnnAssign: x: T = foo(...)
            if isinstance(node, cst.AnnAssign) and node.value and isinstance(node.value, cst.Call):
                return node.value
            return None

        call = _extract_top_call(stmt)
        if not call:
            return updated_node
        # print("[TRACE CANDIDATE]", self._get_full_func_name(call.func))
        # print("[TRACED]", self.traced_functions)
        func_name = self._get_full_func_name(call.func)
        if func_name not in self.traced_functions:
                return updated_node

        print("[TRACE CANDIDATE]", self._get_full_func_name(call.func))


        # Wrap the WHOLE simple statement line in a with-block
        tracer_call = cst.Call(
            func=cst.Name("CallsiteTracer"),
            args=[cst.Arg(value=cst.Integer("100000"))],
        )
        with_stmt = cst.With(
            items=[cst.WithItem(item=tracer_call, asname=cst.AsName(name=cst.Name("tracer")))],
            body=cst.IndentedBlock(body=[updated_node]),
        )
        return with_stmt





    def leave_Module(self, original_node, updated_node):
        # Add import if we need it and it's not already present
        if not self.seen_import and self.traced_functions:
            # Create the import statement
            import_stmt = cst.ImportFrom(
                module=cst.Attribute(
                    value=cst.Attribute(
                        value=cst.Name("opguard"),
                        attr=cst.Name("callsite_tracer")
                    ),
                    attr=cst.Name("trace")
                ),
                names=[cst.ImportAlias(name=cst.Name("CallsiteTracer"))],
                relative=[]
            )

            import_line = cst.SimpleStatementLine([import_stmt])

            # Find insertion point (after any docstring or __future__ imports)
            insert_at = 0
            for idx, stmt in enumerate(updated_node.body):
                if m.matches(stmt, m.SimpleStatementLine(m.Expr(m.SimpleString()))):
                    insert_at = idx + 1
                elif m.matches(stmt, m.ImportFrom(module=m.Name("__future__"))):
                    insert_at = idx + 1
                else:
                    break

            # Insert the import
            new_body = (
                list(updated_node.body[:insert_at])
                + [import_line]
                + list(updated_node.body[insert_at:])
            )
            
            return updated_node.with_changes(body=new_body)
            
        return updated_node

def instrument_file(py_path: str, config: dict):
    """
    Instrument a Python file with callsite tracer based on configuration.
    """
    with open(py_path, 'r') as f:
        source = f.read()
    
    # Parse the source code
    tree = cst.parse_module(source)
    
    # Apply the transformer
    transformer = CallsiteTracerInjector(config)
    modified_tree = tree.visit(transformer)
    
    # Write back the modified code
    with open(py_path, 'w') as f:
        f.write(modified_tree.code)

def _get_git_root(path: str) -> str:
    """Find the git root directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            cwd=path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return path

def inject_callsite_tracer(config_path: str, log_dir: str = None):
    """
    Main function to inject callsite tracer based on configuration.
    
    Args:
        config_path: Path to the configuration YAML file
        log_dir: Optional directory for logging
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the directory containing the config file
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Process each callsite_tracer entry
    for entry in config.get('callsite_tracer', []):
        file_path = entry.get('file_path')
        if not file_path:
            continue
        # if file_path is relative, make it absolute by joining with config file directory
        if not os.path.isabs(file_path):
            file_path = os.path.join(config_dir, file_path)   
        
        # Normalize the path
        file_path = os.path.normpath(file_path)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist, skipping...")
            continue
            
        print(f"Instrumenting {file_path}...")
        
        try:
            instrument_file(file_path, config)
            print(f"Successfully instrumented {file_path}")
        except Exception as e:
            print(f"Error instrumenting {file_path}: {e}")
            if log_dir:
                import traceback
                with open(os.path.join(log_dir, f"instrumentation_error_{datetime.now().isoformat()}.log"), 'w') as f:
                    f.write(f"Error instrumenting {file_path}:\n")
                    f.write(traceback.format_exc())

def main():
    parser = argparse.ArgumentParser(
        description="Inject CallsiteTracer context managers based on configuration."
    )
    parser.add_argument("config", help="Path to configuration YAML file")
    parser.add_argument("--log-dir", help="Directory for logging errors")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} does not exist")
        sys.exit(1)
    
    inject_callsite_tracer(args.config, args.log_dir)

if __name__ == "__main__":
    main() 