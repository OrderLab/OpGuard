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

class EntryHandlerInjector(cst.CSTTransformer):
    """
    Injects an import + a call to entry['inject'] at the end of
    entry['class'].entry['method'], preserving formatting.
    """
    def __init__(self, entry):
        self.entry = entry
        self.in_target_class = False
        self.did_inject_into_method = False
        self.seen_target_import = False

    def leave_ImportFrom(self, original_node, updated_node):
        # detect if our import is already present
        if (original_node.module
            and original_node.module.value == self.entry["import_module"]
            and any(alias.name.value == self.entry["import_name"]
                    for alias in original_node.names)):
            self.seen_target_import = True
        return updated_node

    def visit_ClassDef(self, node):
        if node.name.value == self.entry["class"]:
            self.in_target_class = True

    def leave_ClassDef(self, original_node, updated_node):
        self.in_target_class = False
        return updated_node

    def leave_FunctionDef(self, original_node, updated_node):
        # when we're in the target class and method
        if (
            self.in_target_class
            and original_node.name.value == self.entry["method"]
        ):
            # Build the injection call
            call = cst.SimpleStatementLine(
                [cst.Expr(cst.Call(func=cst.Name(self.entry["inject"]), args=[]))]
            )
            # Determine insertion index: skip over a leading docstring if present
            body_stmts = list(updated_node.body.body)
            insert_idx = 0
            if body_stmts and m.matches(body_stmts[0],
                                    m.SimpleStatementLine(m.Expr(m.SimpleString()))):
                insert_idx = 1

            # Insert at the beginning (after docstring, if any)
            new_body = (
                body_stmts[:insert_idx]
                + [call]
                + body_stmts[insert_idx:]
            )
            self.did_inject_into_method = True
            return updated_node.with_changes(
                body=updated_node.body.with_changes(body=new_body)
            )

        return updated_node




    def leave_Module(self, original_node, updated_node):
        # If we injected into the target method but haven’t seen its import yet:
        if self.did_inject_into_method and not self.seen_target_import:
            # Build a CST node for the dotted module path
            parts = self.entry["import_module"].split('.')
            module_node = cst.Name(parts[0])
            for attr in parts[1:]:
                module_node = cst.Attribute(value=module_node, dot=cst.Dot(), attr=cst.Name(attr))

            # Create the ImportFrom statement, with no relative dots
            import_stmt = cst.ImportFrom(
                module=module_node,
                names=[cst.ImportAlias(name=cst.Name(self.entry["import_name"]))],
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

            # Splice in the import
            new_body = (
                list(updated_node.body[:insert_at])
                + [import_line]
                + list(updated_node.body[insert_at:])
            )
            return updated_node.with_changes(body=new_body)

        return updated_node



def stage_file(path: str):
    """Run `git add <path>` to stage current contents."""
    subprocess.run(["git", "add", path], check=True)

def instrument_file(py_path: str, entry: dict):
    src = open(py_path, 'r', encoding='utf-8').read()
    mod = cst.parse_module(src)
    transformer = EntryHandlerInjector(entry)
    new_mod = mod.visit(transformer)
    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(new_mod.code)


def _get_git_root(path: str) -> str:
    """
    Given any file path inside a repo, return the repo's top-level directory.
    """
    file_dir = os.path.dirname(os.path.abspath(path))
    try:
        root = (
            subprocess
            .check_output(["git", "rev-parse", "--show-toplevel"], cwd=file_dir)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(f"{path} is not inside a Git repository")
    return root

def inject_entry_code(config, log_dir: str = None):
    log_entries = []

    for entry in config.get("entry", []):
        entry["import_name"] = entry.get("import_name", entry["inject"])
        rel = entry["original_module"]
        target = os.path.normpath(rel)
        log_entry = {
            "target": target,
            "entry": entry,
            "timestamp": datetime.now().isoformat()
        }

        if not os.path.isfile(target):
            print(f"⚠️  cannot find {target}", file=sys.stderr)
            log_entry["status"] = "not_found"
            log_entries.append(log_entry)
            continue

        print(f"\n🔍 Parsing & staging original: {target}")

        try:
            _ = cst.parse_module(open(target, 'r', encoding='utf-8').read())
        except Exception as e:
            print(f"❌ Syntax error in {target}: {e}", file=sys.stderr)
            log_entry["status"] = "syntax_error"
            log_entry["error"] = str(e)
            log_entries.append(log_entry)
            continue

        try:
            repo_root = _get_git_root(target)
            rel_path = os.path.relpath(target, repo_root)
            # subprocess.run(["git", "add", rel_path], cwd=repo_root, check=True)
            # print(f"[INFO] Staged original: {rel_path} (repo root: {repo_root})")

            instrument_file(target, entry)
            # subprocess.run(["git", "add", rel_path], cwd=repo_root, check=True)

            # print(f"[INFO] Staged modified: {rel_path}")
            log_entry["status"] = "success"
        except Exception as e:
            print(f"❌ Failed to inject into {target}: {e}", file=sys.stderr)
            log_entry["status"] = "failed"
            log_entry["error"] = str(e)

        log_entries.append(log_entry)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "entry_injection_log.json")
        with open(log_path, "w") as lf:
            json.dump(log_entries, lf, indent=2)
        print(f"[LOG] Entry injection log saved to {log_path}")

    print("\n✅ Injecting entry points done.")

def main():
    parser = argparse.ArgumentParser(description="Inject entry points into Python files.")
    parser.add_argument("config_file", type=str, help="Path to the config file.")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Path to the log directory.")
    args = parser.parse_args()

    inject_entry_code(args.config_file, args.log_dir)


if __name__ == "__main__":
    main()