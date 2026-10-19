"""
Automatic meta variable injector that scans instrumented files for assignments
to predefined meta variables and wraps them with update_meta calls.
"""
import os
import sys
import libcst as cst
from libcst._exceptions import ParserSyntaxError
from libcst.metadata import MetadataWrapper
from typing import Set, List, Dict, Any
from datetime import datetime
import json


# List of meta variables we're interested in tracking
DEFAULT_META_VARS = {
    "global_step",
    "model_chunk_id",
    "micro_batch_idx",
    "batch_idx",
    "epoch",
    "layer_number",
    "step",
    "iteration",
    "chunk_id",
}


class AutoMetaVarInjector(cst.CSTTransformer):
    """
    Automatically finds and wraps assignments to meta variables with update_meta calls.
    Scans all functions in the file, not just specific classes/methods.
    """
    METADATA_DEPENDENCIES = ()

    def __init__(self, meta_var_names: Set[str]):
        """
        Args:
            meta_var_names: Set of variable names to track (e.g., {"global_step", "micro_batch_idx"})
        """
        self.meta_var_names = meta_var_names
        self.injected = False
        self.assignments_count = 0

    def leave_SimpleStatementLine(
        self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine
    ) -> cst.SimpleStatementLine:
        """Insert update_meta calls after assignments to meta variables."""
        new_body = list(updated_node.body)
        inserted_any = False
        
        # Process each statement in the line
        for idx, stmt in enumerate(new_body):
            if isinstance(stmt, cst.Assign):
                # Check if any target is a meta variable we're interested in
                meta_vars_in_assignment = []
                for target in stmt.targets:
                    if isinstance(target.target, cst.Name):
                        var_name = target.target.value
                        if var_name in self.meta_var_names:
                            meta_vars_in_assignment.append(var_name)
                
                if meta_vars_in_assignment:
                    # Found assignment to meta variable(s) - create update_meta calls
                    self.injected = True
                    self.assignments_count += len(meta_vars_in_assignment)
                    update_calls = []
                    for var_name in meta_vars_in_assignment:
                        # Create: update_meta(var_name=var_name)
                        update_call = cst.Expr(cst.Call(
                            func=cst.Name("update_meta"),
                            args=[cst.Arg(keyword=cst.Name(var_name), value=cst.Name(var_name))]
                        ))
                        update_calls.append(update_call)
                    
                    # Insert update_meta calls after this assignment
                    insert_pos = idx + 1
                    for update_call in update_calls:
                        new_body.insert(insert_pos, update_call)
                        insert_pos += 1
                    inserted_any = True
            
            elif isinstance(stmt, cst.AugAssign):
                # Handle augmented assignments (e.g., +=, -=)
                if isinstance(stmt.target, cst.Name):
                    var_name = stmt.target.value
                    if var_name in self.meta_var_names:
                        self.injected = True
                        self.assignments_count += 1
                        # Create update_meta call after augmented assignment
                        update_call = cst.Expr(cst.Call(
                            func=cst.Name("update_meta"),
                            args=[cst.Arg(keyword=cst.Name(var_name), value=cst.Name(var_name))]
                        ))
                        new_body.insert(idx + 1, update_call)
                        inserted_any = True
        
        if inserted_any:
            return updated_node.with_changes(body=new_body)
        
        return updated_node

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Add imports if we injected any meta variable tracking."""
        if not self.injected:
            return updated_node
        
        # Check if imports already exist
        has_import = False
        for stmt in updated_node.body:
            if isinstance(stmt, cst.SimpleStatementLine):
                for item in stmt.body:
                    if isinstance(item, cst.ImportFrom):
                        if (isinstance(item.module, cst.Attribute) and
                            isinstance(item.module.value, cst.Attribute) and
                            isinstance(item.module.value.value, cst.Name) and
                            item.module.value.value.value == "opguard" and
                            item.module.value.attr.value == "tracer" and
                            item.module.attr.value == "meta_vars"):
                            # Check if update_meta is imported
                            for alias in item.names:
                                if isinstance(alias, cst.ImportAlias) and alias.name.value == "update_meta":
                                    has_import = True
                                    break
                            if has_import:
                                break
                if has_import:
                    break
        
        if not has_import:
            # Add import statement
            import_stmt = cst.SimpleStatementLine([cst.ImportFrom(
                module=cst.Attribute(
                    cst.Attribute(cst.Name("opguard"), cst.Name("tracer")),
                    cst.Name("meta_vars")
                ),
                names=[cst.ImportAlias(cst.Name("update_meta"))]
            )])
            return updated_node.with_changes(body=[import_stmt] + list(updated_node.body))
        
        return updated_node


def instrument_file_auto_meta_vars(
    file_path: str,
    meta_var_names: Set[str] = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Instrument a file to automatically track meta variable assignments.
    
    Args:
        file_path: Path to the Python file to instrument
        meta_var_names: Set of variable names to track (defaults to DEFAULT_META_VARS)
        dry_run: If True, don't write changes, just return what would be changed
    
    Returns:
        Dict with status, assignments_found, etc.
    """
    if meta_var_names is None:
        meta_var_names = DEFAULT_META_VARS
    
    if not os.path.exists(file_path):
        return {"status": "error", "error": f"File not found: {file_path}"}
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        return {"status": "error", "error": f"Failed to read file: {e}"}
    
    try:
        module = cst.parse_module(source_code)
    except ParserSyntaxError as e:
        return {"status": "error", "error": f"Parse error: {e}"}
    except Exception as e:
        return {"status": "error", "error": f"Failed to parse: {e}"}
    
    # Apply transformer
    transformer = AutoMetaVarInjector(meta_var_names)
    wrapper = MetadataWrapper(module)
    transformed_module = wrapper.visit(transformer)
    
    if not transformer.injected:
        return {
            "status": "no_changes",
            "file": file_path,
            "assignments_found": 0
        }
    
    if not dry_run:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(transformed_module.code)
        except Exception as e:
            return {"status": "error", "error": f"Failed to write file: {e}"}
    
    return {
        "status": "success",
        "file": file_path,
        "assignments_found": transformer.assignments_count,
        "meta_vars_tracked": list(meta_var_names)
    }


def extract_files_from_callstacks(callsite_file: str) -> Set[str]:
    """
    Extract all unique file paths from callstacks in a callsite trace JSON file.
    
    Args:
        callsite_file: Path to callsite trace JSON file (e.g., callsite_trace_combined_compressed.json)
    
    Returns:
        Set of unique file paths found in callstacks
    """
    if not os.path.exists(callsite_file):
        return set()
    
    try:
        with open(callsite_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load callsite file {callsite_file}: {e}", file=sys.stderr)
        return set()
    
    file_paths = set()
    
    # Iterate through all entries in the callsite data
    for key, entry in data.items():
        # Each entry has a "callstacks" field containing lists of frames
        callstacks = entry.get("callstacks", [])
        for stack in callstacks:
            # Each stack is a list of frames: [file_path, line_number, function_name]
            for frame in stack:
                if isinstance(frame, (list, tuple)) and len(frame) >= 1:
                    file_path = frame[0]
                    if file_path and isinstance(file_path, str) and file_path not in ("<unknown>", "?", ""):
                        # Normalize and resolve the path
                        if os.path.exists(file_path):
                            file_paths.add(os.path.abspath(file_path))
                        else:
                            # Try relative to current directory or base_dir
                            file_paths.add(file_path)  # Keep original for now, will filter later
    
    return file_paths


def instrument_all_files_auto_meta_vars(
    file_paths: List[str],
    meta_var_names: Set[str] = None,
    log_dir: str = None,
    dry_run: bool = False,
    callsite_file: str = None
) -> Dict[str, Any]:
    """
    Instrument all files to automatically track meta variable assignments.
    
    Args:
        file_paths: List of file paths to instrument
        meta_var_names: Set of variable names to track (defaults to DEFAULT_META_VARS)
        log_dir: Optional directory to save log file
        dry_run: If True, don't write changes
        callsite_file: Optional path to callsite trace JSON to extract additional files from callstacks
    
    Returns:
        Summary dict with results
    """
    if meta_var_names is None:
        meta_var_names = DEFAULT_META_VARS
    
    # Collect all files to instrument
    all_files = set(file_paths)
    
    # Extract files from callstacks if provided
    if callsite_file:
        print(f"[INFO] Extracting files from callstacks in: {callsite_file}", file=sys.stderr)
        callstack_files = extract_files_from_callstacks(callsite_file)
        if callstack_files:
            print(f"[INFO] Found {len(callstack_files)} unique files in callstacks", file=sys.stderr)
            all_files.update(callstack_files)
        else:
            print(f"[WARN] No files found in callstacks from {callsite_file}", file=sys.stderr)
    
    # Filter to only existing Python files
    python_files = []
    for file_path in all_files:
        if os.path.exists(file_path) and file_path.endswith('.py'):
            python_files.append(file_path)
        elif not os.path.exists(file_path):
            # Try to resolve relative paths
            abs_path = os.path.abspath(file_path)
            if os.path.exists(abs_path) and abs_path.endswith('.py'):
                python_files.append(abs_path)
    
    python_files = sorted(set(python_files))  # Remove duplicates and sort
    
    log_entries = []
    total_files = len(python_files)
    success_count = 0
    error_count = 0
    no_changes_count = 0
    
    print(f"[INFO] Auto-instrumenting {total_files} files for meta variables: {sorted(meta_var_names)}", file=sys.stderr)
    if callsite_file:
        print(f"[INFO]   - {len(file_paths)} from instrumented files", file=sys.stderr)
        print(f"[INFO]   - {len(python_files) - len(file_paths)} from callstacks", file=sys.stderr)
    
    for file_path in python_files:
        result = instrument_file_auto_meta_vars(file_path, meta_var_names, dry_run=dry_run)
        log_entries.append({
            "file": file_path,
            **result,
            "timestamp": datetime.now().isoformat()
        })
        
        if result["status"] == "success":
            success_count += 1
            print(f"[INFO] ✅ Instrumented {file_path} ({result.get('assignments_found', 0)} assignments found)", file=sys.stderr)
        elif result["status"] == "no_changes":
            no_changes_count += 1
        else:
            error_count += 1
            print(f"[WARN] ⚠️  Failed to instrument {file_path}: {result.get('error', 'unknown error')}", file=sys.stderr)
    
    summary = {
        "total_files": total_files,
        "success": success_count,
        "no_changes": no_changes_count,
        "errors": error_count,
        "meta_vars": sorted(meta_var_names),
        "entries": log_entries
    }
    
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "auto_meta_var_injection_log.json")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[INFO] Log saved to: {log_path}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to save log: {e}", file=sys.stderr)
    
    print(f"[INFO] Auto meta-var instrumentation complete: {success_count} success, {no_changes_count} no changes, {error_count} errors", file=sys.stderr)
    
    return summary


def main():
    """CLI for auto meta variable instrumentation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automatically instrument meta variable assignments"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Python files to instrument"
    )
    parser.add_argument(
        "--meta-vars",
        nargs="+",
        default=None,
        help="Meta variable names to track (default: predefined list)"
    )
    parser.add_argument(
        "--log-dir",
        help="Directory to save log file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes, just show what would be done"
    )
    
    args = parser.parse_args()
    
    meta_vars = set(args.meta_vars) if args.meta_vars else None
    
    instrument_all_files_auto_meta_vars(
        args.files,
        meta_var_names=meta_vars,
        log_dir=args.log_dir,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
