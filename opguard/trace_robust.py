#!/usr/bin/env python3
"""
Robust version of trace.py that uses the enhanced instrumentor to avoid syntax errors.
"""

import os
import sys
import yaml
import json
import difflib
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

# Import the robust instrumentor
from opguard.instrumentor.op_wrapper.file_api import instrument_file_robust

DEFAULT_CONFIG = "opguard.yaml"


def get_git_branch_and_commit(path):
    """Get current git branch and commit hash."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path, text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path, text=True
        ).strip()
        return path, branch, commit
    except subprocess.CalledProcessError:
        return path, "unknown", "unknown"


def validate_and_stage_source_code(source_path: str):
    """Validate source code and stage for git operations."""
    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=os.path.dirname(os.path.abspath(source_path)),
        text=True
    ).strip()

    branch_before, commit_before = get_git_branch_and_commit(repo_root)[1:]

    # Stage the file
    rel_path = os.path.relpath(source_path, repo_root)
    subprocess.run(["git", "add", rel_path], cwd=repo_root, check=True)

    return repo_root, branch_before, commit_before


def copy_config_to_target_repo(config_path: str, target_dir: str = None) -> str:
    """Copy config file to target repository."""
    if target_dir is None:
        # Find the git root of the current directory
        try:
            target_dir = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True
            ).strip()
        except subprocess.CalledProcessError:
            target_dir = os.getcwd()

    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Copy config file
    config_name = os.path.basename(config_path)
    target_config = os.path.join(target_dir, config_name)

    import shutil
    shutil.copy2(config_path, target_config)

    return target_config


def resolve_file_path(file_path: str) -> str:
    """Resolve file path. If relative, resolve against current working directory."""
    if os.path.isabs(file_path):
        return file_path
    else:
        # Use current working directory if relative
        return os.path.abspath(file_path)


def get_trace_output_dir():
    """Get the trace output directory."""
    from opguard._running import get_trace_output_dir as _get_trace_output_dir
    return _get_trace_output_dir()


def instrument_per_file_robust(config_path, original, output_file_dir=None, log_dir=None):
    """Robust version of instrument_per_file that uses the enhanced instrumentor."""
    assert os.path.exists(config_path), f"Config file not found: {config_path}"

    # Load config using unified config loader
    try:
        from opguard.utils.config_loader import load_opguard_config
        config_data = load_opguard_config(config_path)
    except ImportError:
        # Fallback to direct loading if config loader not available
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

    # Handle both old and new schema formats
    if "instrumentations" in config_data:
        config = config_data["instrumentations"]
    else:
        # Fallback for direct instrumentations list
        config = config_data

    # ---- path matching helper (handles absolute/relative/root differences) ----
    def _entry_matches_current_file(entry_path: str, current_file: str) -> bool:
        if not entry_path:
            return False
        try:
            # Fast path: samefile if both exist
            if os.path.exists(entry_path) and os.path.exists(current_file):
                try:
                    if os.path.samefile(entry_path, current_file):
                        return True
                except Exception:
                    pass
        except Exception:
            pass
        # Fallbacks: basename match OR normalized-suffix match
        if os.path.basename(entry_path) == os.path.basename(current_file):
            ep = os.path.normpath(entry_path).replace("\\", "/")
            cp = os.path.normpath(current_file).replace("\\", "/")
            return cp.endswith(ep) or ep.endswith(cp) or True
        return False

    # Filter config for new schema (with file_path entries).
    # For old schema, use all entries; for new schema, filter by file_path.
    filtered_config = []
    has_file_paths = any("file_path" in entry for entry in config)

    if has_file_paths:
        # New schema: filter entries that match the current file using robust matcher
        current_file = os.path.abspath(original)

        for entry in config:
            if "file_path" in entry:
                if _entry_matches_current_file(entry["file_path"], current_file):
                    # Include entries for this file; tag-based matching happens later
                    filtered_config.append(entry)
                    if os.getenv("OPGUARD_DEBUG"):
                        print(f"[DEBUG] Matched entry: {entry.get('function')} @ {entry.get('file_path')}:{entry.get('line_num','N/A')}")
            else:
                # Entry without file_path (old schema), include it
                filtered_config.append(entry)

        print(f"[INFO] Filtered config: {len(filtered_config)}/{len(config)} entries match current file {os.path.basename(original)}", file=sys.stderr)

        # If nothing matched and user asked to pass all, fall back.
        if not filtered_config and os.getenv("OPGUARD_FALLBACK_PASS_ALL", "0").lower() in ("1", "true", "yes"):
            print("[WARN] No entries matched this file_path; falling back to ALL entries per OPGUARD_FALLBACK_PASS_ALL")
            filtered_config = list(config)
    else:
        # Old schema: use all entries
        filtered_config = config
        print(f"[INFO] Using all {len(config)} config entries (old schema)", file=sys.stderr)

    # Final debug: what we will pass to the CST instrumentor
    if os.getenv("OPGUARD_DEBUG"):
        print(f"[DBG] Passing {len(filtered_config)} targets to instrumentor")
        if filtered_config:
            peek = dict(filtered_config[0])
            peek.pop("ops", None)
            peek.pop("stream_counts", None)
            print(f"[DBG] First target: {peek}")

    # repo_root, branch_before, commit_before = validate_and_stage_source_code(original)

    instrumented = output_file_dir or original
    diff_dir = log_dir or os.path.join(os.path.dirname(instrumented), "diffs")
    os.makedirs(diff_dir, exist_ok=True)
    diff_log = []

    def record_diff(step_name, before_lines, after_lines):
        diff = list(difflib.unified_diff(
            before_lines, after_lines,
            fromfile=f"{step_name}/before",
            tofile=f"{step_name}/after",
            lineterm=""
        ))
        diff_path = os.path.join(diff_dir, f"{step_name}_{os.path.basename(original)}.diff")
        with open(diff_path, "w", encoding="utf-8") as df:
            df.write("\n".join(diff))
        diff_log.append({"step": step_name, "diff_path": diff_path})

    before = open(instrumented, "r", encoding="utf-8").read().splitlines()

    # Use the robust instrumentor
    instrument_file_robust(original, instrumented, filtered_config)

    after = open(instrumented, "r", encoding="utf-8").read().splitlines()
    record_diff("instrument_file", before, after)

    # The two-pass line-number injector is optional in the robust flow.
    # Kept commented, but easy to re-enable:
    # before = after
    # from opguard.instrumentor.line_num_injector import apply_two_pass_rewrite
    # apply_two_pass_rewrite(instrumented, instrumented)
    # after = open(instrumented, "r", encoding="utf-8").read().splitlines()
    # record_diff("apply_two_pass_rewrite", before, after)

    print(f"[INFO] Robust instrumentation complete for {original}. Diffs and commit info saved to {diff_dir}")


def inject_global_code_robust(config, log_dir=None):
    """Robust version of inject_global_code."""
    print(f"[INFO] Running robust global injections...", file=sys.stderr)

    # Auto-inject meta_vars by scanning instrumented files and callstacks
    # Get list of instrumented files from config
    instrumented_files = []
    for item in config.get('instrumentations', []):
        if 'file_path' in item:
            file_path = item['file_path']
            if os.path.exists(file_path):
                instrumented_files.append(file_path)
    
    # Find callsite trace file to extract files from callstacks
    callsite_file = None
    trace_output_base = os.environ.get('OPGUARD_TRACE_OUTPUT_BASE')
    if trace_output_base:
        # Look for callsite_trace_combined_compressed.json in multiple locations
        search_dirs = [
            os.path.join(trace_output_base, "instrumentation"),
            os.path.join(trace_output_base, "callsite_trace"),
            trace_output_base
        ]
        
        possible_names = [
            "callsite_trace_combined_compressed.json",
            "*_combined_compressed.json",
            "*_compressed.json"
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
            for pattern in possible_names:
                import glob
                matches = glob.glob(os.path.join(search_dir, pattern))
                if matches:
                    # Use the most recent one
                    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                    callsite_file = matches[0]
                    print(f"[INFO] Found callsite trace file: {callsite_file}", file=sys.stderr)
                    break
            if callsite_file:
                break
    
    if instrumented_files or callsite_file:
        from opguard.instrumentor.auto_meta_var_injector import instrument_all_files_auto_meta_vars
        instrument_all_files_auto_meta_vars(
            instrumented_files,
            meta_var_names=None,  # Use default list
            log_dir=log_dir,
            dry_run=False,
            callsite_file=callsite_file
        )
    else:
        print(f"[WARN] No instrumented files or callsite trace found for auto meta-var injection", file=sys.stderr)

    # Inject entry points
    from opguard.instrumentor.entry_injector import inject_entry_code
    inject_entry_code(config, log_dir=log_dir)

    print(f"[INFO] Robust global injections complete.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run robust OpGuard instrumentation pipeline.")
    parser.add_argument("--config", help="Path to config.yaml (defaults to OPGUARD_CONFIG_PATH)")
    parser.add_argument(
        "--original", nargs="+", required=True,
        help="One or more original training script paths (supports wildcards)"
    )
    parser.add_argument(
        "--output-file", "-o",
        dest="output_file",
        help="Path to the output file. If omitted, the input file will be overwritten."
    )
    parser.add_argument("--parse-only", action="store_true", help="Only parse the original script")
    parser.add_argument("--copy-config", action="store_true",
                        help="Copy config to target repo before processing")
    parser.add_argument("--target-dir",
                        help="Target directory in code repo for config copy (defaults to code repo base)")
    # new quality-of-life flags
    parser.add_argument("--debug", action="store_true", help="Enable verbose OpGuard debug output")
    parser.add_argument("--fallback-pass-all", action="store_true",
                        help="If no entries match file_path, pass ALL config entries")
    parser.add_argument("--ratio", type=float, help="Override OPGUARD_TAG_MATCH_RATIO (e.g., 0.6)")
    parser.add_argument("--min-match", type=int, help="Override OPGUARD_TAG_MIN_MATCH (e.g., 1)")
    parser.add_argument("--skip-prologue", action="store_true",
                        help="Set OPGUARD_SKIP_PROLOGUE_INSERT=1 to only do Phase-A")

    args = parser.parse_args()

    import glob

    # Apply env toggles early so instrumentor sees them
    if args.debug:
        os.environ["OPGUARD_DEBUG"] = "1"
    if args.fallback_pass_all:
        os.environ["OPGUARD_FALLBACK_PASS_ALL"] = "1"
    if args.ratio is not None:
        os.environ["OPGUARD_TAG_MATCH_RATIO"] = str(args.ratio)
    if args.min_match is not None:
        os.environ["OPGUARD_TAG_MIN_MATCH"] = str(args.min_match)
    if args.skip_prologue:
        os.environ["OPGUARD_SKIP_PROLOGUE_INSERT"] = "1"

    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = os.environ.get('OPGUARD_CALLSITE_CONFIG')
        if not config_path:
            print("[ERROR] No config specified and OPGUARD_CALLSITE_CONFIG not set")
            sys.exit(1)

    # Copy config to target repo if requested
    if args.copy_config:
        config_path = copy_config_to_target_repo(config_path, args.target_dir)
        print(f"[INFO] Using copied config: {config_path}")

    expanded_paths = []
    for path in args.original:
        matched = glob.glob(path)
        if not matched:
            print(f"[WARNING] No files matched pattern: {path}")
        expanded_paths.extend(matched)

    if not expanded_paths:
        print("[ERROR] No matching files found.")
        sys.exit(1)

    # Run per-file instrumentation
    log_dir = f"{get_trace_output_dir()}_diff/opguard_logs"

    for orig in expanded_paths:
        if args.parse_only:
            validate_and_stage_source_code(orig)
            print(f"[INFO] Parsed and staged {orig}", file=sys.stderr)
        else:
            instrument_per_file_robust(
                config_path=config_path,
                original=orig,
                output_file_dir=args.output_file,
                log_dir=log_dir,
            )


if __name__ == "__main__":
    main()
