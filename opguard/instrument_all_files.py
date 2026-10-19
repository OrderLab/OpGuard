#!/usr/bin/env python3
"""
Robust instrumentation script that processes all files in a configuration.
Now with safe backup + revert support.

Usage:
  python instrument_all_files.py CONFIG.yaml [SOURCE_DIR]
  python instrument_all_files.py --mode revert [--dry-run] [--force]
"""

import os
import sys
import json
import yaml
import ast
import hashlib
import tempfile
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# ------------- Config / paths -------------
OPGUARD_DIR = Path(".opguard")
BACKUP_ROOT = OPGUARD_DIR / "backups"
MANIFEST = OPGUARD_DIR / "manifest.json"

# Safe backup location (system temp directory, not easily overwritten)
def get_safe_backup_root() -> Path:
    """Get a safe backup root directory in system temp that won't be easily overwritten."""
    temp_base = Path(tempfile.gettempdir())
    safe_root = temp_base / "opguard_safe_backups"
    safe_root.mkdir(parents=True, exist_ok=True)
    return safe_root

# ------------- Utilities -------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_manifest() -> Dict[str, Any]:
    if not MANIFEST.exists():
        return {"version": 1, "created": None, "items": {}, "globals": {}}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))

def save_manifest(m: Dict[str, Any]) -> None:
    OPGUARD_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

def _normalize_path_for_backup(path: Path) -> str:
    """Normalize absolute path for use in backup directory structure."""
    # Use absolute path, replacing path separators with underscores for filesystem safety
    abs_path = str(path.resolve())
    # Replace / with _ and handle special characters
    normalized = abs_path.replace(os.sep, '_').replace(':', '_')
    # Limit length to avoid filesystem issues
    if len(normalized) > 200:
        # Use hash for very long paths
        import hashlib
        hash_suffix = hashlib.md5(abs_path.encode()).hexdigest()[:8]
        normalized = normalized[:190] + '_' + hash_suffix
    return normalized

def ensure_backup(path: Path, dry_run: bool = False) -> Optional[Path]:
    """Copy original file to .opguard/backups/<normalized_path>.orig if not already."""
    normalized = _normalize_path_for_backup(path)
    backup_path = BACKUP_ROOT / (normalized + ".orig")
    if backup_path.exists():
        return backup_path  # already backed up
    if dry_run:
        return backup_path
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path.write_bytes(path.read_bytes())
    return backup_path

def ensure_safe_backup(path: Path, dry_run: bool = False) -> Optional[Path]:
    """
    Copy original file to a safe location (system temp) that won't be easily overwritten.
    Returns the safe backup path.
    """
    if not path.exists():
        return None
    normalized = _normalize_path_for_backup(path)
    safe_root = get_safe_backup_root()
    safe_backup_path = safe_root / (normalized + ".orig")
    if safe_backup_path.exists():
        return safe_backup_path  # already backed up
    if dry_run:
        return safe_backup_path
    safe_backup_path.parent.mkdir(parents=True, exist_ok=True)
    safe_backup_path.write_bytes(path.read_bytes())
    return safe_backup_path

def restore_backup(target: Path, dry_run: bool = False) -> bool:
    """Restore backup for the given target path."""
    normalized = _normalize_path_for_backup(target)
    backup_path = BACKUP_ROOT / (normalized + ".orig")
    if not backup_path.exists():
        return False
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(backup_path.read_bytes())
    return True

def restore_from_safe_backup(target: Path, safe_backup_path: Path, dry_run: bool = False) -> bool:
    """Restore from safe backup location and delete the safe backup after restoration."""
    if not safe_backup_path.exists():
        return False
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(safe_backup_path.read_bytes())
    # Delete the safe backup after successful restoration
    try:
        safe_backup_path.unlink()
    except Exception as e:
        print(f"[WARN] Failed to delete safe backup {safe_backup_path}: {e}")
    return True

def strip_marked_blocks(p: Path, dry_run: bool = False) -> bool:
    """
    Remove blocks between:
      # OPGUARD-BEGIN
      ...
      # OPGUARD-END
    Return True if modifications would be (or were) made.
    """
    try:
        s = p.read_text(encoding="utf-8")
    except Exception:
        return False
    import re
    pattern = re.compile(r"^[ \t]*#\s*OPGUARD-BEGIN[^\n]*\n.*?^[ \t]*#\s*OPGUARD-END[^\n]*\n?",
                         re.DOTALL | re.MULTILINE)
    if not pattern.search(s):
        return False
    if dry_run:
        return True
    s2 = pattern.sub("", s)
    p.write_text(s2, encoding="utf-8")
    return True

def extract_unique_file_paths(config_file):
    """Extract unique file paths from the instrumentation config."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    file_paths = set()
    if 'instrumentations' in config:
        for entry in config['instrumentations']:
            if 'file_path' in entry:
                file_paths.add(entry['file_path'])
    return list(file_paths)

def compile_check_python_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        ast.parse(source_code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except UnicodeDecodeError as e:
        return False, f"Unicode decode error: {e}"
    except Exception as e:
        return False, f"Compilation error: {e}"

def _load_file_log_any(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Could not parse log {path}: {e}")
        return None

    # v0: plain list of wrapped entries
    if isinstance(data, list):
        return {"wrapped": data, "missed": [], "config_only": []}

    # v1: dict with wrapped/missed/config_only
    if isinstance(data, dict):
        if any(k in data for k in ("wrapped", "missed", "config_only")):
            return {
                "wrapped": data.get("wrapped", []),
                "missed": data.get("missed", []),
                "config_only": data.get("config_only", []),
            }
        # very old: {"wrapped_log": [...]}
        if "wrapped_log" in data:
            return {"wrapped": data.get("wrapped_log", []), "missed": [], "config_only": []}

    print(f"[WARN] Unexpected log schema in {path}: {type(data).__name__}")
    return None


# ------------- Instrument path (existing) -------------

def run_instrumentation(config_file=None, dry_run: bool = False) -> bool:
    """Instrument files; create backups + manifest; run global injections."""
    # Load config
    try:
        from opguard.utils.config_loader import load_opguard_config
        config = load_opguard_config(config_file)
        print(f"[INFO] Loaded unified OpGuard config", file=sys.stderr)
        print(f"[INFO] Sections found:", file=sys.stderr)
        print(f"  - instrumentations: {len(config.get('instrumentations', []))} entries", file=sys.stderr)
        print(f"  - meta_vars: {len(config.get('meta_vars', []))} entries", file=sys.stderr)
        print(f"  - entry: {len(config.get('entry', []))} entries", file=sys.stderr)
    except ImportError:
        print("[ERROR] Config loader not available", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return False

    # Files — never rewrite torch/torchgen (breaks @torch.jit.script at import)
    from opguard.instrument_skip import filter_instrumentation_entries
    kept, skipped_paths = filter_instrumentation_entries(config.get('instrumentations', []))
    if skipped_paths:
        uniq_skipped = sorted(set(skipped_paths))
        print(
            f"[INFO] Skipping {len(skipped_paths)} protected third-party sites in {len(uniq_skipped)} files "
            f"(unsafe site-packages). Set OPGUARD_INSTRUMENT_UNSAFE=1 to override.",
            file=sys.stderr,
        )
        for p in uniq_skipped[:8]:
            print(f"  - {p}", file=sys.stderr)
        if len(uniq_skipped) > 8:
            print(f"  ... {len(uniq_skipped) - 8} more", file=sys.stderr)
        config['instrumentations'] = kept
        print(f"  - instrumentations after skip: {len(kept)} entries", file=sys.stderr)

    file_paths: List[str] = []
    for item in config.get('instrumentations', []):
        if 'file_path' in item:
            file_paths.append(item['file_path'])
    file_paths = sorted(set(file_paths))
    print(f"Found {len(file_paths)} unique files to instrument", file=sys.stderr)

    # Env
    if not os.environ.get("OPGUARD_TRACE_OUTPUT_BASE"):
        trace_output_base = os.path.abspath(os.path.join(os.getcwd(), "opguard-traces"))
        os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = trace_output_base
        print(f"[OpGuard] Set OPGUARD_TRACE_OUTPUT_BASE to: {trace_output_base}", file=sys.stderr)
    Path(os.environ['OPGUARD_TRACE_OUTPUT_BASE']).mkdir(parents=True, exist_ok=True)

    # NEW: choose lineno injection mode (default = begin)
    lineno_mode = os.environ.get("OPGUARD_LINENO_MODE", "begin").strip().lower()
    if lineno_mode not in ("begin", "end"):
        print(f"[WARN] Unknown OPGUARD_LINENO_MODE={lineno_mode!r}; defaulting to 'begin'", file=sys.stderr)
        lineno_mode = "begin"

    manifest = load_manifest()
    if manifest.get("created") is None:
        manifest["created"] = datetime.now().isoformat(timespec="seconds")
    manifest.setdefault("items", {})
    manifest.setdefault("globals", {"created_files": [], "notes": []})

    success_count = 0
    compilation_errors: List[Tuple[str, str]] = []
    repo_wrapped, repo_missed, repo_config_only = [], [], []

    for file_path in file_paths:
        try:
            print(f"Processing: {file_path}", file=sys.stderr)
            source_path = Path(file_path).resolve()
            if not source_path.exists():
                print(f"  Warning: File not found at {source_path}", file=sys.stderr)
                fallback = Path(os.path.basename(file_path)).resolve()
                if fallback.exists():
                    source_path = fallback
                    print(f"  Using local file: {source_path}", file=sys.stderr)

            # Backup before instrumentation (both local and safe backup)
            bkp = ensure_backup(source_path, dry_run=dry_run)
            if bkp is None:
                print(f"  ⚠️  Skipped backup (file missing?): {source_path}", file=sys.stderr)
            
            # Also create safe backup in system temp (won't be easily overwritten)
            safe_bkp = ensure_safe_backup(source_path, dry_run=dry_run)
            if safe_bkp is None:
                print(f"  ⚠️  Skipped safe backup (file missing?): {source_path}", file=sys.stderr)

            orig_hash = sha256_file(source_path) if source_path.exists() else None

            # Run instrumentation
            cmd = [
                sys.executable, '-m', 'opguard.trace_robust',
                '--original', str(source_path)
            ]
            if config_file:
                cmd.extend(['--config', config_file])
            print(f"Running: {' '.join(cmd)}", file=sys.stderr)
            if dry_run:
                result = 0
            else:
                result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr).returncode

            if result == 0:
                success_count += 1
                print(f"✅ Successfully instrumented: {file_path}", file=sys.stderr)

                # Post-compile check
                if str(source_path).endswith('.py') and source_path.exists():
                    print(f"🔍 Checking compilation: {source_path}", file=sys.stderr)
                    is_valid, error_msg = compile_check_python_file(str(source_path))
                    if is_valid:
                        print(f"✅ Compilation check passed: {file_path}", file=sys.stderr)
                    else:
                        print(f"❌ Compilation check failed: {file_path}", file=sys.stderr)
                        print(f"   Error: {error_msg}", file=sys.stderr)
                        compilation_errors.append((file_path, error_msg))
                else:
                    print(f"⚠️  Skipping compilation check (not a Python file): {file_path}", file=sys.stderr)

                # Update manifest item (use absolute path as key)
                new_hash = sha256_file(source_path) if source_path.exists() else None
                abs_path_str = str(source_path.resolve())
                manifest["items"][abs_path_str] = {
                    "backup": str(bkp) if bkp else None,
                    "safe_backup": str(safe_bkp) if safe_bkp else None,  # Safe backup in system temp
                    "original_sha256": orig_hash,
                    "instrumented_sha256": new_hash,
                    "mtime": datetime.now().isoformat(timespec="seconds"),
                }
                # Accumulate per-file log (accept .log.json or legacy .log)
                try:
                    log_candidates = [
                        Path(str(source_path) + ".log.json"),
                        Path(str(source_path) + ".log"),
                    ]
                    file_data = None
                    for fp in log_candidates:
                        if fp.exists():
                            file_data = _load_file_log_any(fp)
                            if file_data is not None:
                                break

                    if file_data:
                        # annotate with file, and be defensive about item shapes
                        def _extend_annotated(dst_list: list, items: list):
                            for it in items or []:
                                if isinstance(it, dict):
                                    it.setdefault("_file", str(source_path))
                                    dst_list.append(it)
                                else:
                                    # keep malformed entries from breaking aggregation
                                    dst_list.append({"_file": str(source_path), "_raw": it})

                        _extend_annotated(repo_wrapped, file_data.get("wrapped", []))
                        _extend_annotated(repo_missed, file_data.get("missed", []))
                        _extend_annotated(repo_config_only, file_data.get("config_only", []))
                except Exception as _e:
                    print(f"[WARN] Could not read per-file log for {source_path}: {_e}", file=sys.stderr)

            else:
                print(f"❌ Failed to instrument: {file_path}", file=sys.stderr)
                print(f"   Error: {result}", file=sys.stderr)
                print(f"   Command: {' '.join(cmd)}", file=sys.stderr)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}", file=sys.stderr)

    # Global injections
    print(f"\n[INFO] Running global injections...", file=sys.stderr)
    try:
        from opguard.trace_robust import inject_global_code_robust
        log_dir = f"{os.environ.get('OPGUARD_TRACE_OUTPUT_BASE', 'opguard-traces')}_diff/opguard_logs"
        if config:
            if dry_run:
                print(f"[DRY-RUN] Would inject global code into repo (log_dir={log_dir})", file=sys.stderr)
            else:
                created_files = inject_global_code_robust(
                    config, log_dir=log_dir
                ) or []
                # Record created files so we can remove on revert
                manifest["globals"].setdefault("created_files", [])
                manifest["globals"]["created_files"].extend(
                    [str(Path(p).resolve()) for p in created_files]
                )
            print(f"[INFO] Global injections completed", file=sys.stderr)
            # Apply line number injection
            print(f"[INFO] Applying line number injection (mode={lineno_mode})...", file=sys.stderr)
            try:
                if lineno_mode == "end":
                    from opguard.instrumentor.line_num_injector import apply_rewrite
                    for instrumented_file in file_paths:
                        if dry_run:
                            print(f"[DRY-RUN] Would apply line number rewrite (end): {instrumented_file}", file=sys.stderr)
                        else:
                            apply_rewrite(instrumented_file, instrumented_file)
                    print(f"[INFO] Line number injection (end) completed", file=sys.stderr)
                else:
                    print(f"[INFO] Skipping post-pass; using inline injection in wrapper injector (begin)", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Line number injection step skipped/failed: {e}", file=sys.stderr)
        else:
            print(f"[WARNING] OPGUARD_CALLSITE_CONFIG not set, skipping global injections", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Global injections failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    if not dry_run:
        save_manifest(manifest)
        try:
            report = {
                "wrapped_count": len(repo_wrapped),
                "missed_count": len(repo_missed),
                "config_only_count": len(repo_config_only),
                "wrapped": repo_wrapped,
                "missed": repo_missed,
                "config_only": repo_config_only,
            }
            OPGUARD_DIR.mkdir(parents=True, exist_ok=True)
            (OPGUARD_DIR / "instrument_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[INFO] Wrote repo report: {OPGUARD_DIR / 'instrument_report.json'}", file=sys.stderr)
        except Exception as _e:
            print(f"[WARN] Could not write repo report: {_e}", file=sys.stderr)


    print(f"\nInstrumentation complete: {success_count}/{len(file_paths)} files processed successfully", file=sys.stderr)
    if compilation_errors:
        print(f"\n⚠️  Compilation Issues Found ({len(compilation_errors)} files):", file=sys.stderr)
        for file_path, error_msg in compilation_errors:
            print(f"   {file_path}: {error_msg}", file=sys.stderr)
        print("\nFiles with compilation errors may need manual review.", file=sys.stderr)
    else:
        print("✅ All instrumented Python files passed compilation checks!", file=sys.stderr)

    # Short miss summary
    if repo_missed or repo_config_only:
        print("\n[SUMMARY] Instrumentation misses:", file=sys.stderr)
        # Count by reason
        from collections import Counter
        reason_ct = Counter(m.get("reason", "unknown") for m in repo_missed)
        for reason, cnt in reason_ct.most_common():
            print(f"  - {reason}: {cnt}", file=sys.stderr)
        if repo_config_only:
            print(f"  - config_only (targets never seen in AST): {len(repo_config_only)}", file=sys.stderr)
        print(f"\n  Full report: {OPGUARD_DIR / 'instrument_report.json'}", file=sys.stderr)
    else:
        print("\n[SUMMARY] No misses. 🎉", file=sys.stderr)

    all_success = success_count == len(file_paths) and len(compilation_errors) == 0
    return all_success

# ------------- Revert path -------------

def run_revert(dry_run: bool = False, force: bool = False) -> bool:
    """
    Revert instrumentation using .opguard/manifest.json:
      - Restore each backup
      - Strip OPGUARD markers
      - Remove files created by global injection
    """
    manifest = load_manifest()
    items: Dict[str, Any] = manifest.get("items", {})
    globals_meta: Dict[str, Any] = manifest.get("globals", {})
    created_files: List[str] = list(dict.fromkeys(globals_meta.get("created_files", [])))  # dedup

    if not items and not created_files:
        print("[INFO] Nothing to revert (manifest empty).", file=sys.stderr)
        return True

    print(f"[INFO] Starting revert (dry_run={dry_run}, force={force})", file=sys.stderr)
    ok = True

    # 1) Restore original files from backups
    for abs_path_str, meta in items.items():
        target = Path(abs_path_str)
        instr_hash_manifest = meta.get("instrumented_sha256")
        if target.exists():
            current_hash = sha256_file(target)
            if instr_hash_manifest and current_hash != instr_hash_manifest and not force:
                print(f"⛔  Hash mismatch, refusing to overwrite modified file: {target}", file=sys.stderr)
                print(f"    (use --force to override)", file=sys.stderr)
                ok = False
                continue
        restored = restore_backup(target, dry_run=dry_run)
        if restored:
            print(f"🔁 Restored: {target}", file=sys.stderr)
        else:
            print(f"⚠️  Backup missing for: {target} (skipped)", file=sys.stderr)
            ok = False

        # Best-effort: strip any residual OPGUARD markers in case backups are stale
        changed = strip_marked_blocks(target, dry_run=dry_run)
        if changed:
            print(f"🧹 Stripped OPGUARD blocks in: {target}", file=sys.stderr)

    # 2) Remove files created by global injections
    for f in created_files:
        p = Path(f)
        if p.exists():
            if dry_run:
                print(f"[DRY-RUN] Would delete injected file: {p}", file=sys.stderr)
            else:
                try:
                    p.unlink()
                    print(f"🗑️  Deleted injected file: {p}", file=sys.stderr)
                except Exception as e:
                    print(f"⚠️  Failed to delete {p}: {e}", file=sys.stderr)
                    ok = False

    # 3) (Optional) remove empty dirs under .opguard/backups on success
    if ok and not dry_run and force:
        # if user forced, assume they want a clean slate
        try:
            import shutil
            shutil.rmtree(OPGUARD_DIR)
            print(f"[INFO] Removed {OPGUARD_DIR} (clean slate).", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Could not remove {OPGUARD_DIR}: {e}", file=sys.stderr)

    print(f"[INFO] Revert {'completed' if ok else 'completed with issues'}", file=sys.stderr)
    return ok

# ------------- CLI -------------

def main_entry():
    # Simple flag parser to maintain compatibility with your argv expectations
    args = sys.argv[1:]

    mode = "instrument"
    dry_run = False
    force = False

    # Recognize flags
    if "--mode" in args:
        i = args.index("--mode")
        mode = args[i+1]
        del args[i:i+2]
    if "--dry-run" in args:
        dry_run = True
        args.remove("--dry-run")
    if "--force" in args:
        force = True
        args.remove("--force")

    # Legacy positional handling (CONFIG, SOURCE_DIR)
    if mode == "instrument":
        if len(args) < 1:
            callsite_config_file = os.environ.get("OPGUARD_CALLSITE_CONFIG")
            if not callsite_config_file:
                print("Error: OPGUARD_CALLSITE_CONFIG is not set")
                sys.exit(1)
        else:
            callsite_config_file = args[0] if len(args) >= 1 else None
        success = run_instrumentation(callsite_config_file, dry_run=dry_run)
        sys.exit(0 if success else 1)

    elif mode == "revert":
        # SOURCE_DIR is no longer needed since we use absolute paths
        success = run_revert(dry_run=dry_run, force=force)
        sys.exit(0 if success else 1)

    else:
        print(f"Unknown --mode {mode}; use 'instrument' or 'revert'")
        sys.exit(2)

if __name__ == "__main__":
    main_entry()
