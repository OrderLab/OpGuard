#!/usr/bin/env python3
"""
Unified OpGuard CLI - Simplified workflow for tracing and analysis

Two usage modes:
1. Quick mode: opguard run <scriptA> <scriptB> -> perfetto trace export
2. Traditional profiler mode: opguard start/stop/compare
"""

import os
import sys
import argparse
import subprocess
import yaml
import time
import tempfile
import shutil
import shlex
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import OpGuard modules
from opguard._running import get_trace_output_dir, get_trace_target_dir
from opguard.preprocessing import PreprocessingPipeline


class OpGuardCLI:
    """Main CLI coordinator for OpGuard"""
    
    def __init__(self):
        self.state_file = Path.home() / ".opguard_state.yaml"
        self.state = self._load_state()
    
    def _log_session_context(self, phase: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log high-level session context for debugging.
        
        NOTE: This intentionally logs to stderr so it never interferes with
        shell eval / process substitution output that goes to stdout.
        """
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            ctx: Dict[str, Any] = {
                "phase": phase,
                "pid": os.getpid(),
                "ppid": os.getppid() if hasattr(os, "getppid") else None,
                "cwd": os.getcwd(),
                "user": os.environ.get("USER") or os.environ.get("USERNAME"),
                "shell": os.environ.get("SHELL"),
                "output_dir": self.state.get("start_output_dir") or self.state.get("output_dir"),
                "tracing_active": self.state.get("tracing_active"),
                "state_file": str(self.state_file),
            }
            if extra:
                ctx.update(extra)
            # Render as a single line to keep logs easy to grep
            flat = " ".join(f"{k}={v}" for k, v in ctx.items() if v is not None)
            print(f"[CTX] {timestamp} {flat}", file=sys.stderr)
        except Exception:
            # Context logging must never break the main CLI flow
            pass
    
    def _load_state(self) -> Dict[str, Any]:
        """Load persistent CLI state"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}
    
    def _save_state(self):
        """Save persistent CLI state"""
        try:
            with open(self.state_file, 'w') as f:
                yaml.safe_dump(self.state, f)
        except Exception as e:
            print(f"[WARN] Failed to save state: {e}", file=sys.stderr)
    
    def _get_output_dir(self) -> str:
        """Get the current output directory"""
        return get_trace_target_dir()
    
    def _detect_script_type(self, script_path: str) -> tuple[str, list[str]]:
        """
        Detect script type and return (interpreter, command_args)
        Returns (interpreter_path, [script_path, ...]) or (None, [script_path]) for executable scripts
        """
        script = Path(script_path)
        
        # Check if file exists
        if not script.exists():
            # Default to Python if file doesn't exist (will fail later)
            return sys.executable, [script_path]
        
        # Check shebang
        try:
            with open(script, 'rb') as f:
                first_line = f.readline()
                if first_line.startswith(b'#!'):
                    shebang = first_line[2:].strip().decode('utf-8', errors='ignore')
                    # Parse shebang (e.g., "#!/bin/bash", "#!/usr/bin/env python3")
                    parts = shebang.split()
                    if parts:
                        interpreter = parts[0]
                        # Handle /usr/bin/env python3 case
                        if interpreter == '/usr/bin/env':
                            if len(parts) > 1:
                                interpreter = parts[1]
                            else:
                                interpreter = 'python3'
                        return interpreter, [script_path]
        except Exception:
            pass
        
        # Fall back to extension-based detection
        ext = script.suffix.lower()
        if ext in ('.sh', '.bash'):
            return '/bin/bash', [script_path]
        elif ext in ('.py', '.pyw'):
            return sys.executable, [script_path]
        elif ext == '':
            # No extension - check if executable
            if os.access(script_path, os.X_OK):
                # Try to read shebang again or default to bash
                try:
                    with open(script, 'rb') as f:
                        first_line = f.readline()
                        if first_line.startswith(b'#!'):
                            shebang = first_line[2:].strip().decode('utf-8', errors='ignore')
                            parts = shebang.split()
                            if parts:
                                interpreter = parts[0]
                                if interpreter == '/usr/bin/env' and len(parts) > 1:
                                    interpreter = parts[1]
                                return interpreter, [script_path]
                except Exception:
                    pass
                # Default executable to bash
                return '/bin/bash', [script_path]
        
        # Default to Python
        return sys.executable, [script_path]
    
    def _find_script_between_start_stop(self) -> Optional[str]:
        """
        Find the script that was likely run between opguard start and stop.
        Strategy:
        1. Check if we have a saved script from previous replay
        2. Look for recently modified Python/bash scripts in current directory and subdirectories
        3. Prefer scripts that were modified after start_time
        """
        # First, check if we have a saved script that still exists
        if self.state.get("last_script"):
            saved_script = self.state.get("last_script")
            if os.path.exists(saved_script):
                return saved_script
        
        # Get start time
        start_time = self.state.get("start_time", 0)
        if start_time == 0:
            return None
        
        # Look for scripts modified after start_time
        current_dir = os.getcwd()
        script_extensions = ['.py', '.sh', '.bash']
        candidates = []
        
        # Search in current directory and immediate subdirectories (not too deep)
        search_dirs = [current_dir]
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.') and item not in ['__pycache__', 'build', 'dist', 'node_modules']:
                search_dirs.append(item_path)
        
        for search_dir in search_dirs:
            try:
                for root, dirs, files in os.walk(search_dir):
                    # Skip hidden directories and common build/cache directories
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist', 'node_modules', '.git']]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        if any(file.endswith(ext) for ext in script_extensions):
                            try:
                                mtime = os.path.getmtime(file_path)
                                # Script should be modified after start_time (or very close to it, within 5 seconds before)
                                if mtime >= (start_time - 5):
                                    # Check if file is executable or has shebang
                                    if os.access(file_path, os.X_OK) or self._has_shebang(file_path):
                                        # Prefer scripts modified after start
                                        score = mtime if mtime >= start_time else mtime - 1000
                                        candidates.append((file_path, score, mtime))
                            except (OSError, IOError):
                                continue
            except (OSError, PermissionError):
                continue
        
        if candidates:
            # Sort by score (prefer scripts modified after start, then by recency)
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            # Return the best candidate
            return candidates[0][0]
        
        return None
    
    def _has_shebang(self, file_path: str) -> bool:
        """Check if file has a shebang line"""
        try:
            with open(file_path, 'rb') as f:
                first_line = f.readline()
                return first_line.startswith(b'#!')
        except Exception:
            return False
    
    def _generate_shell_activation_script(self, output_dir: str, enable_cupti: bool = False, 
                                         disable_cudaapi: bool = False, disable_python: bool = False) -> str:
        """Generate shell activation script that tracks executed commands"""
        # Get directory structure
        dirs = self._ensure_output_structure(output_dir)
        callsite_dir = dirs["callsite_trace"]
        exec_dir = dirs["exec_trace"]
        
        # Create a temporary directory for opguard wrappers
        opguard_bin_dir = os.path.join(output_dir, ".opguard_bin")
        os.makedirs(opguard_bin_dir, exist_ok=True)
        
        # Create wrapper scripts for common interpreters
        wrappers = {
            'python': 'python',
            'python3': 'python3',
            'bash': 'bash',
            'sh': 'sh',
        }
        
        # Get the actual opguard CLI path
        opguard_cli_path = shutil.which('opguard') or sys.executable + ' -m opguard.cli'
        
        for wrapper_name, interpreter in wrappers.items():
            wrapper_path = os.path.join(opguard_bin_dir, wrapper_name)
            
            # For bash/sh wrappers: log the command and set flag to prevent nested logging
            if wrapper_name in ('bash', 'sh'):
                wrapper_content = f"""#!/bin/bash
# OpGuard wrapper for {interpreter}
# This script tracks executed commands for opguard replay

# Check if we're already inside a bash script by checking parent process
# If parent is bash/sh and we're being called from within a script, don't log
SHOULD_LOG=true
if [ -n "$OPGUARD_IN_BASH_SCRIPT" ]; then
    SHOULD_LOG=false
elif [ -n "$BASH_SOURCE" ] && [ "${{BASH_SOURCE[0]}}" != "$0" ]; then
    # We're being sourced or called from within a script
    SHOULD_LOG=false
fi

# Log the command being executed (only if not already in a bash script)
if [ "$SHOULD_LOG" = "true" ]; then
    OPGUARD_SCRIPT_LOG="${{OPGUARD_SCRIPT_LOG:-{output_dir}/.opguard_scripts.log}}"
    echo "$(date +%s) {interpreter} $*" >> "$OPGUARD_SCRIPT_LOG"
    # Set flag to prevent nested commands from being logged
    export OPGUARD_IN_BASH_SCRIPT=1
fi

# Execute the actual command with the environment variable set
OPGUARD_IN_BASH_SCRIPT=1 exec {shutil.which(interpreter) or interpreter} "$@"
"""
            else:
                # For python wrappers: only log if not inside a bash script
                wrapper_content = f"""#!/bin/bash
# OpGuard wrapper for {interpreter}
# This script tracks executed commands for opguard replay

# Only log if not already inside a bash script (to avoid redundancy)
if [ -z "$OPGUARD_IN_BASH_SCRIPT" ]; then
    OPGUARD_SCRIPT_LOG="${{OPGUARD_SCRIPT_LOG:-{output_dir}/.opguard_scripts.log}}"
    echo "$(date +%s) {interpreter} $*" >> "$OPGUARD_SCRIPT_LOG"
fi

# Execute the actual command
exec {shutil.which(interpreter) or interpreter} "$@"
"""
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_content)
            os.chmod(wrapper_path, 0o755)
        
        # Generate activation script
        activation_script = os.path.join(output_dir, ".opguard_activate.sh")
        activation_content = f"""# OpGuard Shell Mode Activation Script
# Source this script to enter opguard monitoring mode
# Usage: source {activation_script}

if [ -z "$OPGUARD_MODE" ]; then
    export OPGUARD_MODE=1
    export OPGUARD_ORIGINAL_PATH="$PATH"
    export OPGUARD_BIN_DIR="{opguard_bin_dir}"
    export OPGUARD_SCRIPT_LOG="{output_dir}/.opguard_scripts.log"
    
    # Prepend opguard bin directory to PATH
    export PATH="$OPGUARD_BIN_DIR:$PATH"
    
    # Export tracing environment variables for callsite and runtime tracing
    export OPGUARD_AUTOSTART=1
    export OPGUARD_CALLSITE_TRACER=1
    export OPGUARD_RUNTIME_CALLSTACK=1
    export OPGUARD_TRACE_OUTPUT_BASE="{output_dir}"
    export OPGUARD_CALLSITE_OUTPUT_DIR="{callsite_dir}"
    export OPGUARD_EXEC_OUTPUT_DIR="{exec_dir}"
"""
        # CUPTI is disabled by default, only enable if explicitly requested
        if enable_cupti:
            activation_content += "    export OPGUARD_CALLSITE_CUPTI=1\n"
        else:
            activation_content += "    export OPGUARD_CALLSITE_CUPTI=0\n"
        if disable_cudaapi:
            activation_content += "    export OPGUARD_CALLSITE_CUDAAPI=0\n"
        if disable_python:
            activation_content += "    export OPGUARD_CALLSITE_PYTHON=0\n"
        
        activation_content += """
    
    # Clear the script log
    > "$OPGUARD_SCRIPT_LOG"
    
    echo "[OpGuard] Entered monitoring mode"
    echo "[OpGuard] All python/bash commands will be tracked"
    echo "[OpGuard] Callsite tracing: enabled"
    echo "[OpGuard] Run 'opguard stop' to exit this mode"
else
    echo "[OpGuard] Already in monitoring mode"
fi
"""
        
        with open(activation_script, 'w') as f:
            f.write(activation_content)
        os.chmod(activation_script, 0o644)
        
        # Generate deactivation script
        deactivation_script = os.path.join(output_dir, ".opguard_deactivate.sh")
        deactivation_content = f"""# OpGuard Shell Mode Deactivation Script
# Source this script to exit opguard monitoring mode
# Usage: source {deactivation_script}

# Stop tracing in subsequent Python processes.
export OPGUARD_AUTOSTART=0
export OPGUARD_CALLSITE_TRACER=0

if [ -n "$OPGUARD_MODE" ]; then
    # Restore original PATH
    if [ -n "$OPGUARD_ORIGINAL_PATH" ]; then
        export PATH="$OPGUARD_ORIGINAL_PATH"
    fi
    unset OPGUARD_MODE
    unset OPGUARD_ORIGINAL_PATH
    unset OPGUARD_BIN_DIR
    unset OPGUARD_SCRIPT_LOG
    unset OPGUARD_TRACE_OUTPUT_BASE
    unset OPGUARD_CALLSITE_OUTPUT_DIR
    unset OPGUARD_EXEC_OUTPUT_DIR
    
    echo "[OpGuard] Exited monitoring mode" >&2
else
    echo "[OpGuard] Not in monitoring mode" >&2
fi
"""
        
        with open(deactivation_script, 'w') as f:
            f.write(deactivation_content)
        os.chmod(deactivation_script, 0o644)
        
        return activation_script

    def _infer_output_dir_from_config(self, config_file: Optional[str]) -> Optional[str]:
        """Infer output directory from an instrumentation config path if possible."""
        if not config_file:
            return None
        try:
            cfg_path = Path(config_file)
            if cfg_path.parent.name == "instrumentation":
                return str(cfg_path.parent.parent.resolve())
        except Exception:
            return None
        return None

    def _resolve_callsite_input_dir(self, path: str) -> str:
        """Resolve a dump folder to the directory that actually holds callsite JSON."""
        p = Path(path).expanduser().resolve()
        if not p.exists():
            print(f"[ERROR] Path not found: {p}", file=sys.stderr)
            sys.exit(1)
        if p.is_file():
            return str(p)
        nested = p / "callsite_trace"
        if nested.is_dir() and any(nested.rglob("*.json*")):
            return str(nested)
        if any(p.rglob("*.json*")):
            return str(p)
        print(f"[ERROR] No callsite json/jsonl found under: {p}", file=sys.stderr)
        print("[ERROR] Expected <dir>/callsite_trace/*.json or json files in <dir>", file=sys.stderr)
        sys.exit(1)

    def _find_instrumentation_yaml(self, instrumentation_dir: str) -> Optional[str]:
        """Pick the best generated instrumentation YAML in a directory."""
        if not os.path.isdir(instrumentation_dir):
            return None
        summary_file = os.path.join(instrumentation_dir, "pipeline_summary.json")
        if os.path.exists(summary_file):
            try:
                import json
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                config_file = summary.get("sync_yaml") or summary.get("filtered_yaml") or summary.get("final_combined_yaml")
                if config_file and os.path.exists(config_file):
                    return config_file
            except Exception as e:
                print(f"[WARN] Failed to read pipeline summary: {e}", file=sys.stderr)
        yaml_patterns = [
            "*_sync_instrumentation.yaml",
            "*__filtered_sync_instrumentation.yaml",
            "*_filtered_instrumentation.yaml",
            "final_combined_instrumentation.yaml",
            "*_instrumentation.yaml",
            "*.yaml",
        ]
        for pattern in yaml_patterns:
            matches = list(Path(instrumentation_dir).glob(pattern))
            if matches:
                matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(matches[0])
        return None

    def _preprocess_callsite_dir(self, callsite_dir: str, output_dir: str, base_dir: Optional[str] = None) -> str:
        """Merge/compress callsite dumps and generate instrumentation YAML."""
        print(f"[INFO] Merging callsite traces from: {callsite_dir}", file=sys.stderr)
        dirs = self._ensure_output_structure(output_dir)
        instrumentation_dir = dirs["instrumentation"]
        pipeline = PreprocessingPipeline(base_dir=base_dir)
        original_stdout = sys.stdout
        try:
            class StderrRedirect:
                def write(self, s):
                    sys.stderr.write(s)
                    sys.stderr.flush()
                def flush(self):
                    sys.stderr.flush()
                def __getattr__(self, name):
                    return getattr(sys.stderr, name)
            sys.stdout = StderrRedirect()
            summary = pipeline.run_pipeline(
                inputs=[callsite_dir],
                output_dir=instrumentation_dir,
            )
        finally:
            sys.stdout = original_stdout
        config_file = (
            summary.get("sync_yaml")
            or summary.get("filtered_yaml")
            or summary.get("final_combined_yaml")
        )
        if not config_file or not os.path.exists(str(config_file)):
            config_file = self._find_instrumentation_yaml(instrumentation_dir)
        if not config_file:
            print(f"[ERROR] Preprocessing finished but no instrumentation YAML in {instrumentation_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Generated instrumentation config: {config_file}", file=sys.stderr)
        return str(config_file)

    def _emit_instrument_env_exports(self, output_dir: str, dirs: Dict[str, str], args, config_file: Optional[str]) -> None:
        """Emit shell exports for running instrumented code with runtime tracing + local dump."""
        is_tty = sys.stdout.isatty() and sys.stderr.isatty()
        shell = os.environ.get("SHELL", "").split("/")[-1] if os.environ.get("SHELL") else ""
        can_use_process_subst = shell in ("bash", "zsh") and is_tty

        if can_use_process_subst:
            cmd = ["opguard", "instrument"]
            if config_file:
                cmd += ["-c", config_file]
            if getattr(args, "disable_local_dump", False):
                cmd.append("--disable-local-dump")
            else:
                sync_interval = getattr(args, "local_dump_sync_interval", 1800.0)
                if sync_interval != 1800.0:
                    cmd += ["--local-dump-sync-interval", str(sync_interval)]
            if getattr(args, "no_emit_env", False):
                cmd.append("--no-emit-env")

            quoted = " ".join(shlex.quote(x) for x in cmd)
            print("[INFO] To export runtime env for instrumented runs, use:", file=sys.stderr)
            print(f"  source <({quoted})", file=sys.stderr)
            print("", file=sys.stderr)
            print("[INFO] Or use eval:", file=sys.stderr)
            print(f"  eval $({quoted})", file=sys.stderr)
            print("", file=sys.stderr)
            print("[INFO] Shell commands (for eval/process substitution):", file=sys.stderr)

        # Runtime tracing envs (no PATH or wrapper injection here)
        print("export OPGUARD_AUTOSTART=1")
        print("export OPGUARD_CALLSITE_TRACER=0")
        print("export OPGUARD_RUNTIME_CALLSTACK=1")
        print(f"export OPGUARD_TRACE_OUTPUT_BASE=\"{output_dir}\"")
        print(f"export OPGUARD_CALLSITE_OUTPUT_DIR=\"{dirs['callsite_trace']}\"")
        print(f"export OPGUARD_EXEC_OUTPUT_DIR=\"{dirs['exec_trace']}\"")

        if not getattr(args, "disable_local_dump", False):
            sync_interval = getattr(args, "local_dump_sync_interval", 1800.0)
            print("export OPGUARD_LOCAL_DUMP_ENABLED=1")
            print(f"export OPGUARD_LOCAL_DUMP_TARGET=\"{output_dir}\"")
            print(f"export OPGUARD_LOCAL_DUMP_SYNC_INTERVAL={sync_interval}")
            base_local = os.environ.get("OPGUARD_LOCAL_DUMP_BASE", "/tmp/opguard_local_dump")
            print(f"export OPGUARD_LOCAL_DUMP_BASE=\"{base_local}\"")
        else:
            print("export OPGUARD_LOCAL_DUMP_ENABLED=0")

        print("echo '[OpGuard] Instrumentation env loaded' >&2")
    
    def _get_tracked_scripts(self, output_dir: str) -> List[str]:
        """Get list of scripts executed in opguard mode (only top-level scripts)"""
        script_log = os.path.join(output_dir, ".opguard_scripts.log")
        if not os.path.exists(script_log):
            return []
        
        scripts = []
        seen_scripts = set()
        try:
            with open(script_log, 'r') as f:
                lines = f.readlines()
            
            # Process lines in order, but only keep the first occurrence of each script
            # This handles the case where a bash script internally calls other scripts
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Format: timestamp interpreter args...
                parts = line.split(' ', 2)
                if len(parts) >= 3:
                    interpreter = parts[1]
                    args = parts[2]
                    # Extract script path from args (first non-flag argument)
                    arg_parts = args.split()
                    for arg in arg_parts:
                        # Skip if it's a flag or doesn't look like a file path
                        if arg.startswith('-') or arg.startswith('+'):
                            continue
                        # Check if it's an existing file
                        if os.path.exists(arg):
                            # Only add if we haven't seen this script before
                            # This ensures we only track the top-level script
                            if arg not in seen_scripts:
                                scripts.append(arg)
                                seen_scripts.add(arg)
                            break
                        # Also check if it's a relative path that might exist
                        elif '/' in arg or arg.endswith('.py') or arg.endswith('.sh'):
                            # Try to resolve it
                            resolved = os.path.abspath(arg) if not os.path.isabs(arg) else arg
                            if os.path.exists(resolved) and resolved not in seen_scripts:
                                scripts.append(resolved)
                                seen_scripts.add(resolved)
                            break
        except Exception as e:
            print(f"[WARN] Failed to read script log: {e}")
        
        # Return only the first script (the top-level one)
        # If multiple scripts were executed, the first one is usually the main one
        return scripts[:1] if scripts else []
    
    def _ensure_output_structure(self, base_dir: str) -> Dict[str, str]:
        """Ensure the unified output directory structure exists"""
        base = Path(base_dir)
        dirs = {
            "callsite_trace": base / "callsite_trace",
            "exec_trace": base / "exec_trace",
            "instrumentation": base / "instrumentation",
            "report": base / "report",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return {k: str(v) for k, v in dirs.items()}
    
    def _check_trace_exists(self, base_dir: str) -> bool:
        """Check if trace files already exist in the directory"""
        base = Path(base_dir)
        # Check for any existing trace files
        for pattern in ["*.json", "*.jsonl", "*.yaml"]:
            if list(base.rglob(pattern)):
                return True
        return False
    
    def configure(self, args):
        """Display and confirm all OPGUARD_* environment variables"""
        print("=" * 80)
        print("OpGuard Configuration")
        print("=" * 80)
        
        opguard_vars = {k: v for k, v in os.environ.items() if k.startswith("OPGUARD_")}
        
        if not opguard_vars:
            print("No OPGUARD_* environment variables set.")
        else:
            print("\nCurrent OPGUARD_* variables:")
            for key in sorted(opguard_vars.keys()):
                value = opguard_vars[key]
                # Mask sensitive values
                if "token" in key.lower() or "key" in key.lower():
                    value = "*" * len(value) if value else ""
                print(f"  {key}={value}")
        
        print("\nCurrent output directory:")
        try:
            output_dir = self._get_output_dir()
            print(f"  {output_dir}")
        except Exception as e:
            print(f"  [ERROR] Failed to determine output directory: {e}")
            print(f"  [INFO] Fallback: ~/.opguard/traces")
        print("=" * 80)
    
    def dumpto(self, args):
        """Set the output directory for OpGuard traces"""
        target_dir = args.dir
        
        if target_dir.lower() == "default":
            # Reset to default output configuration
            if "output_dir" in self.state:
                del self.state["output_dir"]
            self._save_state()
            print(f"[INFO] Reset output directory to default (output configuration)")
            print(f"[INFO] Current output: {get_trace_output_dir()}")
        else:
            # Set custom directory
            if not os.path.isabs(target_dir):
                # Resolve relative paths relative to current working directory
                target_dir = os.path.abspath(target_dir)
            
            self.state["output_dir"] = target_dir
            self._save_state()
            print(f"[INFO] Set output directory to: {target_dir}")
            print(f"[INFO] Directory structure will be created at: {target_dir}")
    
    def start(self, args):
        """Start tracing (callsite + runtime)"""
        # If directory argument is provided, set it as output directory (equivalent to dumpto)
        if args.dir:
            target_dir = args.dir
            if target_dir.lower() == "default":
                # Reset to default output configuration
                if "output_dir" in self.state:
                    del self.state["output_dir"]
                self._save_state()
                print(f"[INFO] Reset output directory to default (output configuration)", file=sys.stderr)
            else:
                # Set custom directory
                if not os.path.isabs(target_dir):
                    # Resolve relative paths relative to current working directory
                    target_dir = os.path.abspath(target_dir)
                
                self.state["output_dir"] = target_dir
                self._save_state()
                print(f"[INFO] Set output directory to: {target_dir}", file=sys.stderr)
        
        # Log session context as early as possible (after potential dumpto handling)
        self._log_session_context(
            "start_begin",
            extra={
                "arg_dir": getattr(args, "dir", None),
                "enable_cupti": getattr(args, "enable_cupti", False),
                "disable_cudaapi": getattr(args, "disable_cudaapi", False),
                "disable_python": getattr(args, "disable_python", False),
                "no_auto_activate": getattr(args, "no_auto_activate", False),
            },
        )
        
        # Check if code has been instrumented and restore it if needed
        from opguard.instrument_all_files import load_manifest, run_revert, restore_from_safe_backup
        from pathlib import Path
        manifest = load_manifest()
        manifest_items = manifest.get("items", {})
        is_instrumented = len(manifest_items) > 0
        
        if is_instrumented:
            print("[INFO] Detected instrumented code. Checking for safe backups...", file=sys.stderr)
            
            # First, try to restore from safe backups if they exist
            safe_backups_found = False
            safe_backups_used = 0
            safe_backups_failed = 0
            
            for abs_path_str, meta in manifest_items.items():
                safe_backup_path_str = meta.get("safe_backup")
                if safe_backup_path_str:
                    safe_backup_path = Path(safe_backup_path_str)
                    if safe_backup_path.exists():
                        safe_backups_found = True
                        target_path = Path(abs_path_str)
                        print(f"[INFO] Found safe backup for {target_path}, restoring...", file=sys.stderr)
                        try:
                            if restore_from_safe_backup(target_path, safe_backup_path, dry_run=False):
                                safe_backups_used += 1
                                print(f"[INFO] Restored from safe backup: {target_path}", file=sys.stderr)
                            else:
                                safe_backups_failed += 1
                                print(f"[WARN] Failed to restore from safe backup: {target_path}", file=sys.stderr)
                        except Exception as e:
                            safe_backups_failed += 1
                            print(f"[WARN] Error restoring from safe backup {target_path}: {e}", file=sys.stderr)
            
            if safe_backups_found:
                print(f"[INFO] Safe backup restoration: {safe_backups_used} succeeded, {safe_backups_failed} failed", file=sys.stderr)
                # After using safe backups, clean up any remaining instrumented files
                # by running the regular revert to strip OPGUARD markers
                print("[INFO] Cleaning up any remaining instrumentation markers...", file=sys.stderr)
                try:
                    run_revert(dry_run=False, force=False)
                except Exception as e:
                    print(f"[WARN] Failed to clean up markers: {e}", file=sys.stderr)
            else:
                # No safe backups found, use regular revert
                print("[INFO] No safe backups found. Using regular restore...", file=sys.stderr)
                try:
                    revert_success = run_revert(dry_run=False, force=False)
                    if revert_success:
                        print("[INFO] Original files restored successfully", file=sys.stderr)
                    else:
                        print("[WARN] Some files could not be restored. Continuing anyway...", file=sys.stderr)
                except Exception as e:
                    print(f"[WARN] Failed to restore instrumented files: {e}", file=sys.stderr)
                    print("[WARN] Continuing with start...", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
        
        output_dir = self._get_output_dir()
        original_target_dir = output_dir  # Save original target before translation
        
        # Initialize local dump if requested (enabled by default, unless --disable-local-dump is set)
        local_dump_manager = None
        if not getattr(args, 'disable_local_dump', False):
            try:
                from opguard.local_dump_manager import enable_local_dump
                sync_interval = getattr(args, 'local_dump_sync_interval', 1800.0)
                local_dump_manager = enable_local_dump(original_target_dir, sync_interval)
                print(f"[INFO] Local dump enabled: shadow={local_dump_manager.local_base}, target={original_target_dir}", file=sys.stderr)
                # Update output_dir to use local shadow path for operations
                output_dir = local_dump_manager.get_local_path(output_dir)
                print(f"[INFO] Using local shadow directory: {output_dir}", file=sys.stderr)
            except Exception as e:
                raise RuntimeError("Local staging failed; refusing direct shared-storage writes") from e
        
        # Log resolved output directory and trace existence before we enforce checks
        self._log_session_context(
            "start_resolved_output",
            extra={
                "resolved_output_dir": output_dir,
                "trace_exists": self._check_trace_exists(output_dir),
            },
        )
        
        # Check if traces already exist
        if self._check_trace_exists(output_dir):
            print(f"[ERROR] Trace files already exist in {output_dir}", file=sys.stderr)
            print("[ERROR] To prevent trace contamination, please use a different directory or use 'opguard resume' to continue", file=sys.stderr)
            sys.exit(1)
        
        # Ensure directory structure
        dirs = self._ensure_output_structure(output_dir)
        print(f"[INFO] Starting OpGuard tracing", file=sys.stderr)
        print(f"[INFO] Output directory: {output_dir}", file=sys.stderr)
        print(f"[INFO] Directory structure:", file=sys.stderr)
        for name, path in dirs.items():
            print(f"  - {name}: {path}", file=sys.stderr)
        
        # Print where exec_trace and source_files will be saved
        exec_dir = dirs["exec_trace"]
        source_files_dir = os.path.join(exec_dir, "source_files")
        print(f"[INFO] Execution traces: {exec_dir}, Source files: {source_files_dir}", file=sys.stderr)
        
        # Clear trace output cache to ensure get_trace_output_dir() uses the new directory
        # This is important when switching from one output directory to another
        try:
            from opguard._running import _trace_output_cache
            _trace_output_cache.clear()
            print(f"[INFO] Cleared trace output cache to use new directory: {output_dir}", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to clear trace output cache: {e}", file=sys.stderr)
        
        # Get directory paths for shell export (don't set in current process to avoid starting tracer in CLI)
        callsite_dir = dirs["callsite_trace"]
        exec_dir = dirs["exec_trace"]
        
        # NOTE: We do NOT set these environment variables in the current Python process
        # because that would cause usercustomize.py to start the tracer in the CLI process itself.
        # Instead, we only export them in the shell commands that are output to stdout.
        
        # Generate shell activation script
        activation_script = self._generate_shell_activation_script(
            output_dir,
            enable_cupti=getattr(args, 'enable_cupti', False),
            disable_cudaapi=getattr(args, 'disable_cudaapi', False),
            disable_python=getattr(args, 'disable_python', False)
        )
        
        # Save state
        self.state["tracing_active"] = True
        self.state["start_output_dir"] = output_dir  # Local shadow path for operations
        if local_dump_manager:
            self.state["target_output_dir"] = original_target_dir  # Original target for syncing
        self.state["activation_script"] = activation_script
        self._save_state()
        
        # Check if we're in an interactive shell and can auto-activate
        is_tty = sys.stdout.isatty() and sys.stderr.isatty()
        shell = os.environ.get("SHELL", "").split("/")[-1] if os.environ.get("SHELL") else ""
        
        # Check if we can use process substitution (bash/zsh)
        can_use_process_subst = shell in ("bash", "zsh") and is_tty
        
        if can_use_process_subst and not args.no_auto_activate:
            # Method 1: Output instructions for process substitution
            print("[INFO] To automatically activate monitoring mode, run:", file=sys.stderr)
            print(f"  source <(opguard start)", file=sys.stderr)
            print("", file=sys.stderr)
            print("[INFO] Or use eval:", file=sys.stderr)
            print(f"  eval $(opguard start)", file=sys.stderr)
            print("", file=sys.stderr)
            print("[INFO] Shell commands (for eval/process substitution):", file=sys.stderr)
        
        # Always output shell commands to stdout for eval/process substitution
        # IMPORTANT: Export all OPGUARD_* environment variables so they're available to Python processes
        # Note: dirs are already created from output_dir (which is translated for local dump)
        # So callsite_dir and exec_dir are already local shadow paths if local dump is enabled
        # For OPGUARD_TRACE_OUTPUT_BASE: set to original target so get_trace_output_dir() can translate
        # For OPGUARD_EXEC_OUTPUT_DIR and OPGUARD_CALLSITE_OUTPUT_DIR: use dirs directly (already translated)
        env_output_dir = original_target_dir if local_dump_manager else output_dir
        
        print("export OPGUARD_MODE=1")
        print("export OPGUARD_ORIGINAL_PATH=\"$PATH\"")
        print(f"export OPGUARD_BIN_DIR=\"{os.path.join(output_dir, '.opguard_bin')}\"")
        print("export OPGUARD_RUNTIME_CALLSTACK=1")
        print(f"export OPGUARD_SCRIPT_LOG=\"{os.path.join(output_dir, '.opguard_scripts.log')}\"")
        print("export PATH=\"$OPGUARD_BIN_DIR:$PATH\"")
        print("> \"$OPGUARD_SCRIPT_LOG\"")
        # Export tracing environment variables
        # OPGUARD_TRACE_OUTPUT_BASE: original target (will be translated by get_trace_output_dir() if needed)
        # OPGUARD_EXEC_OUTPUT_DIR: already translated (from dirs which uses translated output_dir)
        # OPGUARD_CALLSITE_OUTPUT_DIR: already translated (from dirs which uses translated output_dir)
        print("export OPGUARD_AUTOSTART=1")
        print("export OPGUARD_CALLSITE_TRACER=1")
        print("export OPGUARD_RUNTIME_CALLSTACK=1")
        print(f"export OPGUARD_TRACE_OUTPUT_BASE=\"{env_output_dir}\"")
        print(f"export OPGUARD_CALLSITE_OUTPUT_DIR=\"{callsite_dir}\"")
        print(f"export OPGUARD_EXEC_OUTPUT_DIR=\"{exec_dir}\"")
        # Export local dump configuration if enabled
        if local_dump_manager:
            print(f"export OPGUARD_LOCAL_DUMP_ENABLED=1")
            print(f"export OPGUARD_LOCAL_DUMP_TARGET=\"{original_target_dir}\"")
            print(f"export OPGUARD_LOCAL_DUMP_SYNC_INTERVAL={local_dump_manager.sync_interval_sec}")
            # OPGUARD_LOCAL_DUMP_BASE should always be the base directory, not include any target path
            base_local = os.environ.get("OPGUARD_LOCAL_DUMP_BASE", "/tmp/opguard_local_dump")
            print(f"export OPGUARD_LOCAL_DUMP_BASE=\"{base_local}\"")
        # Export disable flags if set
        # CUPTI is disabled by default, only enable if --enable-cupti is set
        if getattr(args, 'enable_cupti', False):
            print("export OPGUARD_CALLSITE_CUPTI=1")
        else:
            print("export OPGUARD_CALLSITE_CUPTI=0")
        if getattr(args, 'disable_cudaapi', False):
            print("export OPGUARD_CALLSITE_CUDAAPI=0")
        if getattr(args, 'disable_python', False):
            print("export OPGUARD_CALLSITE_PYTHON=0")
        print("echo '[OpGuard] Entered monitoring mode automatically' >&2")
        
        # All info messages go to stderr
        if not can_use_process_subst or args.no_auto_activate:
            print(f"[INFO] To enter monitoring mode, run: source {activation_script}", file=sys.stderr)
    
    def stop(self, args):
        """Stop tracing and process instrumentation"""
        if not self.state.get("tracing_active"):
            print("[WARN] No active tracing session found", file=sys.stderr)
            return
        
        # Get the local shadow directory (where files were written)
        output_dir = self.state.get("start_output_dir") or self._get_output_dir()
        
        # Get the original target directory (where we need to sync to)
        target_output_dir = self.state.get("target_output_dir")
        
        # Initialize local dump if requested
        local_dump_manager = None
        if not getattr(args, 'disable_local_dump', False) or target_output_dir:
            try:
                from opguard.local_dump_manager import enable_local_dump, get_local_dump_manager
                # Check if already enabled (from start command)
                local_dump_manager = get_local_dump_manager()
                if local_dump_manager is None:
                    # If not enabled, use target_output_dir if available, otherwise use output_dir
                    sync_target = target_output_dir or output_dir
                    sync_interval = getattr(args, 'local_dump_sync_interval', 1800.0)
                    local_dump_manager = enable_local_dump(sync_target, sync_interval)
                # Use the manager's target_dir (which is the original target)
                if local_dump_manager:
                    target_output_dir = local_dump_manager.target_dir
                    print(f"[INFO] Local dump enabled: shadow={local_dump_manager.local_base}, target={target_output_dir}", file=sys.stderr)
            except Exception as e:
                raise RuntimeError("Local staging failed; refusing direct shared-storage writes") from e
        
        # Use local shadow directory for reading files, but sync to target
        dirs = self._ensure_output_structure(output_dir)
        
         # Log session context at stop time
        self._log_session_context(
            "stop_begin",
            extra={
                "resolved_output_dir": output_dir,
                "callsite_dir": dirs.get("callsite_trace"),
                "exec_dir": dirs.get("exec_trace"),
            },
        )
        
        print(f"[INFO] Stopping OpGuard tracing", file=sys.stderr)
        print(f"[INFO] Processing directory: {output_dir}", file=sys.stderr)
        
        # Disable tracing; numerical settings are controlled by the workload.
        os.environ["OPGUARD_AUTOSTART"] = "0"
        os.environ["OPGUARD_CALLSITE_TRACER"] = "0"
        
        # Try to stop the callsite tracer if it's already running in this process
        try:
            from opguard.callsite_tracer import load_autostart_module
            _autostart = load_autostart_module()
            if _autostart._started and _autostart._tracer is not None:
                print("[INFO] Stopping callsite tracer in current process...", file=sys.stderr)
                try:
                    _autostart._tracer.stop()
                    # Reset the started flag so it can be restarted if needed
                    _autostart._started = False
                    _autostart._tracer = None
                    print("[INFO] Callsite tracer stopped", file=sys.stderr)
                except Exception as e:
                    print(f"[WARN] Failed to stop callsite tracer: {e}", file=sys.stderr)
        except (ImportError, AttributeError):
            pass  # Module not imported yet or tracer not started, nothing to stop
        except Exception as e:
            print(f"[WARN] Could not access callsite tracer to stop it: {e}", file=sys.stderr)
        
        # Generate deactivation script to disable tracing in shell
        deactivation_script = os.path.join(output_dir, ".opguard_deactivate.sh")
        deactivation_content = f"""# OpGuard Shell Mode Deactivation Script
# Source this script to exit opguard monitoring mode
# Usage: source {deactivation_script}

# Stop tracing in subsequent Python processes.
# Only disable callsite tracing
export OPGUARD_AUTOSTART=0
export OPGUARD_CALLSITE_TRACER=0

if [ -n "$OPGUARD_MODE" ]; then
    # Restore original PATH
    if [ -n "$OPGUARD_ORIGINAL_PATH" ]; then
        export PATH="$OPGUARD_ORIGINAL_PATH"
    fi
    
    unset OPGUARD_MODE
    unset OPGUARD_ORIGINAL_PATH
    unset OPGUARD_BIN_DIR
    unset OPGUARD_SCRIPT_LOG
    unset OPGUARD_TRACE_OUTPUT_BASE
    unset OPGUARD_CALLSITE_OUTPUT_DIR
    unset OPGUARD_EXEC_OUTPUT_DIR
    
    echo "[OpGuard] Exited monitoring mode" >&2
fi
"""
        with open(deactivation_script, 'w') as f:
            f.write(deactivation_content)
        os.chmod(deactivation_script, 0o755)
        
        # Process callsite traces
        callsite_dir = dirs["callsite_trace"]
        instrumentation_success = False
        if os.path.exists(callsite_dir) and any(Path(callsite_dir).rglob("*.json*")):
            print(f"[INFO] Processing callsite traces from: {callsite_dir}", file=sys.stderr)
            
            # Run preprocessing pipeline
            # IMPORTANT: Capture stdout and redirect to stderr to prevent it from being executed as shell commands
            # Use a context manager approach to ensure all output is captured
            pipeline = PreprocessingPipeline()
            original_stdout = sys.stdout
            try:
                # Create a wrapper that redirects all writes to stderr
                class StderrRedirect:
                    def write(self, s):
                        sys.stderr.write(s)
                        sys.stderr.flush()
                    def flush(self):
                        sys.stderr.flush()
                    def __getattr__(self, name):
                        return getattr(sys.stderr, name)
                
                # Redirect stdout to stderr for preprocessing and instrumentation
                sys.stdout = StderrRedirect()
                try:
                    summary = pipeline.run_pipeline(
                        inputs=[callsite_dir],
                        output_dir=dirs["instrumentation"]
                    )
                    
                    # Find the generated YAML
                    final_yaml = summary.get("sync_yaml") or summary.get("filtered_yaml") or summary.get("final_combined_yaml")
                    if final_yaml:
                        print(f"[INFO] Generated instrumentation config: {final_yaml}", file=sys.stderr)
                        
                        # Sync after callsite processing
                        if local_dump_manager:
                            local_dump_manager.sync_step("callsite")
                        
                        # Run instrumentation
                        print("[INFO] Running instrumentation...", file=sys.stderr)
                        from opguard.instrument_all_files import run_instrumentation
                        success = run_instrumentation(config_file=final_yaml)
                        if success:
                            print("[INFO] Instrumentation completed successfully", file=sys.stderr)
                            instrumentation_success = True
                        else:
                            print("[WARN] Instrumentation completed with warnings", file=sys.stderr)
                            instrumentation_success = True  # Still consider it success for replay
                        
                        # Sync after instrumentation
                        if local_dump_manager:
                            local_dump_manager.sync_step("instrumentation")
                finally:
                    # Restore stdout before outputting shell commands
                    sys.stdout = original_stdout
            except Exception as e:
                # Make sure stdout is restored even on error
                sys.stdout = original_stdout
                print(f"[ERROR] Failed to process callsite traces: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        else:
            print(f"[WARN] No callsite traces found in {callsite_dir}", file=sys.stderr)
        
        # Final sync if local dump is enabled
        if local_dump_manager:
            print(f"[INFO] Performing final sync to target directory: {target_output_dir}", file=sys.stderr)
            local_dump_manager.sync_all(force=True, exact_match=True)
            print(f"[INFO] Final sync completed. Files are now in: {target_output_dir}", file=sys.stderr)
        
        # Update state
        self.state["tracing_active"] = False
        self._save_state()
        
        print("[INFO] OpGuard tracing stopped and processed", file=sys.stderr)
        if local_dump_manager and target_output_dir:
            print(f"[INFO] Local shadow directory: {output_dir}", file=sys.stderr)
            print(f"[INFO] Target directory (final output): {target_output_dir}", file=sys.stderr)
        else:
            print(f"[INFO] Output directory: {output_dir}", file=sys.stderr)
        
        # Cleanup processes to prevent interference from previous runs
        # NOTE: Only cleanup specific processes that are safe to kill (not python3 which would kill training)
        # print("\n[INFO] Cleaning up processes from previous runs...")
        # cleanup_commands = [
        #     ("pkill", "redis-server"),
        # ]
        
        # for cmd in cleanup_commands:
        #     try:
        #         result = subprocess.run(
        #             cmd,
        #             capture_output=True,
        #             text=True,
        #             timeout=5
        #         )
        #         if result.returncode == 0:
        #             print(f"[INFO] Successfully ran: {' '.join(cmd)}")
        #         elif result.returncode == 1:
        #             # pkill returns 1 if no processes were found (which is fine)
        #             print(f"[INFO] No processes found for: {' '.join(cmd)}")
        #         else:
        #             print(f"[WARN] Command {' '.join(cmd)} exited with code {result.returncode}")
        #             if result.stderr:
        #                 print(f"[WARN] Error: {result.stderr}")
        #     except subprocess.TimeoutExpired:
        #         print(f"[WARN] Command {' '.join(cmd)} timed out")
        #     except Exception as e:
        #         print(f"[WARN] Failed to run {' '.join(cmd)}: {e}")
        
        # Inform user about next steps
        if instrumentation_success:
            print("\n" + "=" * 80, file=sys.stderr)
            print("[INFO] Instrumentation completed successfully", file=sys.stderr)
            print("[INFO] To run the instrumented code and collect execution traces, use:", file=sys.stderr)
            print("[INFO]   opguard replay", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            
            # Get tracked scripts to suggest what to replay
            tracked_scripts = self._get_tracked_scripts(output_dir)
            if tracked_scripts:
                print(f"[INFO] Detected script(s) that can be replayed:", file=sys.stderr)
                for script in tracked_scripts:
                    print(f"  - {script}", file=sys.stderr)
                print(f"[INFO] Run: opguard replay", file=sys.stderr)
            elif self.state.get("last_script") and os.path.exists(self.state.get("last_script")):
                print(f"[INFO] Previous script found: {self.state.get('last_script')}", file=sys.stderr)
                print(f"[INFO] Run: opguard replay", file=sys.stderr)
            else:
                print("[INFO] Use 'opguard replay' to run the instrumented code", file=sys.stderr)
        else:
            print("[INFO] Use 'opguard replay' to run the instrumented code", file=sys.stderr)
        
        # IMPORTANT: Output shell commands LAST, after all processing is complete
        # This ensures no other output goes to stdout before the shell commands
        deactivation_script = os.path.join(output_dir, ".opguard_deactivate.sh")
        
        # Check if we're in an interactive shell and can use process substitution
        is_tty = sys.stdout.isatty() and sys.stderr.isatty()
        shell = os.environ.get("SHELL", "").split("/")[-1] if os.environ.get("SHELL") else ""
        can_use_process_subst = shell in ("bash", "zsh") and is_tty
        
        if can_use_process_subst:
            # Method 1: Output instructions for process substitution
            print("[INFO] To automatically deactivate monitoring mode, run:", file=sys.stderr)
            print(f"  source <(opguard stop)", file=sys.stderr)
            print("", file=sys.stderr)
            print("[INFO] Or use eval:", file=sys.stderr)
            print(f"  eval $(opguard stop)", file=sys.stderr)
            print("", file=sys.stderr)
            print("[INFO] Shell commands (for eval/process substitution):", file=sys.stderr)
        else:
            print(f"\n[INFO] To disable callsite tracing in your current shell, run:", file=sys.stderr)
            print(f"  source {deactivation_script}", file=sys.stderr)
        
        # Output deactivation commands to stdout for eval/process substitution
        # IMPORTANT: This must be the LAST thing output to stdout - all info messages go to stderr
        # Stop tracing in future processes.
        print("# Stop OpGuard autostart")
        print("export OPGUARD_AUTOSTART=0")
        print("export OPGUARD_CALLSITE_TRACER=0")
        print("# Restore original PATH if OPGUARD_MODE is set")
        print("if [ -n \"$OPGUARD_MODE\" ]; then")
        print("    if [ -n \"$OPGUARD_ORIGINAL_PATH\" ]; then")
        print("        export PATH=\"$OPGUARD_ORIGINAL_PATH\"")
        print("    fi")
        print("    unset OPGUARD_MODE")
        print("    unset OPGUARD_ORIGINAL_PATH")
        print("    unset OPGUARD_BIN_DIR")
        print("    unset OPGUARD_SCRIPT_LOG")
        print("    unset OPGUARD_TRACE_OUTPUT_BASE")
        print("    unset OPGUARD_CALLSITE_OUTPUT_DIR")
        print("    unset OPGUARD_EXEC_OUTPUT_DIR")
        print("    echo '[OpGuard] Exited monitoring mode' >&2")
        print("fi")
    
    def instrument(self, args):
        """Instrument code from a YAML, or merge a callsite dump folder then instrument."""
        config_file = args.config
        source_dir = getattr(args, "dir", None)

        if source_dir and os.path.isfile(source_dir) and source_dir.endswith((".yaml", ".yml")):
            config_file = source_dir
            source_dir = None

        if source_dir:
            source_dir = os.path.abspath(os.path.expanduser(source_dir))
            callsite_input = self._resolve_callsite_input_dir(source_dir)
            if os.path.isfile(callsite_input) and callsite_input.endswith((".yaml", ".yml")):
                config_file = callsite_input
            else:
                src = Path(source_dir)
                if src.is_file():
                    output_dir = str(src.parent)
                elif Path(callsite_input).name == "callsite_trace":
                    output_dir = str(Path(callsite_input).parent)
                else:
                    output_dir = source_dir
                dirs = self._ensure_output_structure(output_dir)
                existing = None if getattr(args, "force_preprocess", False) else self._find_instrumentation_yaml(dirs["instrumentation"])
                if existing:
                    print(f"[INFO] Reusing existing instrumentation YAML: {existing}", file=sys.stderr)
                    config_file = existing
                else:
                    config_file = self._preprocess_callsite_dir(
                        callsite_input,
                        output_dir,
                        base_dir=getattr(args, "base_dir", None),
                    )
                self.state["output_dir"] = output_dir
                self.state["target_output_dir"] = output_dir
                self._save_state()

        # If no config specified, try to find the most recent one from the output directory
        if not config_file:
            output_dir = self._get_output_dir()
            dirs = self._ensure_output_structure(output_dir)
            instrumentation_dir = dirs["instrumentation"]
            config_file = self._find_instrumentation_yaml(instrumentation_dir)
            if config_file:
                print(f"[INFO] Found instrumentation config: {config_file}", file=sys.stderr)
            else:
                # Last resort: merge callsite_trace in the same output dir
                callsite_dir = dirs["callsite_trace"]
                if os.path.isdir(callsite_dir) and any(Path(callsite_dir).rglob("*.json*")):
                    config_file = self._preprocess_callsite_dir(
                        callsite_dir,
                        output_dir,
                        base_dir=getattr(args, "base_dir", None),
                    )
                else:
                    print(f"[ERROR] No instrumentation YAML and no callsite traces in: {output_dir}", file=sys.stderr)
                    print("[ERROR] Pass the dump folder: opguard instrument <callsite_dir>", file=sys.stderr)
                    sys.exit(1)
        else:
            if not os.path.exists(config_file):
                print(f"[ERROR] Config file not found: {config_file}", file=sys.stderr)
                sys.exit(1)
            if not os.path.isabs(config_file):
                config_file = os.path.abspath(config_file)
        
        print(f"[INFO] Instrumenting code using config: {config_file}", file=sys.stderr)
        
        # Run instrumentation
        from opguard.instrument_all_files import run_instrumentation
        success = run_instrumentation(config_file=config_file)
        
        if success:
            print("[INFO] Instrumentation completed successfully", file=sys.stderr)
            if not getattr(args, "no_emit_env", False):
                # Resolve output directory for env exports
                output_dir = self._infer_output_dir_from_config(config_file)
                if not output_dir:
                    output_dir = self.state.get("target_output_dir") or self._get_output_dir()

                # Ensure output directory structure exists (for exec_trace, etc.)
                dirs = self._ensure_output_structure(output_dir)

                self._emit_instrument_env_exports(output_dir, dirs, args, config_file)
        else:
            print("[ERROR] Instrumentation completed with errors", file=sys.stderr)
            sys.exit(1)
    
    def clear(self, args):
        """Clear all instrumentation and clean up OpGuard artifacts"""
        print("[INFO] Clearing OpGuard instrumentation and artifacts...", file=sys.stderr)
        instrumentation_only = getattr(args, 'instrumentation_only', False)
        
        # 1. Uninstrument all files (force mode to override hash mismatches)
        print("[INFO] Step 1: Uninstrumenting all files...", file=sys.stderr)
        from opguard.instrument_all_files import run_revert, load_manifest
        try:
            manifest = load_manifest()
            manifest_items = manifest.get("items", {})
            if manifest_items:
                print(f"[INFO] Found {len(manifest_items)} instrumented file(s) to restore", file=sys.stderr)
                revert_success = run_revert(dry_run=False, force=True)
                if revert_success:
                    print("[INFO] ✅ All files uninstrumented successfully", file=sys.stderr)
                else:
                    print("[WARN] ⚠️  Some files could not be restored, but continuing cleanup...", file=sys.stderr)
            else:
                print("[INFO] No instrumented files found", file=sys.stderr)
        except Exception as e:
            print(f"[WARN] Failed to uninstrument files: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        
        if not instrumentation_only:
            # 2. Clean up safe backups from system temp
            print("[INFO] Step 2: Cleaning up safe backups...", file=sys.stderr)
            try:
                from opguard.instrument_all_files import get_safe_backup_root
                safe_backup_root = get_safe_backup_root()
                if safe_backup_root.exists():
                    backup_count = len(list(safe_backup_root.glob("*.orig")))
                    if backup_count > 0:
                        print(f"[INFO] Found {backup_count} safe backup file(s)", file=sys.stderr)
                        if getattr(args, 'yes', False) or input(f"Delete all safe backups in {safe_backup_root}? [y/N]: ").lower() == 'y':
                            shutil.rmtree(safe_backup_root)
                            print(f"[INFO] ✅ Removed safe backup directory: {safe_backup_root}", file=sys.stderr)
                        else:
                            print("[INFO] Skipped safe backup cleanup", file=sys.stderr)
                    else:
                        print("[INFO] No safe backups found", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to clean safe backups: {e}", file=sys.stderr)
            
            # 3. Clean up local .opguard directory
            print("[INFO] Step 3: Cleaning up local .opguard directory...", file=sys.stderr)
            try:
                from opguard.instrument_all_files import OPGUARD_DIR
                if OPGUARD_DIR.exists():
                    if getattr(args, 'yes', False) or input(f"Delete .opguard directory ({OPGUARD_DIR})? [y/N]: ").lower() == 'y':
                        shutil.rmtree(OPGUARD_DIR)
                        print(f"[INFO] ✅ Removed .opguard directory: {OPGUARD_DIR}", file=sys.stderr)
                    else:
                        print("[INFO] Skipped .opguard directory cleanup", file=sys.stderr)
                else:
                    print("[INFO] No .opguard directory found", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to clean .opguard directory: {e}", file=sys.stderr)
            
            # 4. Clear CLI state (optional - only if user wants)
            if getattr(args, 'clear_state', False):
                print("[INFO] Step 4: Clearing CLI state...", file=sys.stderr)
                try:
                    if self.state_file.exists():
                        self.state_file.unlink()
                        print(f"[INFO] ✅ Removed CLI state file: {self.state_file}", file=sys.stderr)
                    else:
                        print("[INFO] No CLI state file found", file=sys.stderr)
                except Exception as e:
                    print(f"[WARN] Failed to clear CLI state: {e}", file=sys.stderr)
            
            # 5. Stop any running tracers in current process
            print("[INFO] Step 5: Stopping any running tracers...", file=sys.stderr)
            try:
                from opguard.callsite_tracer import load_autostart_module
                _autostart = load_autostart_module()
                if _autostart._started and _autostart._tracer is not None:
                    _autostart._tracer.stop()
                    _autostart._started = False
                    _autostart._tracer = None
                    print("[INFO] ✅ Stopped callsite tracer", file=sys.stderr)
                else:
                    print("[INFO] No active tracer found", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to stop tracer: {e}", file=sys.stderr)
            
            # 6. Reset OPGUARD_AUTOSTART
            print("[INFO] Step 6: Resetting OPGUARD_AUTOSTART...", file=sys.stderr)
            os.environ["OPGUARD_AUTOSTART"] = "0"
            os.environ["OPGUARD_CALLSITE_TRACER"] = "0"
            print("[INFO] ✅ Reset OPGUARD_AUTOSTART=0 (autostart disabled for future processes)", file=sys.stderr)
            
            # 7. Disable local dump if enabled
            print("[INFO] Step 7: Disabling local dump and cleaning up...", file=sys.stderr)
            try:
                from opguard.local_dump_manager import disable_local_dump, get_local_dump_manager
                local_dump_manager = get_local_dump_manager()
                if local_dump_manager is not None:
                    disable_local_dump()
                    print("[INFO] ✅ Local dump manager disabled and cleaned up", file=sys.stderr)
                else:
                    print("[INFO] No local dump manager found", file=sys.stderr)
            except ImportError:
                print("[INFO] Local dump manager not available", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to disable local dump: {e}", file=sys.stderr)
            
            # 7. Unset all OPGUARD environment variables
            print("[INFO] Step 7: Unsetting all OPGUARD environment variables...", file=sys.stderr)
            # Find all environment variables that start with OPGUARD_
            opguard_env_vars = [var for var in os.environ.keys() if var.startswith("OPGUARD_")]
            unset_count = 0
            for var in opguard_env_vars:
                del os.environ[var]
                unset_count += 1
            if unset_count > 0:
                print(f"[INFO] ✅ Unset {unset_count} OPGUARD environment variable(s) in current process: {', '.join(sorted(opguard_env_vars))}", file=sys.stderr)
            else:
                print("[INFO] No OPGUARD environment variables found to unset in current process", file=sys.stderr)
        
        print("\n" + "=" * 80, file=sys.stderr)
        print("[INFO] ✅ OpGuard clear completed!", file=sys.stderr)
        if instrumentation_only:
            print("[INFO] All instrumentation has been removed. Backups, .opguard directory, and CLI state were left intact.", file=sys.stderr)
        else:
            print("[INFO] All instrumentation has been removed and artifacts cleaned up.", file=sys.stderr)
            print("[INFO] Local dump has been disabled and all OPGUARD environment variables have been unset in current process.", file=sys.stderr)
            print("[INFO] To unset OPGUARD variables in your shell, run:", file=sys.stderr)
            print("  eval $(opguard clear)", file=sys.stderr)
            print("  # or", file=sys.stderr)
            print("  source <(opguard clear)", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        # Output shell commands to stdout to unset all OPGUARD_* variables in the shell
        # IMPORTANT: This must be the LAST thing output to stdout - all info messages go to stderr
        # This allows users to run: eval $(opguard clear) or source <(opguard clear)
        print("# Unset all OPGUARD_* environment variables")
        print("# Restore original PATH if OPGUARD_MODE is set")
        print("if [ -n \"$OPGUARD_MODE\" ]; then")
        print("    if [ -n \"$OPGUARD_ORIGINAL_PATH\" ]; then")
        print("        export PATH=\"$OPGUARD_ORIGINAL_PATH\"")
        print("    fi")
        print("fi")
        print("# Unset all OPGUARD_* variables")
        # Generate unset commands for all possible OPGUARD_* variables
        # We'll unset all of them to be safe, even if some don't exist
        opguard_vars_to_unset = [
            "OPGUARD_MODE",
            "OPGUARD_ORIGINAL_PATH",
            "OPGUARD_BIN_DIR",
            "OPGUARD_SCRIPT_LOG",
            "OPGUARD_AUTOSTART",
            "OPGUARD_CALLSITE_TRACER",
            "OPGUARD_RUNTIME_CALLSTACK",
            "OPGUARD_TRACE_OUTPUT_BASE",
            "OPGUARD_CALLSITE_OUTPUT_DIR",
            "OPGUARD_EXEC_OUTPUT_DIR",
            "OPGUARD_LOCAL_DUMP_ENABLED",
            "OPGUARD_LOCAL_DUMP_TARGET",
            "OPGUARD_LOCAL_DUMP_SYNC_INTERVAL",
            "OPGUARD_LOCAL_DUMP_BASE",
            "OPGUARD_CALLSITE_CUPTI",
            "OPGUARD_CALLSITE_CUDAAPI",
            "OPGUARD_CALLSITE_PYTHON",
            "OPGUARD_IN_BASH_SCRIPT",
            "OPGUARD_TRACE_OUTDIR",
        ]
        for var in opguard_vars_to_unset:
            print(f"unset {var} 2>/dev/null || true")
        # Also unset any other OPGUARD_* variables that might exist
        # This handles any variables we might have missed
        print("# Unset any other OPGUARD_* variables")
        print("for var in $(env | grep '^OPGUARD_' | cut -d= -f1); do")
        print("    unset \"$var\" 2>/dev/null || true")
        print("done")
        print("echo '[OpGuard] All OPGUARD environment variables cleared' >&2")
    
    def sync(self, args):
        """Sync files from local shadow directory to target directory"""
        try:
            from opguard.local_dump_manager import get_local_dump_manager
            local_dump_manager = get_local_dump_manager()
            
            if local_dump_manager is None:
                # Try to initialize from environment variables if available
                if os.environ.get("OPGUARD_LOCAL_DUMP_ENABLED") == "1":
                    target_dir = os.environ.get("OPGUARD_LOCAL_DUMP_TARGET")
                    sync_interval = float(os.environ.get("OPGUARD_LOCAL_DUMP_SYNC_INTERVAL", "1800.0"))
                    if target_dir:
                        from opguard.local_dump_manager import enable_local_dump
                        local_dump_manager = enable_local_dump(target_dir, sync_interval)
                        print(f"[INFO] Local dump manager initialized from environment variables", file=sys.stderr)
                    else:
                        print("[ERROR] OPGUARD_LOCAL_DUMP_ENABLED=1 but OPGUARD_LOCAL_DUMP_TARGET not set", file=sys.stderr)
                        sys.exit(1)
                else:
                    print("[ERROR] Local dump is not enabled. Local dump is enabled by default; use 'opguard sync' to manually sync.", file=sys.stderr)
                    sys.exit(1)
            
            print("[INFO] Starting manual sync from local shadow to target directory...", file=sys.stderr)
            print(f"[INFO] Local shadow: {local_dump_manager.local_base}", file=sys.stderr)
            print(f"[INFO] Target directory: {local_dump_manager.target_dir}", file=sys.stderr)
            
            # Perform the sync with exact match mode
            local_dump_manager.sync_all(force=True, exact_match=True)
            
            print("[INFO] ✅ Sync completed successfully", file=sys.stderr)
            local_dump_manager.log_stats()
            
        except ImportError:
            print("[ERROR] Local dump manager not available", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Failed to sync: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    
    def _replay_internal(self, script: str, local_dump_manager=None, target_output_dir=None):
        """Internal replay logic - simply rerun the recorded command"""
        # Save script to state for future reference
        self.state["last_script"] = script
        self._save_state()
        
        output_dir = self.state.get("start_output_dir") or self._get_output_dir()
        dirs = self._ensure_output_structure(output_dir)
        
        print(f"[INFO] Replaying recorded command: {script}", file=sys.stderr)
        print(f"[INFO] Output directory: {output_dir}", file=sys.stderr)
        
        # Set environment for runtime tracing (not callsite tracing)
        os.environ["OPGUARD_AUTOSTART"] = "1"  # Enable autostart for runtime tracer
        os.environ["OPGUARD_CALLSITE_TRACER"] = "0"  # Disable callsite tracer (already done)
        os.environ["OPGUARD_RUNTIME_CALLSTACK"] = "1"  # Enable runtime callstack tracing
        os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = output_dir
        os.environ["OPGUARD_EXEC_OUTPUT_DIR"] = dirs["exec_trace"]
        
        # Detect script type and get execution command
        interpreter, cmd_args = self._detect_script_type(script)
        exec_cmd = [interpreter] + cmd_args
        
        print(f"[INFO] Executing: {' '.join(exec_cmd)}", file=sys.stderr)
        
        # Execute the command
        result = subprocess.run(
            exec_cmd,
            check=True,
            env=os.environ.copy()
        )
        
        # Sync after runtime tracing
        if local_dump_manager:
            local_dump_manager.sync_step("runtime_tracing")
            local_dump_manager.sync_all(force=True, exact_match=True)
            print(f"[INFO] Files synced to target directory: {target_output_dir}", file=sys.stderr)
        
        print(f"[INFO] Replay completed successfully", file=sys.stderr)
        if local_dump_manager and target_output_dir:
            print(f"[INFO] Local shadow directory: {dirs['exec_trace']}", file=sys.stderr)
            print(f"[INFO] Target directory (final output): {target_output_dir}/exec_trace", file=sys.stderr)
        else:
            print(f"[INFO] Execution traces saved to: {dirs['exec_trace']}", file=sys.stderr)
    
    def replay(self, args):
        """Run instrumented user code to collect execution traces (uses script from recording phase)"""
        # Get the local shadow directory (where files were written)
        output_dir = self.state.get("start_output_dir") or self._get_output_dir()
        
        # Get the original target directory (where we need to sync to)
        target_output_dir = self.state.get("target_output_dir")
        
        # Initialize local dump if requested
        local_dump_manager = None
        if not getattr(args, 'disable_local_dump', False) or target_output_dir:
            try:
                from opguard.local_dump_manager import enable_local_dump, get_local_dump_manager
                # Check if already enabled (from start command)
                local_dump_manager = get_local_dump_manager()
                if local_dump_manager is None:
                    # If not enabled, use target_output_dir if available, otherwise use output_dir
                    sync_target = target_output_dir or output_dir
                    sync_interval = getattr(args, 'local_dump_sync_interval', 1800.0)
                    local_dump_manager = enable_local_dump(sync_target, sync_interval)
                # Use the manager's target_dir (which is the original target)
                if local_dump_manager:
                    target_output_dir = local_dump_manager.target_dir
                    print(f"[INFO] Local dump enabled: shadow={local_dump_manager.local_base}, target={target_output_dir}", file=sys.stderr)
            except Exception as e:
                raise RuntimeError("Local staging failed; refusing direct shared-storage writes") from e
        
        # Get tracked scripts from the recording phase
        tracked_scripts = self._get_tracked_scripts(output_dir)
        
        # Determine which script to replay
        script = None
        if tracked_scripts:
            # Use the first (or most recent) tracked script
            script = tracked_scripts[0]
            if len(tracked_scripts) > 1:
                print(f"[INFO] Multiple scripts detected: {tracked_scripts}")
                print(f"[INFO] Using first script: {script}")
            else:
                print(f"[INFO] Using recorded script: {script}")
        elif self.state.get("last_script") and os.path.exists(self.state.get("last_script")):
            script = self.state.get("last_script")
            print(f"[INFO] Using script from previous session: {script}")
        else:
            print("[ERROR] No script found from recording phase")
            print("[ERROR] Make sure you ran 'opguard start' and executed a script before running 'opguard stop'")
            print("[ERROR] The script should have been tracked during the recording phase")
            sys.exit(1)
        
        try:
            self._replay_internal(script, local_dump_manager, target_output_dir)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Replay failed with exit code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print(f"[INFO] Replay interrupted by user")
            sys.exit(130)
        except FileNotFoundError as e:
            print(f"[ERROR] Interpreter not found")
            print(f"[ERROR] Please ensure the interpreter is installed and in PATH")
            sys.exit(127)
        except Exception as e:
            print(f"[ERROR] Replay failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def resume(self, args):
        """Resume tracing (bypass existence check)"""
        print("[WARN] Resume mode: continuing in existing directory (not recommended)")
        output_dir = self._get_output_dir()
        dirs = self._ensure_output_structure(output_dir)
        
        os.environ["OPGUARD_AUTOSTART"] = "1"
        os.environ["OPGUARD_CALLSITE_TRACER"] = "1"
        os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = output_dir
        os.environ["OPGUARD_CALLSITE_OUTPUT_DIR"] = dirs["callsite_trace"]
        os.environ["OPGUARD_EXEC_OUTPUT_DIR"] = dirs["exec_trace"]
        
        self.state["tracing_active"] = True
        self.state["start_output_dir"] = output_dir
        self._save_state()
        
        print("[INFO] OpGuard tracing resumed")
    
    def run(self, args):
        """Quick mode: run two scripts and generate perfetto comparison"""
        if len(args.scripts) < 1:
            print("[ERROR] At least one script required for 'run' command")
            sys.exit(1)
        
        scripts = args.scripts
        output_base = self._get_output_dir()
        
        # Initialize local dump if requested (enabled by default, unless --disable-local-dump is set)
        local_dump_manager = None
        if not getattr(args, 'disable_local_dump', False):
            try:
                from opguard.local_dump_manager import enable_local_dump
                sync_interval = getattr(args, 'local_dump_sync_interval', 1800.0)
                local_dump_manager = enable_local_dump(output_base, sync_interval)
                print(f"[INFO] Local dump enabled: shadow={local_dump_manager.local_base}, target={output_base}", file=sys.stderr)
                # Update output_base to use local shadow path
                output_base = local_dump_manager.get_local_path(output_base)
            except Exception as e:
                raise RuntimeError("Local staging failed; refusing direct shared-storage writes") from e
        
        print("=" * 80)
        print("OpGuard Quick Mode")
        print("=" * 80)
        print(f"Scripts: {scripts}")
        print(f"Output base: {output_base}")
        
        run_dirs = []
        
        # Run each script
        for idx, script in enumerate(scripts, 1):
            print(f"\n[Run {idx}/{len(scripts)}] Executing: {script}")
            
            # Create unique output directory for this run
            if len(scripts) > 1:
                run_dir = os.path.join(output_base, f"run_{idx}")
            else:
                run_dir = output_base
            
            dirs = self._ensure_output_structure(run_dir)
            run_dirs.append((run_dir, dirs))
            
            # Set environment
            os.environ["OPGUARD_AUTOSTART"] = "1"
            os.environ["OPGUARD_CALLSITE_TRACER"] = "1"
            os.environ["OPGUARD_RUNTIME_CALLSTACK"] = "1"
            os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = run_dir
            os.environ["OPGUARD_CALLSITE_OUTPUT_DIR"] = dirs["callsite_trace"]
            os.environ["OPGUARD_EXEC_OUTPUT_DIR"] = dirs["exec_trace"]
            # Set disable flags if specified
            if getattr(args, 'disable_cupti', False):
                os.environ["OPGUARD_CALLSITE_CUPTI"] = "0"
            if getattr(args, 'disable_cudaapi', False):
                os.environ["OPGUARD_CALLSITE_CUDAAPI"] = "0"
            if getattr(args, 'disable_python', False):
                os.environ["OPGUARD_CALLSITE_PYTHON"] = "0"
            
            # Detect script type and get execution command
            interpreter, cmd_args = self._detect_script_type(script)
            exec_cmd = [interpreter] + cmd_args
            
            print(f"[Run {idx}] Executing: {' '.join(exec_cmd)}")
            
            # Execute script
            try:
                result = subprocess.run(
                    exec_cmd,
                    check=True,
                    env=os.environ.copy()
                )
                print(f"[Run {idx}] Completed successfully")
            except subprocess.CalledProcessError as e:
                print(f"[Run {idx}] Failed with exit code {e.returncode}")
                sys.exit(e.returncode)
            except KeyboardInterrupt:
                print(f"[Run {idx}] Interrupted by user")
                sys.exit(130)
            except FileNotFoundError as e:
                print(f"[Run {idx}] Interpreter not found: {interpreter}")
                print(f"[Run {idx}] Please ensure the interpreter is installed and in PATH")
                sys.exit(127)
        
        # Process callsite traces and instrument
        print("\n[Post-processing] Processing callsite traces...")
        for idx, (run_dir, dirs) in enumerate(run_dirs, 1):
            callsite_dir = dirs["callsite_trace"]
            if os.path.exists(callsite_dir) and any(Path(callsite_dir).rglob("*.json*")):
                pipeline = PreprocessingPipeline()
                try:
                    summary = pipeline.run_pipeline(
                        inputs=[callsite_dir],
                        output_dir=dirs["instrumentation"]
                    )
                    final_yaml = summary.get("sync_yaml") or summary.get("filtered_yaml") or summary.get("final_combined_yaml")
                    if final_yaml:
                        print(f"[Run {idx}] Generated instrumentation: {final_yaml}")
                        # Sync after callsite processing
                        if local_dump_manager:
                            local_dump_manager.sync_step("callsite")
                        from opguard.instrument_all_files import run_instrumentation
                        run_instrumentation(config_file=final_yaml)
                        # Sync after instrumentation
                        if local_dump_manager:
                            local_dump_manager.sync_step("instrumentation")
                except Exception as e:
                    print(f"[WARN] Run {idx} post-processing failed: {e}")
        
        # Final sync if local dump is enabled
        if local_dump_manager:
            local_dump_manager.sync_all(force=True, exact_match=True)
        
        # Generate perfetto comparison if multiple runs
        if len(run_dirs) >= 2:
            print("\n[Comparison] Generating perfetto trace...")
            # Use first run's directory for report output
            first_run_dir, first_run_dirs = run_dirs[0]
            self._generate_perfetto_comparison(run_dirs, first_run_dirs["report"], args)
        else:
            print("\n[INFO] Single run completed. Use 'opguard compare' to compare with another run.")
        
        print("\n" + "=" * 80)
        print("Quick mode completed")
        print("=" * 80)
    
    def _generate_perfetto_comparison(self, run_dirs: List, report_base: str, args):
        """Generate perfetto comparison between runs"""
        if len(run_dirs) < 2:
            return
        
        # For now, compare first two runs
        run1_dir, run1_dirs = run_dirs[0]
        run2_dir, run2_dirs = run_dirs[1]
        
        # Find exec_trace directories
        exec1 = run1_dirs["exec_trace"]
        exec2 = run2_dirs["exec_trace"]
        
        # Find GPU directories (default to gpu_000)
        gpu_pattern = "gpu_*"
        gpu_dirs1 = list(Path(exec1).glob(gpu_pattern))
        gpu_dirs2 = list(Path(exec2).glob(gpu_pattern))
        
        if not gpu_dirs1 or not gpu_dirs2:
            print("[WARN] No GPU trace directories found, skipping perfetto generation")
            return
        
        # Use first GPU or specified GPU
        gpu_spec = args.gpu if hasattr(args, 'gpu') else "0"
        gpu_indices = self._parse_gpu_spec(gpu_spec, len(gpu_dirs1))
        
        # Use run directory names for report
        run1_name = Path(run1_dir).name
        run2_name = Path(run2_dir).name
        report_dir = Path(report_base) / f"{run1_name}_vs_{run2_name}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        for gpu_idx in gpu_indices:
            gpu_name = f"gpu_{gpu_idx:03d}"
            gpu_dir1 = Path(exec1) / gpu_name
            gpu_dir2 = Path(exec2) / gpu_name
            
            if not gpu_dir1.exists() or not gpu_dir2.exists():
                print(f"[WARN] GPU {gpu_name} not found in both runs, skipping")
                continue
            
            output_file = report_dir / f"perfetto_{gpu_name}.json.gz"
            
            print(f"[Perfetto] Comparing {gpu_name}: {gpu_dir1} vs {gpu_dir2}")
            
            try:
                from opguard.trace_export.perfetto.build import build_perfetto
                build_perfetto(
                    str(gpu_dir1),
                    str(gpu_dir2),
                    str(output_file)
                )
                print(f"[Perfetto] Generated: {output_file}")
            except Exception as e:
                print(f"[ERROR] Perfetto generation failed for {gpu_name}: {e}")
                import traceback
                traceback.print_exc()
    
    def _parse_gpu_spec(self, spec: str, max_gpus: int) -> List[int]:
        """Parse GPU specification like '0', '0,2,3', '1-8', 'all'"""
        if spec.lower() == "all":
            return list(range(max_gpus))
        
        indices = []
        for part in spec.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                indices.extend(range(start, end + 1))
            else:
                indices.append(int(part))
        
        return sorted(set(indices))
    
    def compare(self, args):
        """Restore immutable upload batches locally before invoking analysis."""
        from opguard.local_sync.materialize import has_bundles, materialize
        with tempfile.TemporaryDirectory(prefix="opguard_snapshot_") as scratch:
            resolved = argparse.Namespace(**vars(args))
            resolved._report_dir1 = os.path.abspath(args.dir1)
            resolved._report_dir2 = os.path.abspath(args.dir2)
            for name in ("dir1", "dir2"):
                source = getattr(args, name)
                if has_bundles(source):
                    setattr(resolved, name, materialize(source, os.path.join(scratch, name)))
            return self._compare_local(resolved)

    def _compare_local(self, args):
        """Compare two trace directories using perfetto"""
        dir1 = args.dir1
        dir2 = args.dir2
        gpu_spec = args.gpu if args.gpu else "0"
        
        # Load ignore config from config.yaml
        import yaml
        ignore_config = {}
        config_path = getattr(args, 'ignore_config', None)
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                    ignore_config = cfg.get("inspector_ignore_fields", {}) or {}
                    if ignore_config:
                        print(f"[Compare] Loaded ignore config from {config_path}")
                        print(f"[Compare]   Ignoring fields for {len(ignore_config)} function patterns")
            except Exception as e:
                print(f"[WARN] Failed to load ignore config from {config_path}: {e}")
        else:
            print(f"[WARN] Ignore config file not found: {config_path}")
        
        # Resolve directories relative to current working directory
        if not os.path.isabs(dir1):
            dir1 = os.path.abspath(dir1)
        if not os.path.isabs(dir2):
            dir2 = os.path.abspath(dir2)
        
        if not os.path.exists(dir1):
            print(f"[ERROR] Directory not found: {dir1}")
            sys.exit(1)
        if not os.path.exists(dir2):
            print(f"[ERROR] Directory not found: {dir2}")
            sys.exit(1)
        
        # Find exec_trace or use directory directly
        exec1 = os.path.join(dir1, "exec_trace") if os.path.exists(os.path.join(dir1, "exec_trace")) else dir1
        exec2 = os.path.join(dir2, "exec_trace") if os.path.exists(os.path.join(dir2, "exec_trace")) else dir2
        
        # Find GPU directories
        gpu_pattern = "gpu_*"
        gpu_dirs1 = list(Path(exec1).glob(gpu_pattern))
        gpu_dirs2 = list(Path(exec2).glob(gpu_pattern))
        
        if not gpu_dirs1 or not gpu_dirs2:
            print("[ERROR] No GPU trace directories found")
            sys.exit(1)
        
        available = sorted({int(p.name[4:]) for p in gpu_dirs1 if p.name[4:].isdigit()} &
                           {int(p.name[4:]) for p in gpu_dirs2 if p.name[4:].isdigit()})
        gpu_indices = available if gpu_spec.lower() == "all" else self._parse_gpu_spec(gpu_spec, max(available, default=0) + 1)
        
        # Create report directories under both A and B
        report_name1 = Path(getattr(args, "_report_dir1", dir1)).name
        report_name2 = Path(getattr(args, "_report_dir2", dir2)).name
        report_dir1 = Path(getattr(args, "_report_dir1", dir1)) / "report" / f"compare_{report_name1}_vs_{report_name2}"
        report_dir2 = Path(getattr(args, "_report_dir2", dir2)) / "report" / f"compare_{report_name1}_vs_{report_name2}"
        report_dir1.mkdir(parents=True, exist_ok=True)
        report_dir2.mkdir(parents=True, exist_ok=True)
        
        print(f"[Compare] Comparing {dir1} vs {dir2}")
        print(f"[Compare] GPU selection: {gpu_spec} -> {gpu_indices}")
        print(f"[Compare] Reports will be saved to:")
        print(f"  - {report_dir1}")
        print(f"  - {report_dir2}")
        
        # Create temporary local shadow directories for extraction
        import tempfile
        import tarfile
        import shutil
        temp_base = tempfile.mkdtemp(prefix='opguard_compare_')
        temp_dir1 = Path(temp_base) / "run1"
        temp_dir2 = Path(temp_base) / "run2"
        temp_dir1.mkdir(parents=True, exist_ok=True)
        temp_dir2.mkdir(parents=True, exist_ok=True)
        
        try:
            for gpu_idx in gpu_indices:
                gpu_name = f"gpu_{gpu_idx:03d}"
                gpu_dir1 = Path(exec1) / gpu_name
                gpu_dir2 = Path(exec2) / gpu_name
                
                if not gpu_dir1.exists() or not gpu_dir2.exists():
                    print(f"[WARN] GPU {gpu_name} not found in both directories, skipping")
                    continue
                
                # Check if tar.gz files exist
                tar_file1 = gpu_dir1 / "sync.tar.gz"
                tar_file2 = gpu_dir2 / "sync.tar.gz"
                
                # Extract directories for comparison
                extract_dir1 = temp_dir1 / "exec_trace" / gpu_name
                extract_dir2 = temp_dir2 / "exec_trace" / gpu_name
                extract_dir1.mkdir(parents=True, exist_ok=True)
                extract_dir2.mkdir(parents=True, exist_ok=True)
                
                # Handle run1
                if tar_file1.exists():
                    print(f"[Compare] Extracting {tar_file1} to temporary directory...")
                    try:
                        with tarfile.open(tar_file1, 'r:gz') as tarf:
                            tarf.extractall(extract_dir1)
                        print(f"  ✅ Extracted to: {extract_dir1}")
                        compare_dir1 = extract_dir1
                    except Exception as e:
                        print(f"  ❌ Failed to extract {tar_file1}: {e}")
                        # Fallback to original directory if extraction fails
                        compare_dir1 = gpu_dir1
                else:
                    # No tar.gz, use original directory
                    compare_dir1 = gpu_dir1
                
                # Handle run2
                if tar_file2.exists():
                    print(f"[Compare] Extracting {tar_file2} to temporary directory...")
                    try:
                        with tarfile.open(tar_file2, 'r:gz') as tarf:
                            tarf.extractall(extract_dir2)
                        print(f"  ✅ Extracted to: {extract_dir2}")
                        compare_dir2 = extract_dir2
                    except Exception as e:
                        print(f"  ❌ Failed to extract {tar_file2}: {e}")
                        # Fallback to original directory if extraction fails
                        compare_dir2 = gpu_dir2
                else:
                    # No tar.gz, use original directory
                    compare_dir2 = gpu_dir2
                
                # Save report to both directories
                output_file1 = report_dir1 / f"perfetto_{gpu_name}.json.gz"
                output_file2 = report_dir2 / f"perfetto_{gpu_name}.json.gz"
                
                print(f"\n[Perfetto] Comparing {gpu_name}...")
                print(f"  Run1: {compare_dir1}")
                print(f"  Run2: {compare_dir2}")
                
                try:
                    from opguard.trace_export.perfetto.build import build_perfetto
                    
                    # Generate perfetto trace with ignore_fields from config
                    # Use temporary directory for output first
                    temp_output = Path(temp_base) / f"perfetto_{gpu_name}.json.gz"
                    build_perfetto(
                        str(compare_dir1),
                        str(compare_dir2),
                        str(temp_output),
                        ignore_fields=ignore_config
                    )
                    
                    # Copy to both report directories
                    shutil.copy2(temp_output, output_file1)
                    print(f"  ✅ Generated: {output_file1}")
                    
                    shutil.copy2(temp_output, output_file2)
                    print(f"  ✅ Copied to: {output_file2}")
                    
                    # Clean up temporary output
                    if temp_output.exists():
                        temp_output.unlink()
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            # Clean up temporary extraction directories
            try:
                shutil.rmtree(temp_base)
                print(f"[Compare] Cleaned up temporary directory: {temp_base}")
            except Exception as e:
                print(f"[WARN] Failed to clean up temporary directory {temp_base}: {e}")
        
        print(f"\n[Compare] Comparison complete. Reports saved to:")
        print(f"  - {report_dir1}")
        print(f"  - {report_dir2}")


def main():
    """Main CLI entry point"""
    if sys.argv[1:2] == ['doctor']:
        from opguard.doctor import main as doctor_main
        return doctor_main(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description="OpGuard: Unified tracing and analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick mode: run scripts (Python or bash) and compare
  opguard run script1.py script2.py
  opguard run run_training.sh run_inference.sh --gpu 0,2,3

  # Traditional profiler mode (automatic monitoring)
  opguard configure                    # Check configuration
  opguard dumpto /path/to/output       # Set output directory
  eval "$(opguard start)"              # Start tracing and enter monitoring mode automatically
  python your_script.py                # Run your script (automatically tracked)
  eval "$(opguard stop)"               # Finish discovery and deactivate shell monitoring
  opguard replay                       # Run instrumented code to collect execution traces (uses recorded script)

  # Compare two runs
  opguard compare /path/run1 /path/run2 --gpu 0,2,3
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.add_parser('doctor', help='Check Torch, CUDA, CUPTI and native extensions (doctor --help for options)')
    
    # configure
    subparsers.add_parser('configure', help='Display and confirm all OPGUARD_* variables')
    
    # dumpto
    dumpto_parser = subparsers.add_parser('dumpto', help='Set output directory (use "default" for output configuration)')
    dumpto_parser.add_argument('dir', help='Output directory path or "default"')
    
    # start
    start_parser = subparsers.add_parser('start', help='Start tracing (callsite + runtime)')
    start_parser.add_argument('dir', nargs='?', default=None,
                             help='Output directory (optional, equivalent to "opguard dumpto <dir> && opguard start")')
    start_parser.add_argument('--no-auto-activate', action='store_true',
                             help='Do not show auto-activation instructions')
    start_parser.add_argument('--enable-cupti', action='store_true',
                             help='Enable CUPTI tracing mechanism (disabled by default)')
    start_parser.add_argument('--disable-cudaapi', action='store_true',
                             help='Disable CUDA API tracing mechanism')
    start_parser.add_argument('--disable-python', action='store_true',
                             help='Disable Python tracing mechanism')
    start_parser.add_argument('--disable-local-dump', action='store_true',
                             help='Disable local dump mode: write directly to target directory (local dump is enabled by default)')
    start_parser.add_argument('--local-dump-sync-interval', type=float, default=1800.0,
                             help='Periodic sync interval in seconds for local dump (default: 1800 = 30 minutes)')
    
    # stop
    stop_parser = subparsers.add_parser('stop', help='Stop tracing, process callsite traces, and generate instrumentation config')
    stop_parser.add_argument('--disable-local-dump', action='store_true',
                             help='Disable local dump mode: write directly to target directory (local dump is enabled by default)')
    stop_parser.add_argument('--local-dump-sync-interval', type=float, default=1800.0,
                             help='Periodic sync interval in seconds for local dump (default: 1800 = 30 minutes)')
    
    # replay
    replay_parser = subparsers.add_parser('replay', help='Run instrumented code to collect execution traces (uses script from recording phase)')
    replay_parser.add_argument('--disable-local-dump', action='store_true',
                             help='Disable local dump mode: write directly to target directory (local dump is enabled by default)')
    replay_parser.add_argument('--local-dump-sync-interval', type=float, default=1800.0,
                             help='Periodic sync interval in seconds for local dump (default: 1800 = 30 minutes)')
    
    # resume
    subparsers.add_parser('resume', help='Resume tracing (bypass existence check, not recommended)')
    
    # run
    run_parser = subparsers.add_parser('run', help='Quick mode: run scripts (Python or bash) and generate perfetto comparison')
    run_parser.add_argument('scripts', nargs='+', help='Scripts to run (Python .py files or bash .sh files)')
    run_parser.add_argument('--gpu', default='0', help='GPU selection (default: 0, supports: 0,2,3 or 1-8 or all)')
    run_parser.add_argument('--disable-cupti', action='store_true',
                           help='Disable CUPTI tracing mechanism')
    run_parser.add_argument('--disable-cudaapi', action='store_true',
                           help='Disable CUDA API tracing mechanism')
    run_parser.add_argument('--disable-python', action='store_true',
                           help='Disable Python tracing mechanism')
    run_parser.add_argument('--disable-local-dump', action='store_true',
                           help='Disable local dump mode: write directly to target directory (local dump is enabled by default)')
    run_parser.add_argument('--local-dump-sync-interval', type=float, default=1800.0,
                           help='Periodic sync interval in seconds for local dump (default: 1800 = 30 minutes)')
    
    # compare
    compare_parser = subparsers.add_parser('compare', help='Compare two trace directories using perfetto')
    compare_parser.add_argument('dir1', help='First trace directory')
    compare_parser.add_argument('dir2', help='Second trace directory')
    compare_parser.add_argument('--gpu', default='0', help='GPU selection (default: 0, supports: 0,2,3 or 1-8 or all)')
    default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    compare_parser.add_argument('--ignore-config', type=str, default=default_config_path,
                                help=f'YAML file with per-function ignore fields (default: {default_config_path})')
    
    # instrument
    instrument_parser = subparsers.add_parser(
        'instrument',
        help='Merge a callsite dump folder (optional) and instrument code',
    )
    instrument_parser.add_argument(
        'dir', nargs='?', default=None,
        help='Callsite dump folder from the previous run (contains callsite_trace/ or *.json). Merges then instruments.',
    )
    instrument_parser.add_argument('-c', '--config', default=None,
                                  help='Path to instrumentation config YAML (skip merge if provided)')
    instrument_parser.add_argument(
        '--base-dir', default=None,
        help='Code repo root for resolving relative paths in generated YAML (e.g. /path/to/project)',
    )
    instrument_parser.add_argument(
        '--force-preprocess', action='store_true',
        help='Re-merge callsite traces even if an instrumentation YAML already exists',
    )
    instrument_parser.add_argument('--disable-local-dump', action='store_true',
                                  help='Disable local dump mode in emitted runtime env (local dump is enabled by default)')
    instrument_parser.add_argument('--local-dump-sync-interval', type=float, default=1800.0,
                                  help='Periodic sync interval in seconds for local dump (default: 1800 = 30 minutes)')
    instrument_parser.add_argument('--no-emit-env', action='store_true',
                                  help='Do not output shell export commands after instrumentation')
    
    # clear
    clear_parser = subparsers.add_parser('clear', help='Clear all instrumentation and clean up OpGuard artifacts')
    clear_parser.add_argument('--yes', '-y', action='store_true', 
                              help='Skip confirmation prompts and delete everything')
    clear_parser.add_argument('--clear-state', action='store_true',
                              help='Also clear CLI state file (~/.opguard_state.yaml)')
    clear_parser.add_argument('--instrumentation-only', action='store_true',
                              help='Only uninstrument files; keep backups, .opguard directory, and CLI state')
    
    # sync
    sync_parser = subparsers.add_parser('sync', help='Sync files from local shadow directory to target directory (requires local dump to be enabled)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = OpGuardCLI()
    
    try:
        if args.command == 'configure':
            cli.configure(args)
        elif args.command == 'dumpto':
            cli.dumpto(args)
        elif args.command == 'start':
            cli.start(args)
        elif args.command == 'stop':
            cli.stop(args)
        elif args.command == 'replay':
            cli.replay(args)
        elif args.command == 'resume':
            cli.resume(args)
        elif args.command == 'run':
            cli.run(args)
        elif args.command == 'compare':
            cli.compare(args)
        elif args.command == 'instrument':
            cli.instrument(args)
        elif args.command == 'clear':
            cli.clear(args)
        elif args.command == 'sync':
            cli.sync(args)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
