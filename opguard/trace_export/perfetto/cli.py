# cli.py
import argparse, time, os
from pathlib import Path

from .build import build_perfetto
from .config import DEBUG_ENABLED, DEBUG_DIR

def main():
    ap = argparse.ArgumentParser("Export OpGuard diffs to Perfetto")
    ap.add_argument("--run1", required=True)
    ap.add_argument("--run2", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--ignore-output-only", action="store_true")
    ap.add_argument("--filter", type=str, default=None,
                    help="Key=val filter (e.g. meta.global_step=0)")
    ap.add_argument("--event-mode", choices=["flat", "stacked"], default="flat",
                    help="flat: single slice (default). stacked: B/E spans.")
    ap.add_argument("--strict", action="store_true",
                    help="Compatibility flag; function/callsite identity is always required.")
    ap.add_argument("--debug", action="store_true", help="Write debug JSONL")
    ap.add_argument("--debug-dir", default="./perfetto_debug", help="Directory for debug outputs")
    
    # Add ignore-config support
    default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    ap.add_argument("--ignore-config", type=str, default=default_config_path,
                    help=f"YAML file with per-function ignore fields (default: {default_config_path})")
    
    args = ap.parse_args()

    # configure globals
    from . import config
    if args.debug:
        config.DEBUG_ENABLED = True
        config.DEBUG_DIR = Path(args.debug_dir)
    else:
        config.DEBUG_ENABLED = False
        config.DEBUG_DIR = None

    # Load ignore config
    ignore_config = {}
    if args.ignore_config and os.path.exists(args.ignore_config):
        try:
            import yaml
            with open(args.ignore_config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                ignore_config = cfg.get("inspector_ignore_fields", {}) or {}
        except Exception as e:
            print(f"⚠️  Warning: Failed to load ignore config from {args.ignore_config}: {e}")
    
    t0 = time.time()
    build_perfetto(
        args.run1, args.run2, args.out,
        ignore_output_only=args.ignore_output_only,
        filter=args.filter,
        event_mode=args.event_mode,
        strict_fn=args.strict,
        ignore_fields=ignore_config,
    )
    print(f"⏱️ {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
