#!/usr/bin/env python3
"""
Script to run callsite tracer instrumentation
"""

import os
import sys
import argparse

# Guard against recursive calls
if os.environ.get("OPGUARD_CALLSITE_INSTRUMENTATION_RUNNING", "0") == "1":
    print("Callsite instrumentation already running, skipping...")
    sys.exit(0)

def run_callsite_instrumentation(config_path=None):
    """Run callsite tracer instrumentation if enabled."""
    if os.environ.get("OPGUARD_CALLSITE_TRACER", "0") == "1":
        print("=== OpGuard Callsite Tracer Instrumentation ===")
        print(f"OPGUARD_CALLSITE_TRACER: {os.environ.get('OPGUARD_CALLSITE_TRACER')}")
        print("Running callsite tracer instrumentation...")

        # Set flag to prevent recursive calls
        os.environ["OPGUARD_CALLSITE_INSTRUMENTATION_RUNNING"] = "1"

        try:
            env_config = os.environ.get('OPGUARD_CONFIG_PATH')
            # Determine config paths
            config_paths = []
            if config_path:
                config_paths.append(config_path)
            config_paths.append(env_config)

            config_found = False
            for path in config_paths:
                if os.path.exists(path):
                    print(f"Found config: {path}")
                    # Import and run the instrumentor directly
                    parent_dir = os.path.dirname(os.path.dirname(__file__))
                    sys.path.insert(0, parent_dir)

                    try:
                        from opguard.instrumentor.callsite_tracer_injector import inject_callsite_tracer
                        inject_callsite_tracer(path)
                        config_found = True
                        break
                    except ImportError as e:
                        print(f"Warning: Could not import instrumentor module: {e}")
                        print("This is expected during setup.py execution")
                        continue

            if not config_found:
                print("Warning: No config file found for callsite tracer instrumentation")

        except Exception as e:
            print(f"Warning: Failed to run callsite tracer instrumentation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            os.environ.pop("OPGUARD_CALLSITE_INSTRUMENTATION_RUNNING", None)
    else:
        print("OPGUARD_CALLSITE_TRACER not set to 1, skipping instrumentation")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpGuard callsite tracer instrumentation")
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = os.environ.get('OPGUARD_CONFIG_PATH')
        if not config_path:
            print("[ERROR] No config specified and OPGUARD_CONFIG_PATH not set")
            sys.exit(1)

    run_callsite_instrumentation(config_path=args.config)
