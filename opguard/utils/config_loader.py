#!/usr/bin/env python3
"""
Unified config loader for OpGuard that uses OPGUARD_CONFIG_PATH.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_opguard_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load OpGuard configuration with override logic:
    - Load base config from OPGUARD_CONFIG_PATH
    - Override with callsite config (OPGUARD_CALLSITE_CONFIG or provided config_path)
    - Callsite config takes precedence over base config for all sections
    
    Args:
        config_path: Optional override for callsite config path
    
    Returns:
        Dictionary containing the complete config with all sections
    """
    
    # Load base config
    # Default to opguard/collections/default_config.yaml in the package directory
    base_config_path = os.environ.get('OPGUARD_CONFIG_PATH')
    if not base_config_path:
        # Find the opguard package directory
        opguard_package_dir = Path(__file__).parent.parent
        default_config_path = opguard_package_dir / "collections" / "default_config.yaml"
        if default_config_path.exists():
            base_config_path = str(default_config_path)
            print(f"[INFO] Using default OPGUARD_CONFIG_PATH: {base_config_path}", file=sys.stderr)
    
    base_config = {}
    if base_config_path and os.path.exists(base_config_path):
        print(f"[INFO] Loading base config from: {base_config_path}")
        with open(base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
    else:
        print(f"[WARNING] OPGUARD_CONFIG_PATH not set or file not found: {base_config_path}")
    
    # Load callsite config (overrides base config)
    callsite_config_path = config_path or os.environ.get('OPGUARD_CALLSITE_CONFIG')
    callsite_config = {}
    
    if callsite_config_path and os.path.exists(callsite_config_path):
        print(f"[INFO] Loading callsite config from: {callsite_config_path}", file=sys.stderr)
        with open(callsite_config_path, 'r', encoding='utf-8') as f:
            callsite_config = yaml.safe_load(f)
    else:
        print(f"[WARNING] OPGUARD_CALLSITE_CONFIG not set or file not found: {callsite_config_path}")
    
    # Merge configs: callsite config overrides base config
    merged_config = base_config.copy()
    merged_config.update(callsite_config)
    
    # Log what sections are being used
    for section in ['instrumentations', 'meta_vars', 'entry']:
        if section in merged_config:
            count = len(merged_config[section])
            source = "callsite" if section in callsite_config else "base"
            print(f"[INFO] Using {count} {section} from {source} config")
        else:
            print(f"[INFO] No {section} section found")
    
    # Resolve relative paths using config file directory or current working directory
    # Priority: callsite config directory > base config directory > cwd
    base_dir = None
    if callsite_config_path:
        base_dir = os.path.dirname(os.path.abspath(callsite_config_path))
        print(f"[INFO] Using callsite config directory for path resolution: {base_dir}", file=sys.stderr)
    elif base_config_path:
        base_dir = os.path.dirname(os.path.abspath(base_config_path))
        print(f"[INFO] Using base config directory for path resolution: {base_dir}", file=sys.stderr)

    base_dir = os.getenv('DEFAULT_CODE_REPO_BASE', base_dir)
    print(f"[INFO] Using DEFAULT_CODE_REPO_BASE for path resolution: {base_dir}", file=sys.stderr)

    merged_config = resolve_relative_paths(merged_config, base_dir)
    
    return merged_config


def resolve_relative_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in config using the provided base directory or current working directory.
    
    Args:
        config: The configuration dictionary
        base_dir: Base directory for resolving relative paths (defaults to current working directory)
    
    Returns:
        Config with resolved paths
    """
    
    if base_dir is None:
        base_dir = os.getcwd()
    
    # Resolve paths in instrumentations section
    if 'instrumentations' in config:
        for item in config['instrumentations']:
            if 'file_path' in item and not os.path.isabs(item['file_path']):
                item['file_path'] = os.path.join(base_dir, item['file_path'])
    
    # Resolve paths in meta_vars section
    if 'meta_vars' in config:
        for item in config['meta_vars']:
            if 'file_path' in item and not os.path.isabs(item['file_path']):
                item['file_path'] = os.path.join(base_dir, item['file_path'])
    
    # Resolve paths in entry section
    if 'entry' in config:
        for item in config['entry']:
            if 'original_module' in item and not os.path.isabs(item['original_module']):
                item['original_module'] = os.path.join(base_dir, item['original_module'])
    
    return config


def get_instrumentations_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract instrumentations section from config."""
    return config.get('instrumentations', [])


def get_meta_vars_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract meta_vars section from config."""
    return config.get('meta_vars', [])


def get_entry_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract entry section from config."""
    return config.get('entry', [])


def create_default_opguard_config(output_path: str = "opguard_base.yaml"):
    """
    Create a default base OpGuard configuration file.
    This is the base config that can be overridden by callsite configs.
    
    Args:
        output_path: Path where to save the default base config
    """
    
    default_config = {
        "instrumentations": [],
        "meta_vars": [],
        "entry": [],
        "callsite_tracer": [],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"[INFO] Created default base OpGuard config: {output_path}", file=sys.stderr)
    print(f"[INFO] Set OPGUARD_CONFIG_PATH={output_path} to use this as base config", file=sys.stderr)
    print(f"[INFO] Set OPGUARD_CALLSITE_CONFIG to override specific sections", file=sys.stderr)
    
    return output_path


def main():
    """CLI for config management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OpGuard Configuration Management")
    parser.add_argument("--create-default", action="store_true",
                       help="Create a default base OpGuard config file")
    parser.add_argument("--config-path", help="Path to callsite config file to load and display")
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_opguard_config()
        return
    
    if args.config_path:
        try:
            config = load_opguard_config(args.config_path)
            print(f"[INFO] Loaded merged config from callsite: {args.config_path}", file=sys.stderr)
            print(f"[INFO] Sections found:", file=sys.stderr)
            print(f"  - instrumentations: {len(get_instrumentations_config(config))} entries", file=sys.stderr)
            print(f"  - meta_vars: {len(get_meta_vars_config(config))} entries", file=sys.stderr)
            print(f"  - entry: {len(get_entry_config(config))} entries", file=sys.stderr)
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 