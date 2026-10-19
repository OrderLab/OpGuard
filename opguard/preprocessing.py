#!/usr/bin/env python3
"""
Preprocessing script for OpGuard callsite data processing pipeline.

CHANGES (Aug 2025):
- Wildcards are no longer used/needed. To process many files, pass one or more
  directories; we will recursively collect *.jsonl and combine them first.
- Single-file inputs are still supported (explicit file paths).
- For directories we use a folding combine to avoid shell quoting and arg-length issues.

This script handles the complete workflow:
1. Gathers inputs:
   - Directories: recursively collect *.jsonl (skip others)
   - Files: explicit *.json or *.jsonl files
2. Combines (directory contents first; each dir becomes one combined .json)
3. Compresses the combined traces
4. Generates instrumentation YAML configurations
5. Compares configurations if multiple inputs exist
6. Combines configurations using a folding approach
7. (Optional) Applies a basic filter to the final YAML

Usage Examples (no quoting headaches):
    # Process an entire folder of jsonl traces (recursively)
    python3 opguard/preprocessing.py --input /path/to/trace_dir

    # Process multiple folders
    python3 opguard/preprocessing.py --input /path/dirA /path/dirB

    # Mix a folder and specific files
    python3 opguard/preprocessing.py --input /path/dirA /path/file.jsonl /path/file.json

    # Specify base directory for relative paths
    python3 opguard/preprocessing.py --input /path/dirA --base-dir /path/to/code
"""

import os
import sys
import json
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Third-party deps
import yaml  # noqa: F401

# Add the opguard package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _running import get_trace_output_dir  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _is_json_or_jsonl(p: Path) -> bool:
    return p.suffix.lower() in {".json", ".jsonl"}


def _gather_dir_jsonl_recursive(d: Path) -> List[Path]:
    jsonl_files = [p for p in d.rglob("*.jsonl") if p.is_file()]
    json_files = [p for p in d.rglob("*.json") if p.is_file()]
    jsonl_files.sort()
    json_files.sort()
    return jsonl_files + json_files


def _chunk(lst: List[str], n: int) -> List[List[str]]:
    """Chunk a list into size-n pieces."""
    return [lst[i:i+n] for i in range(0, len(lst), n)]


class PreprocessingPipeline:
    """Orchestrates the complete preprocessing pipeline for callsite data."""

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.getcwd()
        self.work_dir: Optional[str] = None
        self.input_entries: List[Dict[str, object]] = []
        self.processed_files: List[Dict[str, object]] = []

    def setup_work_directory(self, output_dir: str) -> str:
        """Create and return a unique work directory for this preprocessing run."""
        if output_dir:
            work_dir = output_dir
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = os.path.join(get_trace_output_dir(), f"preprocessing_{timestamp}")
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
        logger.info(f"Created work directory: {work_dir}")
        return work_dir

    def find_input_files(self, inputs: List[str]) -> List[Dict[str, object]]:
        """
        Resolve inputs into entries:
          - {'type': 'dir', 'name': <dirname>, 'files': [list of *.json/jsonl paths]}
          - {'type': 'single', 'name': <file_stem>, 'files': [file]}
        """
        entries: List[Dict[str, object]] = []

        for raw in inputs:
            p = Path(raw).expanduser().resolve()
            if p.is_dir():
                files = _gather_dir_jsonl_recursive(p)
                if not files:
                    logger.warning(f"[dir] No *.json/jsonl found under: {p}")
                    continue
                entries.append({
                    "type": "dir",
                    "name": p.name,
                    "root": str(p),
                    "files": [str(f) for f in files]
                })
                logger.info(f"[dir] Collected {len(files)} json/jsonl files from {p}")
            elif p.is_file():
                if not _is_json_or_jsonl(p):
                    logger.warning(f"[file] Skipping non-json/jsonl file: {p}")
                    continue
                entries.append({
                    "type": "single",
                    "name": p.stem,
                    "files": [str(p)]
                })
                logger.info(f"[file] Single file: {p}")
            else:
                logger.warning(f"Path not found: {raw}")

        self.input_entries = entries
        return entries

    def _folding_combine(self, inputs: List[str], out_file: str) -> str:
        """
        Combine many inputs robustly using a folding approach to avoid argv limits.
        Repeatedly combine in manageable chunks into temporaries, then fold.
        """
        py = sys.executable
        combine_py = os.path.join(os.path.dirname(__file__), "callsite_tracer", "combine_callsites.py")

        # Limit: number of files per invocation (tune if needed)
        MAX_PER_CALL = 200

        # If small enough, do it directly
        if len(inputs) <= MAX_PER_CALL:
            cmd = [py, combine_py, "--input", *inputs, "--output", out_file]
            logger.info(f"Combining {len(inputs)} files -> {out_file}")
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            return out_file

        # Otherwise, chunk + fold
        temp_files: List[str] = []
        for idx, group in enumerate(_chunk(inputs, MAX_PER_CALL), 1):
            tmp = os.path.join(self.work_dir, f"combine_chunk_{idx}.json")  # type: ignore
            cmd = [py, combine_py, "--input", *group, "--output", tmp]
            logger.info(f"[chunk {idx}] Combining {len(group)} files -> {tmp}")
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            temp_files.append(tmp)

        # Now fold the chunked outputs
        current = temp_files[0]
        for i, nxt in enumerate(temp_files[1:], 1):
            tmp_out = os.path.join(self.work_dir, f"combine_fold_{i}.json")  # type: ignore
            cmd = [py, combine_py, "--input", current, nxt, "--output", tmp_out, "--no-rank"]
            logger.info(f"[fold {i}] {Path(current).name} + {Path(nxt).name} -> {Path(tmp_out).name}")
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            current = tmp_out

        # Final move/rename
        import shutil
        shutil.copy2(current, out_file)
        return out_file

    def combine_callsite_files(self, entry: Dict[str, object]) -> str:
        """Combine files for an entry. For directories, combine ALL *.json/jsonl first."""
        assert self.work_dir, "work_dir must be set before combining"
        name = str(entry.get("name", "combined"))
        files = list(entry.get("files", []))  # type: ignore
        # If it's a single file input, just copy (or normalize to .json)
        if entry["type"] == "single":
            input_file = files[0]
            output_file = os.path.join(self.work_dir, f"{name}_combined.json")
            with open(input_file, "r") as src, open(output_file, "w") as dst:
                dst.write(src.read())
            logger.info(f"Copied single file: {input_file} -> {output_file}")
            return output_file

        # Directory inputs: combine ALL first (folding approach to avoid argv length issues)
        output_file = os.path.join(self.work_dir, f"{name}_combined.json")
        logger.info(f"Combining {len(files)} files -> {output_file}")
        try:
            return self._folding_combine(files, output_file)
        except Exception as e:
            logger.error(f"Failed to combine files: {e}")
            raise

    def compress_callsite_file(self, input_file: str) -> str:
        """Compress a callsite file using the compress_callsite.py script."""
        output_file = input_file.replace(".json", "_compressed.json")  # normalized to .json
        logger.info(f"Compressing {input_file} -> {output_file}")

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "callsite_tracer", "compress_callsite.py"),
            "-i", input_file,
            "-o", output_file,
        ]

        logger.info(f"Running compress_callsite.py: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully compressed {input_file} -> {output_file}")
            if result.stdout:
                logger.debug(f"Compression output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compress {input_file}: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

        return output_file

    def generate_instrumentation_yaml(self, compressed_file: str) -> str:
        """Generate instrumentation YAML from compressed callsite file."""
        base_name = Path(compressed_file).name.replace("_compressed.json", "")
        output_file = os.path.join(self.work_dir, f"{base_name}_instrumentation.yaml")

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "tools", "callsite_to_instrumentation.py"),
            "--callsite-file", compressed_file,
            "--output", output_file,
            "--base-dir", self.base_dir
        ]

        logger.info(f"Generating instrumentation YAML: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully generated instrumentation YAML: {output_file}")
            if result.stdout:
                logger.info(f"Generation output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to generate instrumentation YAML: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

        return output_file

    def compare_configs(self, base_yaml: str, compare_yaml: str, output_log: str):
        """Compare two instrumentation YAML files and save results to log."""
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "callsite_tracer", "compare_configs.py"),
            base_yaml, compare_yaml
        ]

        logger.info(f"Comparing configs: {base_yaml} vs {compare_yaml}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            with open(output_log, 'w') as f:
                f.write(f"Comparison: {os.path.basename(base_yaml)} vs {os.path.basename(compare_yaml)}\n")
                f.write("=" * 80 + "\n\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\nSTDERR:\n")
                    f.write(result.stderr)
            logger.info(f"Comparison results saved to: {output_log}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compare configs: {e}")
            with open(output_log, 'w') as f:
                f.write(f"Comparison failed: {e}\n")
                f.write(f"Error output: {e.stderr}\n")
            raise

    def combine_instrumentation_configs(self, yaml_files: List[str]) -> str:
        """
        Combine multiple instrumentation YAML files using combine_configs.py in a folding approach.
        """
        if len(yaml_files) == 1:
            return yaml_files[0]

        logger.info(f"Combining {len(yaml_files)} instrumentation configs with folding approach")
        current_combined = yaml_files[0]

        for i, next_yaml in enumerate(yaml_files[1:], 1):
            temp_output_dir = os.path.join(self.work_dir, f"temp_combined_{i}")  # type: ignore
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "callsite_tracer", "combine_configs.py"),
                current_combined, next_yaml, "-o", temp_output_dir
            ]
            logger.info(f"Running combine_configs.py: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"Step {i} complete")
                if result.stdout:
                    logger.debug(f"Combine output: {result.stdout}")
                combined_file = os.path.join(temp_output_dir, "combined.yaml")
                if os.path.exists(combined_file):
                    current_combined = combined_file
                else:
                    raise FileNotFoundError(f"Expected combined output file not found: {combined_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to combine configs at step {i}: {e}")
                logger.error(f"Error output: {e.stderr}")
                raise

        import shutil
        final_output = os.path.join(self.work_dir, "final_combined_instrumentation.yaml")  # type: ignore
        shutil.copy2(current_combined, final_output)
        logger.info(f"Final combined instrumentation saved to: {final_output}")
        return final_output

    def apply_basic_filtering(self, input_yaml: str, apply_basic: bool = False) -> str:
        """Apply basic filtering to the final instrumentation YAML using filter_config.py."""
        if not apply_basic:
            return input_yaml

        base_name = Path(input_yaml).name
        if base_name.endswith('_instrumentation.yaml'):
            base_name = base_name[:-len('_instrumentation.yaml')]
        output_file = os.path.join(self.work_dir, f"{base_name}_filtered_instrumentation.yaml")  # type: ignore

        logger.info(f"Applying basic filtering to {input_yaml} -> {output_file}")
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "callsite_tracer", "filter_config.py"),
            input_yaml, "-o", output_file, "--basic"
        ]

        logger.info(f"Running filter_config.py: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully applied basic filtering: {output_file}")
            if result.stderr:
                logger.info(f"Filtering summary: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply basic filtering: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

        return output_file

    def apply_sync_flag(self, input_yaml: str, apply_basic: bool = False ,sync: bool = False, default_stream_id: str = "7") -> str:
        """Add sync flag to the final instrumentation YAML using filter_config.py."""
        if not sync:
            return input_yaml
        base_name = Path(input_yaml).name
        if base_name.endswith('_instrumentation.yaml'):
            base_name = base_name[:-len('_instrumentation.yaml')]
        
        if apply_basic:
            output_file = os.path.join(self.work_dir, f"{base_name}__filtered_sync_instrumentation.yaml")  # type: ignore
        else:
            output_file = os.path.join(self.work_dir, f"{base_name}_sync_instrumentation.yaml")  # type: ignore
        
        logger.info(f"Applying sync flag to {input_yaml} -> {output_file}")

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "callsite_tracer", "add_sync_config.py"),
            input_yaml, "-s", default_stream_id, "-o", output_file
        ]

        logger.info(f"Running add_sync_config.py: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully applied sync flag: {output_file}")
            if result.stderr:
                logger.info(f"Sync flag summary: {result.stderr}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply sync flag: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

        return output_file

    def run_pipeline(self, inputs: List[str], apply_basic: bool = False, sync: bool = False, default_stream_id: str = "7", output_dir: str = None) -> Dict[str, object]:
        """Run the complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")
        work_dir = self.setup_work_directory(output_dir)

        entries = self.find_input_files(inputs)
        if not entries:
            raise ValueError("No valid inputs found (folders with *.j son or explicit json/jsonl files)")

        logger.info(f"Processing {len(entries)} input entries")

        # Step 1: Combine callsite files (directories first)
        combined_files: List[str] = []
        for entry in entries:
            combined_file = self.combine_callsite_files(entry)
            combined_files.append(combined_file)
            self.processed_files.append({
                "entry": entry,
                "combined": combined_file
            })

        # Step 2: Compress combined files
        compressed_files: List[str] = []
        for combined_file in combined_files:
            compressed_files.append(self.compress_callsite_file(combined_file))

        # Step 3: Generate instrumentation YAML
        yaml_files: List[str] = []
        for compressed_file in compressed_files:
            yaml_files.append(self.generate_instrumentation_yaml(compressed_file))

        # Step 4: Compare configs if multiple entries
        if len(yaml_files) > 1:
            logger.info("Multiple entries detected, running comparisons")
            base_yaml = yaml_files[0]
            for i, compare_yaml in enumerate(yaml_files[1:], 1):
                log_file = os.path.join(work_dir, f"comparison_1_vs_{i+1}.log")
                self.compare_configs(base_yaml, compare_yaml, log_file)

        # Step 5: Combine instrumentation configs if multiple entries
        if len(yaml_files) > 1:
            logger.info("Multiple entries detected, combining instrumentation configs")
            final_yaml = self.combine_instrumentation_configs(yaml_files)
        else:
            final_yaml = yaml_files[0]

        # Step 6: Apply basic filtering if requested
        filtered_yaml = self.apply_basic_filtering(final_yaml, apply_basic=apply_basic)

        # Step 7: Add sync flag if requested
        sync_yaml = self.apply_sync_flag(filtered_yaml, apply_basic, sync, default_stream_id)

        # Create summary
        summary: Dict[str, object] = {
            "work_directory": work_dir,
            "input_entries": len(entries),
            "combined_files": combined_files,
            "compressed_files": compressed_files,
            "instrumentation_yamls": yaml_files,
            "final_combined_yaml": final_yaml,
            "filtered_yaml": filtered_yaml,
            "sync_yaml": sync_yaml,
        }

        # Save summary
        summary_file = os.path.join(work_dir, "pipeline_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Preprocessing pipeline completed successfully")
        logger.info(f"Work directory: {work_dir}")
        logger.info(f"Summary saved to: {summary_file}")
        return summary


def main():
    """Main function for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for OpGuard callsite data (directory-first; combines *.json/jsonl)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        nargs="+",
        help="Input paths: one or more directories (recursively gather *.json/jsonl) and/or explicit json/jsonl files"
    )

    parser.add_argument(
        "--base-dir",
        help="Base directory for relative paths (defaults to current working directory)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--basic", "-b",
        action="store_true",
        help="Apply basic filtering to final output"
    )

    parser.add_argument(
        "--sync", "-s",
        action="store_true",
        help="Add sync flag to final output"
    )

    parser.add_argument(
        "--default_stream_id",
        default="7",
        help="Default stream ID (default: 7)"
    )

    parser.add_argument(
        "--output_dir", "-o",
        default=None,
        help="Output path for final instrumentation YAML (default: <work_dir>/final_instrumentation.yaml)"
    )

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        pipeline = PreprocessingPipeline(base_dir=args.base_dir)
        summary = pipeline.run_pipeline(args.input, apply_basic=args.basic, sync=args.sync, default_stream_id=args.default_stream_id, output_dir=args.output_dir)

        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Work Directory: {summary['work_directory']}")
        print(f"Input Entries: {summary['input_entries']}")
        print(f"Combined Files: {len(summary['combined_files'])}")
        print(f"Compressed Files: {len(summary['compressed_files'])}")
        print(f"Instrumentation YAMLs: {len(summary['instrumentation_yamls'])}")
        if summary['final_combined_yaml']:
            print(f"Final Combined YAML: {summary['final_combined_yaml']}")
        if summary['filtered_yaml'] and summary['filtered_yaml'] != summary['final_combined_yaml']:
            print(f"Filtered YAML: {summary['filtered_yaml']}")
        if summary['sync_yaml'] and summary['sync_yaml'] != summary['final_combined_yaml']:
            print(f"Sync Added YAML: {summary['sync_yaml']}")
        print("="*80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
