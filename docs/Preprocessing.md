# OpGuard Preprocessing Pipeline

The `preprocessing.py` script provides a complete automated pipeline for processing OpGuard callsite data from multiple sources into instrumentation configurations.

## Overview

The preprocessing pipeline performs the following steps:

1. **File Discovery & Combination**: Handles both individual JSON files and wildcard patterns (e.g., `fsdp_callsite_rank*.json`), combining multiple files into single traces where needed.

2. **Compression**: Uses `compress_callsite.py` to intelligently compress the combined traces by pruning linear chains and small subtrees, creating compressed versions with `_compressed.json` suffix.

3. **Instrumentation Generation**: Converts compressed callsite data into instrumentation YAML configurations with `_instrumentation.yaml` suffix.

4. **Configuration Comparison**: If multiple inputs exist, compares configurations between the first entry and others, saving results to log files.

5. **Configuration Combination**: Uses `combine_configs.py` with a folding approach to combine multiple instrumentation configs: (1+2)+3, then ((1+2)+3)+4, etc.

6. **Basic Filtering** (optional): Applies intelligent filtering to remove common torch utility functions and keep only meaningful function/method entries.

## Usage

### Basic Usage

```bash
# Process specific files
python3 opguard/preprocessing.py --input fsdp_callsite.json gpt_callsite.json

# Process wildcard patterns
python3 opguard/preprocessing.py --input "fsdp_callsite_rank*.json" "gpt_callsite_rank*.json"

# Process mixed files and patterns
python3 opguard/preprocessing.py --input fsdp_callsite.json "gpt_callsite_rank*.json"
```

### Advanced Options

```bash
# Specify base directory for relative paths
python3 opguard/preprocessing.py --input "*.json" --base-dir /path/to/code

# Enable verbose logging
python3 opguard/preprocessing.py --input "*.json" --verbose

# Apply basic filtering to final output
python3 opguard/preprocessing.py --input "*.json" --basic
```

## Input Types

The script handles two types of inputs:

### Single Files
- Individual JSON files (e.g., `fsdp_callsite.json`)
- Copied directly to work directory with `_combined.json` suffix

### Wildcard Patterns
- Patterns like `fsdp_callsite_rank*.json` or `gpt_callsite_rank*.json`
- Automatically finds matching files and combines them using `combine_callsites.py`
- Useful for distributed training traces from multiple ranks

## Output Structure

Each preprocessing run creates a unique timestamped directory under the trace output directory:

```
{et_trace_output_dir}/preprocessing_{YYYYMMDD_HHMMSS}/
├── {name}_combined.json              # Combined callsite data
├── {name}_compressed.json            # Compressed version (no callstacks)
├── {name}_instrumentation.yaml       # Generated instrumentation config
├── comparison_1_vs_2.log            # Comparison logs (if multiple inputs)
├── comparison_1_vs_3.log            # Additional comparisons
├── temp_combined_1.yaml             # Temporary combination files
├── temp_combined_2.yaml
├── final_combined_instrumentation.yaml  # Final combined config
├── {name}_filtered_instrumentation.yaml  # Filtered output (if --basic used)
└── pipeline_summary.json             # Complete pipeline summary
```

## Workflow Details

### Step 1: File Combination
- Single files are copied directly
- Wildcard patterns trigger `combine_callsites.py` to merge multiple files
- Each entry gets a `_combined.json` file

### Step 2: Compression
- Uses `compress_callsite.py` with intelligent pruning
- Enables leaf pruning to compress linear chains
- Applies subtree pruning with configurable thresholds
- Creates `_compressed.json` files with optimized structure

### Step 3: Instrumentation Generation
- Runs `callsite_to_instrumentation.py` on each compressed file
- Generates YAML configurations with proper context and tags
- Outputs `_instrumentation.yaml` files

### Step 4: Configuration Comparison (Multiple Inputs)
- Compares first config against all others
- Uses `compare_configs.py` to generate detailed diffs
- Saves comparison results to log files

### Step 5: Configuration Combination (Multiple Inputs)
- Uses `combine_configs.py` with folding approach: (1+2)+3, then ((1+2)+3)+4
- Intelligently merges matching entries and combines ops/stream_counts
- Produces final combined configuration

### Step 6: Basic Filtering (Optional)
- Applies `filter_config.py` with `--basic` flag if requested
- Removes common torch utility functions (cat, stack, reshape, etc.)
- Keeps only meaningful function/method entries
- Creates filtered output as the final result

## Environment Variables

- `--base-dir`: Base directory for preprocessing paths (defaults to the current working directory)
- `OPGUARD_TRACE_OUTPUT_BASE`: Override for trace output directory

## Dependencies

The script requires the following OpGuard modules:
- `_running.py`: For trace output directory management
- `callsite_tracer/combine_callsites.py`: For merging multiple files
- `callsite_to_instrumentation.py`: For generating instrumentation configs
- `callsite_tracer/compare_configs.py`: For comparing configurations
- `callsite_tracer/combine_configs.py`: For combining instrumentation configs
- `callsite_tracer/filter_config.py`: For applying basic filtering

## Error Handling

- Comprehensive logging at each step
- Graceful handling of missing files or failed operations
- Detailed error messages and stack traces
- Pipeline continues where possible, with clear failure points

## Example Scenarios

### Single File Processing
```bash
python3 opguard/preprocessing.py --input my_trace.json
```
Creates: `my_trace_combined.json` → `my_trace_compressed.json` → `my_trace_instrumentation.yaml`

### Multi-Rank File Processing
```bash
python3 opguard/preprocessing.py --input "fsdp_callsite_rank*.json"
```
Finds all matching rank files, combines them, then processes through the pipeline.

### Multiple Trace Types
```bash
python3 opguard/preprocessing.py --input fsdp_callsite.json "gpt_callsite_rank*.json"
```
Processes both a single FSDP trace and multiple GPT rank traces, comparing and combining the results.

## Troubleshooting

### Common Issues
1. **No files found**: Check wildcard patterns and file existence
2. **Permission errors**: Ensure write access to output directories
3. **Import errors**: Verify OpGuard package structure and dependencies
4. **Memory issues**: Large traces may require significant memory for processing

### Debug Mode
Use `--verbose` flag for detailed logging and subprocess output capture.

### Log Files
Check the generated log files in the work directory for detailed operation information and any error messages.
