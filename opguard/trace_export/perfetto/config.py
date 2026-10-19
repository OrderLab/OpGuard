from __future__ import annotations

# Display & time mapping
DISPLAY_TIME_UNIT = "ms"
SCALE_NS_PER_TICK = 1_000

# Bar geometry (ticks)
BAR_WIDTH_TICKS   = 60
BAR_GAP_TICKS     = 8
SLICE_STRIDE_TICKS = BAR_WIDTH_TICKS + BAR_GAP_TICKS
FILE_GAP_TICKS    = 8

# Run2 baseline offset (kept from your original)
RUN2_OFFSET_TICKS = SLICE_STRIDE_TICKS - 1

# Event modes
DEFAULT_EVENT_MODE = "flat"  # "flat" | "stacked"

# Perfetto color names
C_OK, C_DIFF, C_ANOM, C_FIRST = "good", "bad", "yellow", "terrible"
C_INS = "olive"

# Compare / ignore fields
IGNORED_TOP = {"pid", "thread_id", "timestamp", "counter", "fname", "tag_prefix", "stack_trace", "line_num", "source_files"}

# Stack track config
STACK_EXCLUDE_FUNC_GLOBS = ["<*>", "*_call_impl"]
STACK_TRACE_MAX_DEPTH = 64

# Diff summarization
DIFF_ATOMS_MAX = 8
DIFF_ATOMS_HARD = 64
_IO_STAT_KEYS = ("dtype", "device", "numel", "xor_signature", "xor_sig_mode", "xor_sig_layout", "contiguous", "has_nan", "has_inf")

# Debug globals (mutated by CLI)
DEBUG_ENABLED: bool = False
DEBUG_DIR = None  # Path | None set in cli
