import os
import logging

ENABLE_CHECKSUM_NONCONTIG = os.getenv("OPGUARD_ENABLE_CHECKSUM_NONCONTIG", 0)
OPGUARD_LINENO_MODE = os.getenv("OPGUARD_LINENO_MODE", "begin")

# enable save runtime callstack in trace
# Use a function to read dynamically from environment (not cached at import time)
def get_runtime_callstack():
    """Get OPGUARD_RUNTIME_CALLSTACK value dynamically from environment (default: 1 to enable stack traces)"""
    return int(os.getenv("OPGUARD_RUNTIME_CALLSTACK", 1))

# For backward compatibility, also provide as a property-like access
# But prefer using get_runtime_callstack() for dynamic reads
OPGUARD_RUNTIME_CALLSTACK = get_runtime_callstack()

# delete ops and stream_counts fields in config yaml
OPGUARD_STRIP_SYNC_YAML = os.getenv("OPGUARD_STRIP_SYNC_YAML", 0)
# instrument multi-ops per line
OPGUARD_MULTI_OPS_PER_LINE = int(os.getenv("OPGUARD_MULTI_OPS_PER_LINE", 0))

# Diagnostic captures use fixed flat-word regions, independently of shape.
# Zero retains the historical explicit opt-out (one whole-tensor signature).
OPGUARD_XOR_REGIONAL = int(os.getenv("OPGUARD_XOR_REGIONAL", "4"))
if not 0 <= OPGUARD_XOR_REGIONAL <= 64:
    raise ValueError("OPGUARD_XOR_REGIONAL must be between 0 and 64")
OPGUARD_XOR_REGIONAL_THRESHOLD = int(os.getenv("OPGUARD_XOR_REGIONAL_THRESHOLD", "0"))

OPGUARD_SAVE_ORI_TRACE = int(os.getenv("OPGUARD_SAVE_ORI_TRACE", 0))

OPGUARD_STACK_TAIL_LINES = int(os.getenv("OPGUARD_STACK_TAIL_LINES", 2))

# Logging will use get_runtime_callstack() to show current value
logging.info(f"""[OpGuard] start with env settings: 
    * ENABLE_CHECKSUM_NONCONTIG={ENABLE_CHECKSUM_NONCONTIG}
    * OPGUARD_LINENO_MODE={OPGUARD_LINENO_MODE}
    * OPGUARD_RUNTIME_CALLSTACK={get_runtime_callstack()} (dynamic)
    * OPGUARD_STRIP_SYNC_YAML={OPGUARD_STRIP_SYNC_YAML}
    * OPGUARD_MULTI_OPS_PER_LINE={OPGUARD_MULTI_OPS_PER_LINE}
    * OPGUARD_XOR_REGIONAL={OPGUARD_XOR_REGIONAL}
    * OPGUARD_XOR_REGIONAL_THRESHOLD={OPGUARD_XOR_REGIONAL_THRESHOLD}
    * OPGUARD_SAVE_ORI_TRACE={OPGUARD_SAVE_ORI_TRACE}
    * OPGUARD_STACK_TAIL_LINES={OPGUARD_STACK_TAIL_LINES}""")
