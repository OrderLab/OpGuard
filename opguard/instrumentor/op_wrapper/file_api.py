
# opguard/instrumentor/op_wrapper/file_api.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import json
import libcst as cst
from libcst.metadata import MetadataWrapper

from .instrumentor import RobustCSTFunctionInstrumentor
from .misslog import MissLogger


def instrument_source_code_robust(source_code: str,
                                  config: List[Dict[str, Any]],
                                  wrap_log_path: Optional[str] = None,
                                  miss_log_path: Optional[str] = None) -> str:
    """
    Phase-A: run CST transformer to wrap call sites and collect prologue intents.
    Phase-B: (unless skipped) insert imports/assignments with PrologueInserter.
    Returns final instrumented source code.
    """
    tree = cst.parse_module(source_code)
    wrapper = MetadataWrapper(tree)

    tx = RobustCSTFunctionInstrumentor(config, source_code, wrapper)
    new_tree = wrapper.visit(tx)
    intermediate_code = new_tree.code

    # persist Phase-A logs
    if wrap_log_path:
        try:
            with open(wrap_log_path, "w", encoding="utf-8") as f:
                json.dump(tx.wrapped_log, f, indent=2)
        except Exception:
            pass

    if miss_log_path:
        try:
            with open(miss_log_path, "w", encoding="utf-8") as f:
                json.dump(tx.miss.summary(config, tx.wrapped_log), f, indent=2)
        except Exception:
            pass

    # Optionally skip Phase-B for external build steps
    if os.getenv("OPGUARD_SKIP_PROLOGUE_INSERT", "0").lower() in ("1", "true", "yes"):
        return intermediate_code

    # Phase-B: import/assign prologue insertion
    try:
        from opguard.instrumentor.prologue_inserter import PrologueInserter
        phase_b = PrologueInserter(
            wrapped_imports=tx.wrapped_imports,
            need_operator_import=tx.need_operator_import,
            need_flush_import=tx.need_flush_import,
        )
        final_module = cst.parse_module(intermediate_code).visit(phase_b)
        return final_module.code
    except Exception as e:
        # If Phase-B fails, fall back to Phase-A result but keep logs
        # Downstream can re-run Phase-B later.
        return intermediate_code


def instrument_file_robust(input_path: str,
                            output_path: str,
                            config: List[Dict[str, Any]],
                            log_dir: Optional[str] = None) -> None:
    """Instrument a single file and write logs alongside the output.
    - wrap log: <output_path>.log
    - miss log: <output_path>.miss.json (with summary counts)
    If log_dir is provided, logs go to that directory instead, named after the input file.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        src = f.read()

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        base = os.path.basename(output_path or input_path)
        wrap_log_path = os.path.join(log_dir, base + ".log")
        miss_log_path = os.path.join(log_dir, base + ".miss.json")
    else:
        wrap_log_path = (output_path or input_path) + ".log"
        miss_log_path = (output_path or input_path) + ".miss.json"

    out = instrument_source_code_robust(src, config, wrap_log_path=wrap_log_path, miss_log_path=miss_log_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out)
