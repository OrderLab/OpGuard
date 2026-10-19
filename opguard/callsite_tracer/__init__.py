"""Callsite tracer package utilities."""

from importlib import import_module


def load_autostart_module():
    """Load autostart module with compatibility fallback for older layouts."""
    try:
        return import_module("opguard.callsite_tracer.cupti_tracer._autostart")
    except ModuleNotFoundError as exc:
        # Fallback only for missing cupti_tracer package/module.
        if exc.name not in {
            "opguard.callsite_tracer.cupti_tracer",
            "opguard.callsite_tracer.cupti_tracer._autostart",
        }:
            raise
        return import_module("opguard.callsite_tracer._autostart")


__all__ = ["load_autostart_module"]
