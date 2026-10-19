"""Compatibility entrypoint; instrumentation alone never changes execution policy."""

from opguard.runtime_policy import apply_requested_policy, enable_torch_deterministic_mode


def _opguard_entry_handler(*args, **kwargs):
    apply_requested_policy()
