"""Public compatibility facade for local staging and batched uploads.

Implementation lives in local_sync: paths, bundles, transport, and lifecycle.
"""
import atexit
import logging
import os
import threading

from opguard.distributed import get_global_rank
from opguard.local_sync.manager import (
    LocalDumpManager, _periodic_sync_registry, _periodic_sync_registry_lock,
)

_get_gpu_rank = get_global_rank
_get_process_rank = get_global_rank
_local_dump_manager = None
_manager_lock = threading.RLock()


def _after_fork():
    global _local_dump_manager, _manager_lock, _periodic_sync_registry, _periodic_sync_registry_lock
    _local_dump_manager = None
    _manager_lock = threading.RLock()
    # No inherited thread or lock is usable in the child.
    from opguard.local_sync import manager
    _periodic_sync_registry = manager._periodic_sync_registry = {}
    _periodic_sync_registry_lock = manager._periodic_sync_registry_lock = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


def _close_on_exit():
    manager = get_local_dump_manager()
    if manager is not None:
        try:
            manager.close()
        except Exception:
            logging.exception('Final upload failed; local staging data retained at %s', manager.local_base)


atexit.register(_close_on_exit)


def enable_local_dump(target_dir, sync_interval_sec=1800.0):
    global _local_dump_manager
    with _manager_lock:
        if _local_dump_manager is not None:
            target = os.path.realpath(os.path.expanduser(target_dir))
            if target != _local_dump_manager.target_dir:
                raise ValueError('Disable the existing local dump manager before changing target')
            return _local_dump_manager
        manager = LocalDumpManager(target_dir, sync_interval_sec)
        manager.start_periodic_sync()
        _local_dump_manager = manager
        return manager


def disable_local_dump():
    global _local_dump_manager
    with _manager_lock:
        if _local_dump_manager is not None:
            _local_dump_manager.close()
            _local_dump_manager = None


def get_local_dump_manager():
    return _local_dump_manager


def translate_path(target_path):
    manager = get_local_dump_manager()
    if manager is None and os.getenv('OPGUARD_LOCAL_DUMP_ENABLED') == '1':
        target = os.getenv('OPGUARD_LOCAL_DUMP_TARGET')
        if not target:
            raise ValueError('OPGUARD_LOCAL_DUMP_ENABLED requires OPGUARD_LOCAL_DUMP_TARGET')
        manager = enable_local_dump(target, float(os.getenv('OPGUARD_LOCAL_DUMP_SYNC_INTERVAL', '1800')))
    # Never silently fall back to direct shared-storage writes when staging fails.
    return manager.get_local_path(target_path) if manager else target_path
