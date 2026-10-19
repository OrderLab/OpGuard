"""Lifecycle only: snapshot locally, publish one batch, commit its cursor."""
import logging
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

from opguard.distributed import get_global_rank, get_local_rank
from . import bundles, transport
from .materialize import BUNDLE_DIR
from .paths import ShadowPaths

logger = logging.getLogger(__name__)
_jitter_rng = random.SystemRandom()
_periodic_sync_registry = {}
_periodic_sync_registry_lock = threading.Lock()


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return value


class LocalDumpManager:
    def __init__(self, target_dir, sync_interval_sec=1800.0):
        self.paths = ShadowPaths(target_dir)
        self.target_dir, self.local_base = str(self.paths.target), str(self.paths.files)
        self.sync_interval_sec = _positive(sync_interval_sec, 'sync interval')
        self.max_bytes = max(1, int(_positive(os.getenv('OPGUARD_SYNC_BATCH_MB', '256'), 'batch MB') * 1024 ** 2))
        self.bytes_per_second = _positive(os.getenv('OPGUARD_SYNC_MBPS', '32'), 'upload MB/s') * 1024 ** 2
        self._pid = os.getpid()
        self._rank = None
        self._local_rank = None
        self._sync_thread = None
        self._stop_sync = threading.Event()
        self._last_sync_time = None
        self._stats = dict(files_synced=0, bytes_synced=0, sync_operations=0, errors=0)
        self._lock = threading.Lock()

    def get_local_path(self, target_path):
        return self.paths.translate(target_path)

    def _identity(self):
        if self._pid != os.getpid():
            raise RuntimeError('Create a new local dump manager after fork')
        if self._rank is None:
            self._rank = get_global_rank()
            self._local_rank = get_local_rank()
            if self._local_rank is None:
                logger.warning("Local rank is unknown; node metadata upload is skipped")
        rank = self._rank
        owner = f'{self.paths.node}-rank{rank:06d}'
        spool = self.paths.work / 'spool' / owner
        return rank, owner, spool

    def sync_all(self, force=True, exact_match=True):
        """Upload at most one bounded batch periodically; drain current data when forced.

        exact_match remains an API compatibility argument. Published history is never
        deleted; no remote directory traversal or cross-node locking is required.
        """
        rank, owner, spool = self._identity()
        stop = None if force else self._stop_sync
        if (not force and self._last_sync_time is not None
                and time.monotonic() - self._last_sync_time < self.sync_interval_sec):
            return
        # Local lock serializes CLI final sync, background sync, and retries for this owner.
        with transport.local_lock(spool / 'owner.lock', stop, timeout=30):
            if (not (spool / 'checkpoint.json').exists() and not (spool / 'pending.tar').exists()
                    and (self.paths.target / BUNDLE_DIR / owner).exists()):
                raise RuntimeError('Local upload checkpoint is missing for an existing run; restore staging or use a new run ID')
            watermarks = None
            if force:
                watermarks = {p.relative_to(self.paths.files).as_posix(): p.stat().st_size
                              for p in self.paths.owned_files(rank, self._local_rank)}
            while True:
                pending = bundles.prepare(self.paths, spool, rank, self._local_rank, self.max_bytes, watermarks)
                if pending is None:
                    break
                manifest = bundles.read_manifest(pending)
                name = f"{manifest['sequence']:012d}-{bundles.digest(pending)}.tar"
                target = self.paths.target / BUNDLE_DIR / owner / name
                try:
                    count = transport.publish(pending, target, self.paths.root / '.upload.lock',
                                              self.bytes_per_second, stop)
                    bundles.commit(spool, pending)
                    if watermarks is not None:
                        for entry in manifest["entries"]:
                            path = entry["path"]
                            if (not path.endswith((".jsonl", ".ndjson")) or
                                    entry["cursor"]["offset"] >= watermarks.get(path, 0)):
                                watermarks.pop(path, None)
                except Exception:
                    with self._lock:
                        self._stats['errors'] += 1
                    raise  # Pending batch and original files remain available for retry.
                with self._lock:
                    self._stats['sync_operations'] += 1
                    self._stats['files_synced'] += len(manifest['entries'])
                    self._stats['bytes_synced'] += count
                if not force:
                    break
        self._last_sync_time = time.monotonic()

    def sync_step(self, step_name):
        self.sync_all(force=True)

    def start_periodic_sync(self):
        rank, _, _ = self._identity()
        key = (self._pid, self.target_dir, rank)
        with _periodic_sync_registry_lock:
            if key in _periodic_sync_registry:
                return
            self._stop_sync.clear()
            _periodic_sync_registry[key] = self
            self._registry_key = key
            self._sync_thread = threading.Thread(target=self._sync_worker, name='opguard-upload', daemon=True)
            self._sync_thread.start()

    def _sync_worker(self):
        # Stagger every cycle, including the first, to avoid synchronized bursts.
        try:
            while not self._stop_sync.wait(self.sync_interval_sec * _jitter_rng.uniform(1.0, 1.2)):
                try:
                    self.sync_all(force=False)
                except InterruptedError:
                    break
                except Exception:
                    logger.exception('Local sync failed; pending data retained for next attempt')
        finally:
            with _periodic_sync_registry_lock:
                if _periodic_sync_registry.get(self._registry_key) is self:
                    del _periodic_sync_registry[self._registry_key]

    def stop_periodic_sync(self):
        self._stop_sync.set()
        thread = self._sync_thread
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                return False
            self._sync_thread = None
        return True

    def close(self):
        if not self.stop_periodic_sync():
            raise TimeoutError('Upload is still blocked in I/O; local data retained; final sync not started')
        # Never create/import a tracing backend just to shut it down.
        runtime = sys.modules.get('opguard.runtime_tracer')
        buffer_module = sys.modules.get('opguard.tracer.trace_buffer')
        buffer = getattr(buffer_module, 'TRACE_BUFFER', None)
        if buffer is not None and not buffer.closed:
            if runtime is not None:
                runtime.flush_trace_buffer(force_dump=True)
            buffer.stop()
        self.sync_all(force=True)

    def cleanup(self):
        """Stop background work; staging data is deliberately retained for recovery."""
        if not self.stop_periodic_sync():
            raise TimeoutError('Upload still in progress')

    def get_stats(self):
        with self._lock:
            return self._stats.copy()

    def log_stats(self):
        logger.info('Local upload statistics: %s', self.get_stats())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
