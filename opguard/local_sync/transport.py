"""Only sequential, rate-limited writes touch shared storage. Locks stay local."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time
import uuid


_lock_files = set()


def _after_fork():
    # Close inherited lock handles without affecting the parent's descriptors.
    for lock in tuple(_lock_files):
        lock.close()
    _lock_files.clear()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def local_lock(path, stop=None, timeout=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+b') as lock:
        _lock_files.add(lock)
        try:
            start = time.monotonic()
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if stop is not None and stop.wait(0.05):
                        raise InterruptedError('Upload stopped while waiting for local lock')
                    if stop is None:
                        time.sleep(0.05)
                    if timeout is not None and time.monotonic() - start >= timeout:
                        raise TimeoutError('Local sync already in progress')
            try:
                yield
            finally:
                if not lock.closed:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        finally:
            _lock_files.discard(lock)


def publish(source, destination, node_lock, bytes_per_second, stop=None):
    """Readers see either no bundle or a complete bundle, never a partial file."""
    destination = Path(destination)
    with local_lock(node_lock, stop):
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name('.' + destination.name + '.' + uuid.uuid4().hex + '.part')
        try:
            start = time.monotonic()
            total = 0
            with open(source, 'rb') as src, partial.open('xb') as dst:
                while chunk := src.read(1024 * 1024):
                    if stop is not None and stop.is_set():
                        raise InterruptedError('Upload stopped; local bundle retained')
                    dst.write(chunk)
                    total += len(chunk)
                    delay = total / bytes_per_second - (time.monotonic() - start)
                    if delay > 0:
                        if stop is None:
                            time.sleep(delay)
                        elif stop.wait(delay):
                            raise InterruptedError('Upload stopped; local bundle retained')
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(partial, destination)
            return total
        finally:
            partial.unlink(missing_ok=True)
