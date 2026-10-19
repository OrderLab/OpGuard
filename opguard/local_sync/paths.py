"""Path mapping and ownership; never guess paths from string prefixes."""
from contextlib import contextmanager
import tempfile
import hashlib
import os
from pathlib import Path
import re
import socket


class ShadowPaths:
    def __init__(self, target, local_root=None):
        self.target = Path(target).expanduser().resolve()
        self.root = Path(local_root or os.getenv('OPGUARD_LOCAL_DUMP_BASE', '/tmp/opguard_local_dump')).expanduser().resolve()
        if self.target == self.root or self.root in self.target.parents or self.target in self.root.parents:
            raise ValueError('Local staging and shared target must not contain each other')
        key = hashlib.sha256(str(self.target).encode()).hexdigest()[:24]
        self.work = self.root / key
        self.files = self.work / 'files'
        self.files.mkdir(parents=True, exist_ok=True)
        node = os.getenv('OPGUARD_NODE_ID') or socket.gethostname()
        self.node = hashlib.sha256(node.encode()).hexdigest()[:16]

    def translate(self, path):
        path = Path(path).expanduser().resolve()
        if path == self.files or self.files in path.parents:
            return str(path)
        try:
            relative = path.relative_to(self.target)
        except ValueError:
            raise ValueError(f'Output {path} is outside local-dump target {self.target}') from None
        return str(self.files / relative)

    def owned_files(self, rank, local_rank):
        """Each rank owns execution data; node leader owns common metadata."""
        gpu = self.files / 'exec_trace' / f'gpu_{rank:03d}'
        roots = [gpu]
        if local_rank == 0:
            roots += [self.files / 'instrumentation', self.files / 'exec_trace' / 'source_files']
        calls = self.files / 'callsite_trace'
        if calls.is_dir():
            for path in sorted(calls.rglob('*')):
                match = re.search(r'(?:^|_)rank(\d+)(?:_|\.)', path.name)
                if (match and int(match[1]) == rank) or (not match and local_rank == 0):
                    if self._regular(path):
                        yield path
        for root in roots:
            if root.is_dir() and not root.is_symlink():
                for path in sorted(root.rglob('*')):
                    if self._regular(path):
                        yield path

    def _regular(self, path):
        return (path.is_file() and not path.is_symlink()
                and self.files in path.resolve().parents
                and not any(p.startswith('.') for p in path.relative_to(self.files).parts)
                and not path.name.endswith(('.tmp', '.part', '.lock')))


@contextmanager
def atomic_output(path):
    """Publish complete non-append producer files; snapshots ignore the temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', suffix='.part', dir=path.parent)
    os.close(fd)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
