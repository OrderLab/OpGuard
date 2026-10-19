"""Single-writer experiment records with atomic, readable snapshots."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import uuid


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class Store:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, experiment_id):
        if not isinstance(experiment_id, str) or not re.fullmatch(r'[a-f0-9]{32}', experiment_id):
            raise ValueError('Invalid experiment ID')
        return self.root / experiment_id

    def load(self, experiment_id):
        return json.loads((self.path(experiment_id) / 'experiment.json').read_text())

    def save(self, record):
        atomic_json(self.path(record['id']) / 'experiment.json', record)

    @contextlib.contextmanager
    def locked(self, experiment_id):
        with (self.path(experiment_id) / '.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Experiment is executing in another process')
            yield self.load(experiment_id)
