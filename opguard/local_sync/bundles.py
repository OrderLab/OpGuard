"""Durable local batches. Cursors advance only after atomic remote publication."""
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile

VERSION = 1


class RecordTooLarge(ValueError):
    pass


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_state(spool):
    path = Path(spool) / 'checkpoint.json'
    return json.loads(path.read_text()) if path.exists() else {'sequence': 0, 'files': {}}


def read_manifest(bundle):
    with tarfile.open(bundle, 'r:') as tar:
        return json.load(tar.extractfile('manifest.json'))


def _tail(f, offset):
    f.seek(max(0, offset - 4096))
    return hashlib.sha256(f.read(min(offset, 4096))).hexdigest()


def _capture(path, payload, previous, limit, watermark=None):
    """Append logs are copied to a newline boundary; other files must stay stable."""
    with path.open('rb') as src:
        before = os.fstat(src.fileno())
        identity = [before.st_dev, before.st_ino]
        append = path.suffix in ('.jsonl', '.ndjson')
        offset = 0
        mode = 'replace'
        if append and previous and previous['identity'] == identity:
            old = previous['offset']
            if before.st_size >= old and _tail(src, old) == previous.get('tail'):
                offset, mode = old, 'append'
        elif not append and previous and previous['identity'] == identity:
            if previous['size'] == before.st_size and previous['mtime'] == before.st_mtime_ns:
                return None
        end = min(before.st_size, watermark) if watermark is not None else before.st_size
        if append and end <= offset:
            if not (mode == "replace" and previous and previous["offset"] and end == 0):
                return None
        src.seek(offset)
        # A non-log file is indivisible; the bundle size cap is soft for that file.
        count = min(max(0, end - offset), limit) if append else before.st_size
        with payload.open('wb') as dst:
            left = count
            while left:
                chunk = src.read(min(left, 1024 * 1024))
                if not chunk:
                    raise OSError(f'File truncated during snapshot: {path}')
                dst.write(chunk)
                left -= len(chunk)
        if append and count:
            # Retain only complete records, without loading the entire payload.
            with payload.open('r+b') as dst:
                scan_end = count
                while scan_end:
                    start = max(0, scan_end - 65536)
                    dst.seek(start)
                    block = dst.read(scan_end - start)
                    newline = block.rfind(b'\n')
                    if newline >= 0:
                        count = start + newline + 1
                        dst.truncate(count)
                        break
                    scan_end = start
                else:
                    if count < end - offset:
                        raise RecordTooLarge(f"JSONL record exceeds sync batch limit: {path}")
                    return None
            # Append writers may grow while copying; rotation/truncation is deferred.
            after = os.fstat(src.fileno())
            if after.st_size < offset + count:
                return None
            tail = _tail(src, offset + count)
        else:
            after = os.fstat(src.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise OSError(f"File changed during snapshot: {path}")
            tail = _tail(src, 0) if append else None
        current = path.stat()
        if [current.st_dev, current.st_ino] != identity:
            return None
        cursor = {'identity': identity, 'offset': offset + count, 'size': after.st_size,
                  'mtime': after.st_mtime_ns, 'tail': tail}
        return {'mode': mode, 'offset': offset, 'size': count, 'sha256': digest(payload), 'cursor': cursor}


def prepare(paths, spool, rank, local_rank, max_bytes, watermarks=None):
    """Return the pending batch first, including after a process restart."""
    spool = Path(spool)
    spool.mkdir(parents=True, exist_ok=True)
    pending = spool / 'pending.tar'
    if pending.exists():
        return pending
    state = load_state(spool)
    entries = []
    total = 0
    with tempfile.TemporaryDirectory(dir=spool) as tmp:
        tmp = Path(tmp)
        archive = tmp / 'bundle.tar'
        with tarfile.open(archive, 'w') as tar:
            for path in paths.owned_files(rank, local_rank):
                relative = path.relative_to(paths.files).as_posix()
                if watermarks is not None and relative not in watermarks:
                    continue
                payload = tmp / 'payload'
                try:
                    entry = _capture(path, payload, state['files'].get(relative), max_bytes - total,
                                     watermarks.get(relative) if watermarks is not None else None)
                except RecordTooLarge:
                    if entries:
                        break  # Retry this record with the next batch's full budget.
                    raise
                if entry is None:
                    continue
                entry.update(path=relative, member=f'data/{len(entries)}')
                if relative.startswith(('instrumentation/', 'exec_trace/source_files/')):
                    entry['restore_path'] = f'nodes/{paths.node}/{relative}'
                tar.add(payload, arcname=entry['member'], recursive=False)
                entries.append(entry)
                total += entry['size']
                if total >= max_bytes:
                    break
            if not entries:
                return None
            manifest = {'version': VERSION, 'sequence': state['sequence'] + 1, 'entries': entries}
            data = json.dumps(manifest).encode()
            info = tarfile.TarInfo('manifest.json')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        with archive.open('rb') as f:
            os.fsync(f.fileno())
        os.replace(archive, pending)
    return pending


def commit(spool, bundle):
    manifest = read_manifest(bundle)
    state = load_state(spool)
    # A crash between checkpoint persistence and removing pending is harmless.
    if manifest['sequence'] > state['sequence']:
        if manifest['sequence'] != state['sequence'] + 1:
            raise ValueError('Local checkpoint sequence gap')
        for entry in manifest['entries']:
            state['files'][entry['path']] = entry['cursor']
        state['sequence'] = manifest['sequence']
        atomic_json(Path(spool) / 'checkpoint.json', state)
    Path(bundle).unlink()
