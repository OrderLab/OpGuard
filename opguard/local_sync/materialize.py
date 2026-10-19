"""Reconstruct a published snapshot on local disk for existing analysis tools.

Usage: python -m opguard.local_sync.materialize SHARED_RUN LOCAL_OUTPUT
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from .bundles import VERSION, digest

BUNDLE_DIR = '.opguard-bundles'


def has_bundles(source):
    return (Path(source) / BUNDLE_DIR).is_dir()


def _apply(bundle, root, expected_sequence):
    # File names carry the complete archive digest, verified before applying anything.
    if digest(bundle) != bundle.stem.split('-', 1)[1]:
        raise ValueError(f'Bundle checksum mismatch: {bundle}')
    with tarfile.open(bundle, 'r:') as tar:
        manifest = json.load(tar.extractfile('manifest.json'))
        if manifest.get('version') != VERSION or manifest['sequence'] != expected_sequence:
            raise ValueError(f'Unsupported format or missing/duplicate batch: {bundle}')
        for entry in manifest['entries']:
            relative = PurePosixPath(entry.get('restore_path', entry['path']))
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or not relative.parts:
                raise ValueError('Unsafe bundle path')
            target = root.joinpath(*relative.parts)
            if root not in target.resolve().parents:
                raise ValueError('Bundle path escapes output')
            member = tar.getmember(entry['member'])
            if not member.isfile() or member.size != entry['size']:
                raise ValueError('Invalid bundle payload')
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = entry['mode']
            if mode not in ('append', 'replace') or entry['offset'] < 0:
                raise ValueError('Invalid bundle operation')
            if mode == 'append':
                size = target.stat().st_size if target.exists() else 0
                if size != entry['offset']:
                    raise ValueError(f'Missing or conflicting log prefix: {relative}')
            elif entry['offset'] != 0:
                raise ValueError('Replacement must start at zero')
            h = hashlib.sha256()
            with tar.extractfile(member) as src, target.open('ab' if mode == 'append' else 'wb') as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b''):
                    h.update(chunk)
                    dst.write(chunk)
            if h.hexdigest() != entry['sha256']:
                raise ValueError(f'Payload checksum mismatch: {relative}')


def materialize(source, destination):
    """Build into a temporary directory; publish the local result only on success."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Restore destination must not exist: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.opguard-restore-', dir=destination.parent) as temp:
        root = Path(temp) / 'snapshot'
        root.mkdir()
        for owner in sorted((source / BUNDLE_DIR).iterdir()):
            if not owner.is_dir() or owner.is_symlink():
                continue
            for sequence, bundle in enumerate(sorted(owner.glob('*.tar')), 1):
                if bundle.is_symlink():
                    raise ValueError('Bundle must not be a symlink')
                _apply(bundle, root, sequence)
        # Metadata is uploaded once per node. Embed that node's version in each
        # rank directory locally, so existing source-aware comparison still works.
        for owner in (source / BUNDLE_DIR).iterdir():
            if not owner.is_dir() or '-rank' not in owner.name:
                continue
            node, rank = owner.name.rsplit('-rank', 1)
            if not rank.isdigit():
                continue
            gpu = root / 'exec_trace' / f'gpu_{int(rank):03d}'
            if not gpu.is_dir():
                continue
            for relative in ('exec_trace/source_files', 'instrumentation'):
                metadata = root / 'nodes' / node / relative
                if metadata.is_dir():
                    shutil.copytree(metadata, gpu / metadata.name, dirs_exist_ok=True)
        os.replace(root, destination)
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(materialize(args.source, args.destination))


if __name__ == '__main__':
    main()
