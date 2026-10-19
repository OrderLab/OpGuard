"""Fetch a case's checksum-pinned public artifacts; never execute their contents."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(manifest_path, destination):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for item in manifest['artifacts']:
        name = item['name']
        if Path(name).name != name or name in ('', '.', '..'):
            raise ValueError('Artifact name must be one path component')
        target = root / name
        if target.is_symlink():
            raise ValueError('Artifact destination must not be a symlink')
        if target.exists():
            data = target.read_bytes()
        else:
            if not item['url'].startswith('https://'):
                raise ValueError('Expected an HTTPS artifact URL')
            with urllib.request.urlopen(item['url'], timeout=60) as response:
                data = response.read(64 * 1024 * 1024 + 1)
            if len(data) > 64 * 1024 * 1024:
                raise ValueError('Download exceeds 64 MiB')
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {name}; inspect the changed asset, do not silently update expectations')
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(data)
        result.append(str(target))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(fetch(args.manifest, args.destination), indent=2))
