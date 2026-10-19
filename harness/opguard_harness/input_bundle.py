"""Copy an explicit input allowlist without repository history or answer files.

This is an artifact boundary, not a filesystem sandbox or a semantic leak detector.
The caller must review the selected source and provide a separate solver workspace.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys


EXCLUDED_PARTS = {".git", ".hg", ".svn", "__pycache__", "answers", "evidence"}


def disallowed_symlink(path):
    if not path.is_symlink():
        return False
    # Only exempt macOS root aliases with their exact system targets.
    # User-created directory and file links must still be rejected.
    return not (sys.platform == "darwin" and path in {Path("/var"), Path("/tmp")}
                and path.resolve() == Path("/private") / path.name)


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or "\\" in value
            or any(part in {".", ".."} | EXCLUDED_PARTS for part in value.split("/"))
            or str(path) != value or path.suffix in {".pyc", ".pyo"}):
        raise ValueError(f"Not an input path: {value}")
    return Path(*path.parts)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_inputs(destination, files):
    """Export {relative destination: explicit source file}; never walk a checkout.

    The public manifest contains only destination paths and content hashes. Source
    paths and case-to-answer mappings belong in the caller's private provenance.
    """
    destination = Path(destination)
    prepared = []
    for name, source in files.items():
        target = relative_path(name)
        if name == "manifest.json":
            raise ValueError("manifest.json is reserved for the input inventory")
        source = Path(source).absolute()
        if any(disallowed_symlink(p) for p in (source, *source.parents)) or not source.is_file():
            raise ValueError(f"Input must be a regular file without symlinks: {source}")
        if any(part in EXCLUDED_PARTS for part in source.parts):
            raise ValueError(f"Excluded source path: {source}")
        prepared.append((target, source))
    # Refuse merges into old bundles: stale captures may contain embedded answers.
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for target, source in prepared:
        output = destination / target
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        hashes[target.as_posix()] = sha256(output)
    manifest = {"version": 1, "files": hashes}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_inputs(directory):
    """Reject missing/changed/unlisted files and symlinks before solver delivery."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported input manifest version")
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in input bundle: {path}")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != expected:
        raise ValueError(f"Input inventory differs: {sorted(actual ^ expected)}")
    for name, digest in manifest["files"].items():
        if sha256(directory / relative_path(name)) != digest:
            raise ValueError(f"Changed input: {name}")
    return manifest
