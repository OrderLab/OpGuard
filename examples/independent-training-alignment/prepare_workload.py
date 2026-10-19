#!/usr/bin/env python3
"""Restore a companion training archive and apply an explicit repair stage."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent
STAGES = {
    "supplied": (),
    "four_fixes": ("01-training-semantics.patch",),
    "gelu_fixed": ("01-training-semantics.patch", "02-gelu-backward.patch"),
    "aligned": ("01-training-semantics.patch", "02-gelu-backward.patch", "03-adam-step.patch"),
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_checked(archive, directory, expected_manifest):
    with tarfile.open(archive, "r:gz") as stream:
        seen = set()
        for member in stream:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsupported archive member: {member.name}")
            target = directory / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            if str(relative) in seen:
                raise ValueError(f"Duplicate archive member: {member.name}")
            seen.add(str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                shutil.copyfileobj(stream.extractfile(member), output)
            target.chmod(member.mode & 0o777)
    original = directory / "manifest.json"
    if sha(original) != expected_manifest:
        raise ValueError("Original manifest changed")
    manifest = json.loads(original.read_text())
    members = [x for x in manifest["files"] if not x["path"].startswith("observed/")]
    if seen != {"manifest.json", *(x["path"] for x in members)}:
        raise ValueError("Companion archive has unexpected or missing members")
    for item in members:
        path = directory / item["path"]
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {item['path']}")
    original.rename(directory / "original-manifest.json")
    return manifest


def apply_patch(directory, patch):
    for extra in (("--check",), ()):
        subprocess.run(["git", "apply", "--no-index", "-p0", *extra, "-"],
                       input=patch, text=True, cwd=directory, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def portable_launchers(directory):
    changed = []
    names = ("launch.sh", "stack_a/launch.sh", "stack_b/launch.sh", "stack_b/run_training.sh")
    # Derive the interpreter from the hash-verified archive, not a private host path.
    interpreters = set()
    for name in names:
        interpreters.update(re.findall(r"/[A-Za-z0-9_./-]+/bin/python(?:3(?:\.\d+)?)?\b",
                                      (directory / name).read_text()))
    if len(interpreters) != 1:
        raise ValueError("Expected one archived Python interpreter across launchers")
    old_python = interpreters.pop()
    for name in names:
        path = directory / name
        original = text = path.read_text()
        if name == "stack_a/launch.sh":
            text = text.replace(old_python, "python3")
        elif name == "stack_b/launch.sh":
            text = text.replace(f'export PATH="{Path(old_python).parent}:$PATH"',
                                'export PATH="$(dirname "$(command -v "${PYTHON_BIN:-python3}")"):$PATH"')
            text = text.replace(old_python, '"${PYTHON_BIN:-python3}"')
        else:
            text = text.replace(old_python, '"${PYTHON_BIN:-python3}"')
        if text == original or old_python in text:
            raise ValueError(f"Unexpected launcher version: {name}")
        path.write_text(text)
        subprocess.run(["bash", "-n", str(path)], check=True)
        changed.append(name)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--stage", choices=STAGES, default="supplied")
    parser.add_argument("--capture", action="store_true",
                        help="Install the six recorded OpGuard observation sites; requires OpGuard at training time")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        parser.error("destination must not exist; each stage needs a separate workspace")
    assets = json.loads((ROOT / "evidence/assets.json").read_text())
    expected = assets["training_archive"]
    if args.archive.stat().st_size != expected["bytes"] or sha(args.archive) != expected["sha256"]:
        parser.error("archive does not match the pinned companion source/input archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alignment-prepare-", dir=destination.parent) as temporary:
        work = Path(temporary) / "work"
        work.mkdir()
        original = extract_checked(args.archive, work, expected["source_manifest_sha256"])
        for name in STAGES[args.stage]:
            apply_patch(work, (ROOT / "patches" / name).read_text())
        if args.capture:
            diagnostics = work / "diagnostics"
            diagnostics.mkdir()
            probe = (ROOT / "capture/probes.py").read_text()
            probe = probe.replace("ROOT=Path('/work/diagnostics')", "ROOT=Path(__file__).resolve().parent")
            (diagnostics / "probes.py").write_text(probe)
            number = {"supplied": 1, "four_fixes": 2, "gelu_fixed": 3, "aligned": 3}[args.stage]
            (diagnostics / "current.json").write_text(json.dumps({"round": number}) + "\n")
            patch = (ROOT / "capture/instrumentation.patch").read_text()
            patch = patch.replace('"/work/diagnostics"', repr(str(destination / "diagnostics")))
            apply_patch(work, patch)
        launchers = portable_launchers(work)
        changed = []
        for item in original["files"]:
            path = work / item["path"]
            if not path.is_file() or sha(path) == item["sha256"]:
                continue
            changed.append(item["path"])
            if path.suffix == ".py":
                ast.parse(path.read_text(), filename=item["path"])
            elif path.suffix == ".json":
                json.loads(path.read_text())
        if args.capture:
            ast.parse((work / "diagnostics/probes.py").read_text())
        preparation = {
            "original_candidate": original["id"], "stage": args.stage,
            "archive_sha256": expected["sha256"], "patches": list(STAGES[args.stage]),
            "capture_installed": args.capture, "interpreter_path_adaptations": launchers,
            "changed_original_files": changed,
            "scope": "Restored runnable inputs, not the original blind package: observed outputs are absent; answers/repair stages are explicit.",
            "training_executed_by_preparer": False,
        }
        (work / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
        manifest = {"id": f"example-{args.stage}", "files": []}
        for path in sorted(work.rglob("*")):
            if path.is_file():
                manifest["files"].append({"path": str(path.relative_to(work)),
                                          "bytes": path.stat().st_size, "sha256": sha(path)})
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(destination)
        work.rename(destination)
    print(json.dumps({"workspace": str(destination), **preparation}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.stderr or str(error))
    except (OSError, ValueError, tarfile.TarError) as error:
        raise SystemExit(f"Preparation failed: {error}")
