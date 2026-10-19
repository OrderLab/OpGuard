"""The doctor must distinguish optional acceleration from a broken base install."""
import json
from types import SimpleNamespace

import pytest

from opguard import doctor


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    package = tmp_path / "opguard"
    package.mkdir()
    monkeypatch.setattr(doctor, "__file__", str(package / "doctor.py"))
    torch = SimpleNamespace(__version__="2.9.1+cu128", version=SimpleNamespace(cuda="12.8"),
                            __file__=str(tmp_path / "torch/__init__.py"),
                            cuda=SimpleNamespace(is_available=lambda: False),
                            compiled_with_cxx11_abi=lambda: True)
    monkeypatch.setitem(doctor.sys.modules, "torch", torch)
    monkeypatch.setattr(doctor, "_module", lambda name, symbols=():
                        {"available": True, "path": name} if name in ("torch", "opguard.tracer.xor_signature")
                        else {"available": False, "reason": "not installed"})
    monkeypatch.setattr(doctor, "_library", lambda paths: {"available": False, "reason": "not installed"})
    return package


def test_base_install_is_success_without_native(runtime, capsys):
    assert doctor.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["base_ready"] and not report["native_ready"]
    assert report["torch"]["cuda"] == "12.8"
    assert not report["torch"]["cuda_available"]
    assert doctor.main(["--require-native", "--json"]) == 1
    json.loads(capsys.readouterr().out)


def test_missing_torch_is_actionable(monkeypatch, runtime, capsys):
    monkeypatch.setattr(doctor, "_module", lambda *args: {"available": False, "reason": "No module named 'torch'"})
    assert doctor.main([]) == 1
    assert "No module named 'torch'" in capsys.readouterr().out


def test_loadable_but_incompatible_native_build(monkeypatch, runtime, capsys):
    monkeypatch.setattr(doctor, "_module", lambda *args: {"available": True, "path": "installed"})
    monkeypatch.setattr(doctor, "_library", lambda *args: {"available": True, "path": "installed"})
    (runtime / "_native_build.json").write_text(json.dumps({"torch": "2.8.0+cu126", "torch_cuda": "12.6", "cxx11_abi": False}))
    assert doctor.main(["--require-native", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert len(report["compatibility_errors"]) == 3
    assert not report["native_ready"]


def test_extension_import_error_is_reported(monkeypatch):
    def broken(name):
        raise OSError("undefined symbol: torch_abi")
    monkeypatch.setattr(doctor.importlib, "import_module", broken)
    assert doctor._module("xor_reduce_cuda") == {"available": False, "reason": "OSError: undefined symbol: torch_abi"}
