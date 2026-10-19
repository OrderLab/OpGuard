import yaml
from opguard.utils.config_loader import create_default_opguard_config, load_opguard_config


def test_default_has_no_project_targets(monkeypatch):
    monkeypatch.delenv("OPGUARD_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OPGUARD_CALLSITE_CONFIG", raising=False)
    config = load_opguard_config()
    assert config == {k: [] for k in ("entry", "meta_vars", "instrumentations", "callsite_tracer")}


def test_generated_default_matches_shipped_default(tmp_path, monkeypatch):
    monkeypatch.delenv("OPGUARD_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OPGUARD_CALLSITE_CONFIG", raising=False)
    path = tmp_path / "config.yaml"
    create_default_opguard_config(str(path))
    assert yaml.safe_load(path.read_text()) == load_opguard_config()


def test_user_configuration_still_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("OPGUARD_CONFIG_PATH", raising=False)
    monkeypatch.delenv("DEFAULT_CODE_REPO_BASE", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text("instrumentations:\n  - file_path: train.py\n    function: matmul\n")
    config = load_opguard_config(str(path))
    assert config["instrumentations"][0]["file_path"] == str(tmp_path / "train.py")
    assert config["entry"] == []
