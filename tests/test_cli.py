"""CLI entry-point smoke tests."""

from __future__ import annotations

import pytest

from vorqd import cli
from vorqd.cli import build_parser, main

from .test_config import MINIMAL


def test_help_lists_config(capsys):
    parser = build_parser()
    help_text = parser.format_help()
    assert "--config" in help_text


def test_missing_config_exits_nonzero_naming_file(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--config", "does-not-exist.yaml"])
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert "does-not-exist.yaml" in err


def _serve_capturing(monkeypatch):
    seen = {}

    async def serve(config):
        seen["config"] = config

    monkeypatch.setattr(cli, "serve", serve)
    return seen


def test_the_document_in_the_environment_is_the_config(monkeypatch):
    seen = _serve_capturing(monkeypatch)
    monkeypatch.setenv("VORQD_CONFIG", MINIMAL)
    assert main([]) == 0
    assert seen["config"].models[0].model == "m:fp8"


def test_a_path_in_the_environment_is_refused_by_name(monkeypatch, capsys, tmp_path):
    p = tmp_path / "vorqd.yaml"
    p.write_text(MINIMAL)
    monkeypatch.setenv("VORQD_CONFIG", str(p))
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "VORQD_CONFIG" in err and "not a path" in err and "--config" in err


def test_the_flag_wins_over_the_environment(monkeypatch, tmp_path):
    seen = _serve_capturing(monkeypatch)
    p = tmp_path / "vorqd.yaml"
    p.write_text(MINIMAL.replace("m:fp8", "from-file:fp8"))
    monkeypatch.setenv("VORQD_CONFIG", MINIMAL)
    assert main(["--config", str(p)]) == 0
    assert seen["config"].models[0].model == "from-file:fp8"


def test_the_default_path_is_read_last(monkeypatch, tmp_path):
    seen = _serve_capturing(monkeypatch)
    p = tmp_path / "vorqd.yaml"
    p.write_text(MINIMAL.replace("m:fp8", "from-default:fp8"))
    monkeypatch.delenv("VORQD_CONFIG", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", str(p))
    assert main([]) == 0
    assert seen["config"].models[0].model == "from-default:fp8"


def test_nothing_at_all_names_the_three_places(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("VORQD_CONFIG", raising=False)
    monkeypatch.setattr(cli, "DEFAULT_CONFIG_PATH", str(tmp_path / "absent.yaml"))
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--config" in err and "VORQD_CONFIG" in err and "absent.yaml" in err
