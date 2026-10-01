"""The isolation guarantee is a code rule; these tests are its enforcement.

Everything TEE lives behind one import seam. Core modules never import
``vorqd/tee``, and a config that flags no confidential model must never execute a
line of it — the daemon runs unchanged on a machine that has no attestation
story at all.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import vorqd

# EVERY module of the package except the seam file — derived, never hand-listed. A
# curated list only ever covers the violations someone thought of: it left
# `__init__.py` uncovered, where a tee import is the WORST case (every `import vorqd`
# would execute the whole tee package, breaking the guarantee for every user, not
# just confidential configs). `cli.py` is the one legitimate importer.
SEAM_FILE = "cli.py"
CORE = sorted(p.name for p in Path(vorqd.__file__).parent.glob("*.py") if p.name != SEAM_FILE)

# A dynamic import is still an import: catch the spelling static analysis cannot.
_DYNAMIC_TEE_IMPORT = re.compile(r"""import_module\(\s*["'][^"']*tee""")


def _tee_imports(src: str) -> list[str]:
    """Every static import of the tee package in ``src``, whatever the spelling.

    Walks the AST rather than grepping for one phrasing: ``from .tee import X``,
    ``from vorqd.tee import X``, ``import vorqd.tee``, ``from . import tee`` and
    ``from vorqd import tee`` are all the same violation, and a rule enforced
    against a single spelling is a rule that gets around.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names
                      if a.name == "vorqd.tee" or a.name.startswith("vorqd.tee.")]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in ("tee", "vorqd.tee") or module.startswith(("tee.", "vorqd.tee.")):
                found.append(module)
            elif module in ("", "vorqd"):   # `from . import tee` / `from vorqd import tee`
                found += [f"{module}.tee" for a in node.names if a.name == "tee"]
    return found


def test_core_modules_keep_zero_tee_imports():
    src_dir = Path(vorqd.__file__).parent
    assert "__init__.py" in CORE and SEAM_FILE not in CORE   # the derivation itself
    for name in CORE:
        src = (src_dir / name).read_text()
        assert not _tee_imports(src), f"{name} imports vorqd/tee — the isolation rule forbids it"
        assert not _DYNAMIC_TEE_IMPORT.search(src), (
            f"{name} imports vorqd/tee dynamically — the isolation rule forbids it"
        )


# --- config fixtures ---------------------------------------------------------

_CONFIDENTIAL_MODEL = (
    "  - model: org/e2ee-model:fp8\n    confidential: true\n"
    "    slas: {\"1h\": {rate_out: \"0.1\"}}\n"
    "    backend:\n      preset: openai-chat\n      base_url: http://x/v1\n      model: rid\n"
)
_PLAIN_MODEL = (
    "  - model: plain/model:fp8\n    slas: {\"1h\": {rate_out: \"0.1\"}}\n"
    "    backend:\n      preset: openai-chat\n      base_url: http://x/v1\n      model: rid\n"
)
_CONFIDENTIAL = "org/e2ee-model:fp8"
_PLAIN = "plain/model:fp8"
_BOX_KEY = "aa" * 32   # a real 32-byte Curve25519 key: build_daemon constructs BoxCipher from it


def _write_config(tmp_path, models: str, *, box_key: str | None = None) -> Path:
    """A ``vorqd.yaml`` serving ``models``. A confidential config passes no box key:
    in that mode it is ephemeral, generated in guest memory each boot."""
    path = tmp_path / "vorqd.yaml"
    path.write_text(
        "provider:\n  api_url: http://localhost:8402\n  capacity: 1\n  wallet_key: env:WK\n"
        # Under tmp_path, so building a daemon here writes no sqlite file into the repo.
        + f"  state_db: {path.parent / 'state.sqlite'}\n"
        + (f'  box_key: "{box_key}"\n' if box_key else "")
        + "models:\n" + models
    )
    return path


# `vorqd.cli` is purged alongside the tee package by the tests below. It is the module
# that HOLDS the import seam, and it is deliberately absent from CORE — so a tee import
# moved to its module level is the single most likely regression of this rule. Left
# cached by an earlier test module, `from vorqd.cli import build_daemon` would not
# re-execute it, and that regression would never show up in sys.modules at all.
_SEAM_MODULES = ("vorqd.cli",)


def _purge_import_state() -> dict:
    saved = {name: mod for name, mod in sys.modules.items()
             if name in _SEAM_MODULES or name.startswith("vorqd.tee")}
    for name in saved:
        del sys.modules[name]
    return saved


def test_unflagged_config_never_imports_tee(tmp_path, monkeypatch):
    # Sibling tests import vorqd.tee, so clear it first: the question is whether
    # *this* build_daemon pulls it back in.
    saved = _purge_import_state()
    try:
        from vorqd.cli import build_daemon
        from vorqd.config import load_config

        monkeypatch.setenv("WK", "0x" + "11" * 32)
        build_daemon(load_config(_write_config(tmp_path, _PLAIN_MODEL, box_key=_BOX_KEY)))
        assert not any(name.startswith("vorqd.tee") for name in sys.modules)
    finally:
        # Restore the originals: a duplicate copy of either module left behind would
        # hand later test modules a second set of class objects.
        sys.modules.update(saved)


def test_confidential_config_takes_the_import_seam(tmp_path, monkeypatch):
    """The mirror image: a flagged config DOES reach vorqd/tee (via cli's lazy
    import) for this boot's attested identity — so the isolation above is a seam,
    not a dead end. The model's driver is the ordinary one: confidential is a
    property of the daemon, not of the backend."""
    saved = _purge_import_state()
    try:
        from vorqd.backend import BackendDriver
        from vorqd.cli import build_daemon
        from vorqd.config import load_config

        monkeypatch.setenv("WK", "0x" + "22" * 32)
        daemon = build_daemon(load_config(_write_config(tmp_path, _CONFIDENTIAL_MODEL + _PLAIN_MODEL)))
        assert any(name.startswith("vorqd.tee") for name in sys.modules)

        assert type(daemon._sched._drivers[_CONFIDENTIAL]) is BackendDriver
        assert type(daemon._sched._drivers[_PLAIN]) is BackendDriver
        # The boot identity is ephemeral and evidence-bound: no configured box key,
        # yet the scheduler has a cipher and the evidence that binds it to this boot.
        assert daemon._sched._cipher is not None
        assert daemon._sched._evidence["type"] == "mock-cvm-v1"
        assert daemon._sched._evidence["report_data"]
    finally:
        sys.modules.update(saved)
