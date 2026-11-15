"""Rule 3: ``ullm.core`` never imports ``mlx``.

The core holds every scheduling and memory decision, and it must run and be
tested on any machine. Two checks: no module in ``core`` imports MLX directly,
and importing the whole of ``core`` does not pull MLX in through another module.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[2] / "src" / "ullm" / "core"
FORBIDDEN_ROOTS = {"mlx", "mlx_lm"}


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def _core_files() -> list[Path]:
    files = sorted(CORE.rglob("*.py"))
    assert files, f"no Python files under {CORE}"
    return files


@pytest.mark.parametrize("path", _core_files(), ids=lambda p: str(p.relative_to(CORE)))
def test_core_module_does_not_import_mlx_directly(path: Path) -> None:
    assert not _imported_roots(path) & FORBIDDEN_ROOTS


def test_importing_core_does_not_load_mlx() -> None:
    # A fresh interpreter, so MLX loaded by another test cannot hide a leak.
    probe = (
        "import importlib, pkgutil, sys\n"
        "import ullm.core\n"
        "for m in pkgutil.walk_packages(ullm.core.__path__, 'ullm.core.'):\n"
        "    importlib.import_module(m.name)\n"
        "leaks = sorted(n for n in sys.modules if n.split('.')[0] in {'mlx', 'mlx_lm'})\n"
        "print(','.join(leaks))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"importing ullm.core loaded: {out}"


def test_the_checker_catches_an_mlx_import(tmp_path: Path) -> None:
    # Guard against the static check silently passing because it parses nothing.
    sample = tmp_path / "bad.py"
    sample.write_text("import os\nfrom mlx.core import array\nimport mlx_lm.models\n")
    assert _imported_roots(sample) & FORBIDDEN_ROOTS == {"mlx", "mlx_lm"}
