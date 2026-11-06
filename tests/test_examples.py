"""The examples run (documentation does not rot)."""
import pathlib
import subprocess
import sys

import pytest

EX = pathlib.Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("script", ["01_quickstart.py", "02_generalized_overlap.py", "03_phonons_thermo.py", "04_windows_ipr.py", "05_projections.py", "06_fullerene_to_graphene.py"])
def test_example_runs(script, tmp_path):
    r = subprocess.run([sys.executable, str(EX / script)], cwd=tmp_path, capture_output=True, text=True, timeout=600,
                       env={**__import__("os").environ, "PYTHONPATH": str(EX.parent / "tests")})
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip()
