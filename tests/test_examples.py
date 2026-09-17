"""Run each documented command-line example in a fresh Python process."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_examples_run_successfully(example):
    # Do not leak pytest's argv into examples that use argparse. A fresh process
    # also isolates C descriptor lifetime and backend-loader state.
    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=EXAMPLES.parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
