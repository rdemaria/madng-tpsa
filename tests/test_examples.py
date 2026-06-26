from __future__ import annotations

import runpy
from pathlib import Path


def test_examples_run_successfully():
    examples_dir = Path(__file__).resolve().parents[1] / "examples"

    for example in sorted(examples_dir.glob("*.py")):
        runpy.run_path(str(example), run_name="__main__")
