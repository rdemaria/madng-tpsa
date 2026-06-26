from __future__ import annotations

import sys
import types

from madng_tpsa import _cffi


def test_pymadng_binary_candidates_are_discovered(monkeypatch, tmp_path):
    package_dir = tmp_path / "pymadng"
    bin_dir = package_dir / "bin"
    bin_dir.mkdir(parents=True)
    executable = bin_dir / "mad_Linux"
    executable.write_text("not a real binary")

    fake = types.ModuleType("pymadng")
    fake.__file__ = str(package_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "pymadng", fake)
    monkeypatch.setattr(_cffi.platform, "system", lambda: "Linux")

    candidates = list(_cffi._pymadng_candidate_paths())
    assert str(executable) in candidates
