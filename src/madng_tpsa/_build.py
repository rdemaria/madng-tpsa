"""CFFI build script for the bundled MAD-NG-compatible TPSA backend."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cffi import FFI

# Importing madng_tpsa during build would try to import the extension being
# built, so load the declaration string directly from the sibling file.
_ns: dict[str, object] = {}
exec(Path(__file__).with_name("_cdefs.py").read_text(), _ns)
CDEF = str(_ns["CDEF"])

ffibuilder = FFI()
ffibuilder.cdef(CDEF)

extra_compile_args = ["-O2"]
if sys.platform != "win32":
    extra_compile_args.append("-std=c99")
extra_link_args: list[str] = []

def _split_libs(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


# The bundled TPSA backend uses LAPACK for map inversion (mad_tpsa_minv).
# Override with MADNG_TPSA_LAPACK_LIBRARIES="openblas" or similar when needed.
lapack_libraries = _split_libs(os.environ.get("MADNG_TPSA_LAPACK_LIBRARIES"))
if not lapack_libraries:
    if sys.platform == "darwin":
        extra_link_args.extend(["-framework", "Accelerate"])
    else:
        lapack_libraries = ["lapack", "blas"]

libraries = []
if sys.platform not in {"darwin", "win32"}:
    libraries.append("m")
libraries += lapack_libraries

ffibuilder.set_source(
    "madng_tpsa._madng_tpsa_cffi",
    '#include "madng_tpsa_backend.h"\n',
    sources=["src/madng_tpsa/vendor/madng_tpsa_backend.c"],
    include_dirs=["src/madng_tpsa/vendor"],
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
