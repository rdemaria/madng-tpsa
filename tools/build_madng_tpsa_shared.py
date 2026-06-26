#!/usr/bin/env python3
"""Build a shared library exporting MAD-NG TPSA symbols from a MAD-NG checkout.

This helper expects an upstream MAD-NG source checkout. It does not vendor or
modify MAD-NG; it runs MAD-NG's own makefile to produce libmad.a, then links a
small shared object suitable for CFFI ``dlopen``.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
from pathlib import Path


def platform_makefile() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "Makefile.macosx"
    if system == "linux":
        return "Makefile.linux"
    raise SystemExit(f"unsupported platform for this helper: {platform.system()}")


def default_output() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "libmadng_tpsa.dylib"
    if system == "linux":
        return "libmadng_tpsa.so"
    return "madng_tpsa.dll"


def build(src: Path, output: Path, makefile: str, cc: str, extra_ldflags: list[str]) -> None:
    src = src.resolve()
    output = output.resolve()
    if not (src / makefile).exists():
        raise SystemExit(f"{src / makefile} does not exist; pass the MAD-NG src directory")

    subprocess.run(["make", "-f", makefile, "libmad.a"], cwd=src, check=True)
    archive = src / "libmad.a"
    if not archive.exists():
        raise SystemExit(f"expected {archive} after make")

    if platform.system().lower() == "darwin":
        command = [
            cc,
            "-dynamiclib",
            "-o",
            str(output),
            "-Wl,-all_load",
            str(archive),
            "-Wl,-noall_load",
            "-lm",
            "-llapack",
            "-lblas",
        ]
    else:
        command = [
            cc,
            "-shared",
            "-o",
            str(output),
            "-Wl,--whole-archive",
            str(archive),
            "-Wl,--no-whole-archive",
            "-lm",
            "-ldl",
            "-lgfortran",
            "-lquadmath",
            "-lgomp",
            "-lstdc++",
            "-llapack",
            "-lblas",
        ]
    command.extend(extra_ldflags)
    subprocess.run(command, cwd=src, check=True)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("madng_src", type=Path, help="path to MAD-NG/src")
    parser.add_argument("-o", "--output", type=Path, default=Path(default_output()))
    parser.add_argument("--makefile", default=platform_makefile())
    parser.add_argument("--cc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--ldflag", action="append", default=[], help="additional linker flag; may be repeated")
    args = parser.parse_args()
    build(args.madng_src, args.output, args.makefile, args.cc, args.ldflag)


if __name__ == "__main__":
    main()
