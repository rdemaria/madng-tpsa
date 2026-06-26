"""Lazy CFFI access to a MAD-NG-compatible TPSA C API.

By default this package uses the compiled, vendored CFFI extension shipped with
``madng-tpsa``.  Advanced users may still set ``MADNG_TPSA_LIBRARY`` or pass an
explicit path to :func:`load_library` to use an external MAD-NG-compatible shared
library instead.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import sys
import importlib
from pathlib import Path
from typing import Iterable

from cffi import FFI

from ._cdefs import CDEF
from .exceptions import MadngLibraryError

try:  # Prefer the self-contained CFFI extension generated at install/build time.
    _vendored_cffi = importlib.import_module(__package__ + "._madng_tpsa_cffi")  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - only hit from an unbuilt source tree
    _vendored_cffi = None
    _vendored_import_error: Exception | None = exc
    ffi = FFI()
    ffi.cdef(CDEF)
else:
    _vendored_import_error = None
    ffi = _vendored_cffi.ffi

_REQUIRED_SYMBOLS = (
    "mad_desc_newv",
    "mad_desc_getnv",
    "mad_tpsa_newd",
    "mad_tpsa_del",
    "mad_tpsa_add",
    "mad_tpsa_sin",
    "mad_tpsa_compose",
    "mad_ctpsa_newd",
    "mad_ctpsa_add",
    "mad_ctpsa_cplx",
    "mad_ctpsa_compose",
)

_lib = None
_lib_path: str | None = None
_last_error: str | None = None


def _split_env_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(os.pathsep) if part]


def _platform_library_names() -> list[str]:
    if sys.platform.startswith("win"):
        return ["madng_tpsa.dll", "libmadng_tpsa.dll", "madng.dll", "libmadng.dll"]
    if sys.platform == "darwin":
        return ["libmadng_tpsa.dylib", "madng_tpsa.dylib", "libmadng.dylib", "madng.dylib"]
    return ["libmadng_tpsa.so", "madng_tpsa.so", "libmadng.so", "madng.so"]


def _pymadng_candidate_paths() -> Iterable[str]:
    """Yield future shared-object candidates from an installed pymadng package.

    Current pymadng wheels mainly ship MAD-NG executables for subprocess use.
    They are not relied on by default now that madng-tpsa has a vendored C core,
    but keeping this probe makes an explicit external fallback future-proof if
    pymadng later ships dlopen-able libraries.
    """

    try:
        import pymadng  # type: ignore[import-not-found]
    except Exception:
        return

    package_dir = Path(getattr(pymadng, "__file__", "")).resolve().parent
    if not package_dir.exists():
        return

    system = platform.system()
    binary_names: list[str]
    if system == "Linux":
        binary_names = ["mad_Linux"]
    elif system == "Darwin":
        binary_names = ["mad_Darwin"]
    elif system == "Windows":
        binary_names = ["mad_Windows.exe", "mad_Windows"]
    else:
        binary_names = []

    names = [*_platform_library_names(), *binary_names]
    seen: set[Path] = set()
    for base in (package_dir / "bin", package_dir):
        for name in names:
            candidate = base / name
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                yield str(candidate)

    bin_dir = package_dir / "bin"
    if bin_dir.exists():
        for pattern in ("*.so", "*.dylib", "*.dll"):
            for candidate in sorted(bin_dir.glob(pattern)):
                if candidate not in seen and candidate.is_file():
                    seen.add(candidate)
                    yield str(candidate)


def _library_candidates(explicit: str | os.PathLike[str] | None = None) -> Iterable[str]:
    if explicit is not None:
        yield os.fspath(explicit)
        return

    for value in _split_env_paths(os.environ.get("MADNG_TPSA_LIBRARY")):
        yield value
    for value in _split_env_paths(os.environ.get("MADNG_LIBRARY")):
        yield value

    seen: set[str] = set()
    for value in _pymadng_candidate_paths():
        if value not in seen:
            seen.add(value)
            yield value

    for name in _platform_library_names():
        if name not in seen:
            seen.add(name)
            yield name
        stem = name[3:] if name.startswith("lib") else name
        stem = stem.split(".", 1)[0]
        found = ctypes.util.find_library(stem)
        if found and found not in seen:
            seen.add(found)
            yield found


def _dlopen_flags() -> int:
    return int(getattr(os, "RTLD_NOW", 0)) | int(getattr(os, "RTLD_GLOBAL", 0))


def _validate_symbols(lib: object, candidate: str) -> None:
    missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(lib, name)]
    if missing:
        joined = ", ".join(missing)
        raise MadngLibraryError(f"{candidate!r} does not export required MAD-NG TPSA symbols: {joined}")


def _load_external(path: str | os.PathLike[str] | None = None):
    errors: list[str] = []
    for candidate in _library_candidates(path):
        expanded = os.path.expanduser(os.path.expandvars(candidate))
        try:
            lib = ffi.dlopen(expanded, _dlopen_flags())
            _validate_symbols(lib, expanded)
        except Exception as exc:  # pragma: no cover - platform-specific dlopen details
            errors.append(f"{expanded}: {exc}")
            continue
        return lib, expanded, None
    return None, None, errors


def load_library(path: str | os.PathLike[str] | None = None):
    """Load and return the TPSA C library.

    With no arguments and no ``MADNG_TPSA_LIBRARY`` / ``MADNG_LIBRARY`` override,
    this returns the vendored CFFI extension included in the package.  Passing a
    path, or setting one of those environment variables, makes the loader try an
    external MAD-NG-compatible shared library first and then fall back to the
    vendored implementation.
    """

    global _lib, _lib_path, _last_error
    if _lib is not None and path is None:
        return _lib

    external_requested = path is not None or os.environ.get("MADNG_TPSA_LIBRARY") or os.environ.get("MADNG_LIBRARY")
    external_errors: list[str] | None = None
    if external_requested:
        lib, lib_path, external_errors = _load_external(path)
        if lib is not None:
            _lib, _lib_path, _last_error = lib, str(lib_path), None
            return lib

    if _vendored_cffi is not None:
        try:
            _validate_symbols(_vendored_cffi.lib, "vendored madng_tpsa C core")
        except Exception as exc:
            _last_error = f"Vendored madng_tpsa C core is present but invalid: {exc}"
            raise MadngLibraryError(_last_error) from exc
        _lib = _vendored_cffi.lib
        _lib_path = "vendored:madng_tpsa._madng_tpsa_cffi"
        if external_requested and external_errors:
            _last_error = "External library load failed; using vendored C core. " + "; ".join(external_errors)
        else:
            _last_error = None
        return _lib

    lib, lib_path, external_errors = _load_external(path)
    if lib is not None:
        _lib, _lib_path, _last_error = lib, str(lib_path), None
        return lib

    detail = "; ".join(external_errors or []) if external_errors else "no external candidates were loadable"
    vendored = f" vendored extension import failed: {_vendored_import_error!r}." if _vendored_import_error else ""
    _last_error = (
        "Could not load a MAD-NG-compatible TPSA library: "
        f"{detail}.{vendored} Reinstall from source or set MADNG_TPSA_LIBRARY "
        "to a shared library exporting the MAD-NG TPSA symbols."
    )
    raise MadngLibraryError(_last_error)


def current_library():
    return _lib


def loaded_library_path() -> str | None:
    return _lib_path


def is_available(path: str | os.PathLike[str] | None = None) -> bool:
    try:
        load_library(path)
    except MadngLibraryError:
        return False
    return True


def availability_error() -> str | None:
    return _last_error


def _ord(value: int) -> int:
    if value < 0 or value > 255:
        raise ValueError(f"MAD-NG order must fit ord_t, got {value}")
    return int(value)


def default_order() -> int:
    return int(load_library().mad_tpsa_dflt)


def same_order() -> int:
    return int(load_library().mad_tpsa_same)


def ptr_address(ptr) -> int:
    return int(ffi.cast("uintptr_t", ptr))


def path_exists(path: str | os.PathLike[str]) -> bool:
    return Path(path).exists()
