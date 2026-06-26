"""Lazy CFFI loader for MAD-NG's TPSA C API."""
from __future__ import annotations

import ctypes.util
import os
from pathlib import Path
from threading import RLock
from typing import Iterable

from cffi import FFI

from ._cdefs import CDEF


class LibraryNotFoundError(RuntimeError):
    """Raised when no loadable MAD-NG library can be found."""


_lock = RLock()
_ffi: FFI | None = None
_lib = None
_loaded_from: str | None = None
_library_override: str | None = None
_load_errors: list[str] = []


_LIBRARY_ENV_VARS = ("MADNG_TPSA_LIBRARY", "MADNG_LIBRARY")
_LIBRARY_DIR_ENV_VAR = "MADNG_TPSA_LIBRARY_DIR"
_LIBRARY_NAMES = ("mad", "libmad.so", "libmad.dylib", "mad.dll")


def set_library_path(path: str | os.PathLike[str] | None) -> None:
    """Set the MAD-NG shared library path used by future loads.

    Call this before constructing descriptors or TPSA objects.  Passing ``None``
    clears the explicit override.  Once the library is loaded, cffi keeps it loaded
    for the Python process; changing the path afterward raises ``RuntimeError``.
    """

    global _library_override
    with _lock:
        if _lib is not None:
            raise RuntimeError("MAD-NG library is already loaded; set the path before first use")
        _library_override = None if path is None else os.fspath(path)


def is_loaded() -> bool:
    """Return ``True`` if the MAD-NG library has been loaded in this process."""

    return _lib is not None


def library_path() -> str | None:
    """Return the path or loader name used for the loaded library, if any."""

    return _loaded_from


def ffi() -> FFI:
    """Return the singleton cffi ``FFI`` object without loading the library."""

    global _ffi
    with _lock:
        if _ffi is None:
            f = FFI()
            f.cdef(CDEF)
            _ffi = f
        return _ffi


def lib():
    """Return the loaded MAD-NG library object, loading it on first use."""

    global _lib, _loaded_from
    with _lock:
        if _lib is not None:
            return _lib

        f = ffi()
        errors: list[str] = []
        for candidate in _candidate_libraries():
            try:
                candidate_lib = f.dlopen(candidate)
                _validate_library(candidate_lib)
                _lib = candidate_lib
                _loaded_from = candidate
                _load_errors.clear()
                return _lib
            except (OSError, AttributeError) as exc:
                errors.append(f"{candidate!r}: {exc}")

        _load_errors[:] = errors
        tried = "\n  - ".join(errors) if errors else "no candidates generated"
        raise LibraryNotFoundError(
            "Could not load MAD-NG TPSA library. Set MADNG_TPSA_LIBRARY to a "
            "shared library exposing mad_tpsa_* and mad_desc_* symbols. Tried:\n  - "
            f"{tried}"
        )


def load_library(path: str | os.PathLike[str] | None = None):
    """Load the MAD-NG library now and return the cffi library object."""

    if path is not None:
        set_library_path(path)
    return lib()


def last_load_errors() -> tuple[str, ...]:
    """Return loader errors captured during the last failed load attempt."""

    return tuple(_load_errors)


def _candidate_libraries() -> Iterable[str]:
    seen: set[str] = set()

    def add(candidate: str | os.PathLike[str] | None):
        if not candidate:
            return
        text = os.fspath(candidate)
        if text and text not in seen:
            seen.add(text)
            yield text

    yield from add(_library_override)

    for env in _LIBRARY_ENV_VARS:
        yield from add(os.environ.get(env))

    lib_dir = os.environ.get(_LIBRARY_DIR_ENV_VAR)
    if lib_dir:
        base = Path(lib_dir)
        for name in _LIBRARY_NAMES:
            yield from add(base / name)

    found = ctypes.util.find_library("mad")
    yield from add(found)

    for name in _LIBRARY_NAMES:
        yield from add(name)


# C ord_t is unsigned char.  MAD-NG stores the sentinels -1 and -2 in ord_t, so
# they appear as 255 and 254 from Python.  We prefer querying globals when the
# library is loaded, but these values are useful before dlopen.
ORD_DFLT_FALLBACK = 255
ORD_SAME_FALLBACK = 254


def ord_dflt() -> int:
    try:
        return int(lib().mad_tpsa_dflt)
    except Exception:
        return ORD_DFLT_FALLBACK


def ord_same() -> int:
    try:
        return int(lib().mad_tpsa_same)
    except Exception:
        return ORD_SAME_FALLBACK


def _validate_library(candidate_lib) -> None:
    """Force resolution of a minimal MAD-NG TPSA symbol set.

    Some systems ship unrelated libraries named ``libmad``.  CFFI resolves
    functions lazily, so ``dlopen('mad')`` can succeed even when the library is
    not MAD-NG.  Resolving these symbols here prevents false-positive loads.
    """

    required = (
        "mad_desc_newv",
        "mad_desc_getnv",
        "mad_tpsa_newd",
        "mad_tpsa_setvar",
        "mad_tpsa_add",
    )
    missing: list[str] = []
    for name in required:
        try:
            getattr(candidate_lib, name)
        except AttributeError:
            missing.append(name)
    if missing:
        raise AttributeError("not a MAD-NG TPSA library; missing " + ", ".join(missing))
