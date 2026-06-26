"""Exceptions raised by :mod:`madng_tpsa`."""

from __future__ import annotations


class MadngTPSAError(RuntimeError):
    """Base class for package-level runtime errors."""


class MadngLibraryError(ImportError, MadngTPSAError):
    """Raised when a MAD-NG TPSA shared library cannot be loaded."""


class DescriptorClosedError(MadngTPSAError):
    """Raised when an operation is attempted on a closed descriptor."""


class TPSAClosedError(MadngTPSAError):
    """Raised when an operation is attempted on a freed TPSA value."""
