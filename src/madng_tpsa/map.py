"""TPSA map wrapper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Callable

from ._cffi import ffi, load_library
from .descriptor import Descriptor
from .tpsa import TPSA, _is_number


class TPSAMap(Sequence[TPSA]):
    """A vector/map of TPSA components.

    ``@`` composes maps: ``outer @ inner`` means ``outer.compose(inner)``.
    Ordinary arithmetic operators are component-wise.
    """

    def __init__(self, values: Iterable[TPSA]) -> None:
        components = tuple(values)
        if not components:
            raise ValueError("TPSAMap requires at least one TPSA component")
        descriptor = components[0].descriptor
        for component in components:
            component._assert_open()
            if component.descriptor.address != descriptor.address:
                raise ValueError("all TPSAMap components must use the same descriptor")
        self._values = components
        self._descriptor = descriptor

    @classmethod
    def identity(cls, descriptor: Descriptor, *, size: int | None = None, order: int | None = None) -> "TPSAMap":
        """Return an identity map ``[x1, x2, ...]``."""

        if size is None:
            size = descriptor.n_variables
        if not (1 <= size <= descriptor.n_variables):
            raise ValueError(f"identity size must be in 1..{descriptor.n_variables}, got {size}")
        return cls(TPSA.variable(descriptor, i + 1, order=order) for i in range(size))

    @classmethod
    def constants(cls, descriptor: Descriptor, values: Sequence[float], *, order: int | None = None) -> "TPSAMap":
        return cls(TPSA.constant(descriptor, value, order=order) for value in values)

    @property
    def descriptor(self) -> Descriptor:
        return self._descriptor

    @property
    def size(self) -> int:
        return len(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[TPSA]:
        return iter(self._values)

    def __getitem__(self, item):
        return self._values[item]

    def copy(self) -> "TPSAMap":
        return TPSAMap(component.copy() for component in self)

    def apply(self, func: Callable[[TPSA], TPSA]) -> "TPSAMap":
        return TPSAMap(func(component) for component in self)

    def compose(self, inner: "TPSAMap") -> "TPSAMap":
        """Compose this map with ``inner`` using ``mad_tpsa_compose``."""

        if not isinstance(inner, TPSAMap):
            inner = TPSAMap(inner)  # type: ignore[arg-type]
        self._assert_same_descriptor(inner)
        lib = load_library()
        outputs = [component._new_like() for component in self]
        ma = ffi.new("tpsa_t*[]", [component.ptr for component in self])
        mb = ffi.new("tpsa_t*[]", [component.ptr for component in inner])
        mc = ffi.new("tpsa_t*[]", [component.ptr for component in outputs])
        lib.mad_tpsa_compose(len(self), ma, len(inner), mb, mc)
        return TPSAMap(outputs)

    def inverse(self, *, size: int | None = None, iterations: int | None = None) -> "TPSAMap":
        """Return the formal inverse map using ``mad_tpsa_minv``.

        The bundled C backend links with LAPACK and uses its linear solve in
        ``mad_tpsa_minv`` before the TPSA Newton refinement.  ``iterations`` is
        accepted for compatibility with earlier releases; the C backend controls
        its own convergence loop.
        """

        if iterations is not None:
            # Kept intentionally as a no-op compatibility parameter.
            int(iterations)
        if size is None:
            size = len(self)
        size = int(size)
        if size <= 0:
            raise ValueError("inverse size must be positive")
        if size > len(self):
            raise ValueError("inverse size cannot exceed the number of map components")
        if size > self.descriptor.n_variables:
            raise ValueError("inverse size cannot exceed the descriptor variable count")

        lib = load_library()
        outputs = [component._new_like() for component in self[:size]]
        ma = ffi.new("tpsa_t*[]", [component.ptr for component in self])
        mc = ffi.new("tpsa_t*[]", [component.ptr for component in outputs])
        lib.mad_tpsa_minv(len(self), ma, size, mc)
        return TPSAMap(outputs)

    def translate(self, values: Sequence[float]) -> "TPSAMap":
        """Translate/evaluate this map around a numeric vector."""

        lib = load_library()
        tb = ffi.new("num_t[]", [float(v) for v in values])
        outputs = [component._new_like() for component in self]
        ma = ffi.new("tpsa_t*[]", [component.ptr for component in self])
        mc = ffi.new("tpsa_t*[]", [component.ptr for component in outputs])
        lib.mad_tpsa_translate(len(self), ma, len(values), tb, mc)
        return TPSAMap(outputs)

    def evaluate(self, values: Sequence[float]) -> tuple[float, ...]:
        """Numerically evaluate this TPSA map."""

        lib = load_library()
        tb = ffi.new("num_t[]", [float(v) for v in values])
        tc = ffi.new("num_t[]", len(self))
        ma = ffi.new("tpsa_t*[]", [component.ptr for component in self])
        lib.mad_tpsa_eval(len(self), ma, len(values), tb, tc)
        return tuple(float(tc[i]) for i in range(len(self)))

    def norm(self) -> float:
        lib = load_library()
        ma = ffi.new("tpsa_t*[]", [component.ptr for component in self])
        return float(lib.mad_tpsa_mnrm(len(self), ma))

    def max_order(self, *, high_order: bool = False) -> int:
        lib = load_library()
        ma = ffi.new("tpsa_t*[]", [component.ptr for component in self])
        return int(lib.mad_tpsa_mord(len(self), ma, bool(high_order)))

    def _assert_same_descriptor(self, other: "TPSAMap") -> None:
        if self.descriptor.address != other.descriptor.address:
            raise ValueError("TPSAMap descriptors differ")

    def _binary(self, other, op: str, *, reflected: bool = False):
        if isinstance(other, TPSAMap):
            self._assert_same_descriptor(other)
            if len(self) != len(other):
                raise ValueError("TPSAMap sizes differ")
            if reflected:
                return TPSAMap(getattr(b, op)(a) for a, b in zip(self, other))
            return TPSAMap(getattr(a, op)(b) for a, b in zip(self, other))
        if isinstance(other, TPSA) or _is_number(other):
            if reflected:
                return TPSAMap(getattr(component, _reflected_name(op))(other) for component in self)
            return TPSAMap(getattr(component, op)(other) for component in self)
        return NotImplemented

    def __matmul__(self, inner: "TPSAMap") -> "TPSAMap":
        return self.compose(inner)

    def __add__(self, other):
        return self._binary(other, "__add__")

    def __radd__(self, other):
        return self._binary(other, "__add__", reflected=True)

    def __sub__(self, other):
        return self._binary(other, "__sub__")

    def __rsub__(self, other):
        return self._binary(other, "__sub__", reflected=True)

    def __mul__(self, other):
        return self._binary(other, "__mul__")

    def __rmul__(self, other):
        return self._binary(other, "__mul__", reflected=True)

    def __truediv__(self, other):
        return self._binary(other, "__truediv__")

    def __rtruediv__(self, other):
        return self._binary(other, "__truediv__", reflected=True)

    def __pow__(self, other):
        return self._binary(other, "__pow__")

    def __neg__(self):
        return TPSAMap(-component for component in self)

    def __pos__(self):
        return self.copy()

    def __repr__(self) -> str:
        return f"TPSAMap(size={len(self)}, descriptor=0x{self.descriptor.address:x})"


def _reflected_name(name: str) -> str:
    return {
        "__add__": "__radd__",
        "__sub__": "__rsub__",
        "__mul__": "__rmul__",
        "__truediv__": "__rtruediv__",
        "__pow__": "__rpow__",
    }[name]
