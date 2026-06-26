"""Complex TPSA map wrapper."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Callable

from ._cffi import ffi, load_library
from .ctpsa import CTPSA, _is_scalar
from .descriptor import Descriptor
from .tpsa import TPSA


class CTPSAMap(Sequence[CTPSA]):
    """A vector/map of complex TPSA components."""

    def __init__(self, components: Iterable[CTPSA | TPSA]) -> None:
        comps = tuple(self._as_ctpsa(component) for component in components)
        if not comps:
            raise ValueError("CTPSAMap requires at least one component")
        descriptor = comps[0].descriptor
        for component in comps[1:]:
            if component.descriptor.address != descriptor.address:
                raise ValueError("CTPSAMap components must share a descriptor")
        self._components = comps
        self._descriptor = descriptor

    @staticmethod
    def _as_ctpsa(component: CTPSA | TPSA) -> CTPSA:
        if isinstance(component, CTPSA):
            return component
        if isinstance(component, TPSA):
            return CTPSA.from_tpsa(component)
        raise TypeError("CTPSAMap components must be CTPSA or TPSA values")

    @classmethod
    def identity(cls, descriptor: Descriptor, size: int | None = None) -> "CTPSAMap":
        if size is None:
            size = descriptor.n_variables
        return cls(CTPSA.variable(descriptor, i + 1) for i in range(int(size)))

    @property
    def descriptor(self) -> Descriptor:
        return self._descriptor

    def copy(self) -> "CTPSAMap":
        return CTPSAMap(component.copy() for component in self)

    def apply(self, func: Callable[[CTPSA], CTPSA]) -> "CTPSAMap":
        return CTPSAMap(func(component) for component in self)

    def compose(self, inner: "CTPSAMap | Sequence[CTPSA | TPSA]") -> "CTPSAMap":
        if not isinstance(inner, CTPSAMap):
            inner = CTPSAMap(inner)
        if self.descriptor.address != inner.descriptor.address:
            raise ValueError("cannot compose maps with different descriptors")
        lib = load_library()
        ma = ffi.new("ctpsa_t*[]", [component.ptr for component in self])
        mb = ffi.new("ctpsa_t*[]", [component.ptr for component in inner])
        outs = [CTPSA.constant(self.descriptor, 0.0, order=component.order) for component in self]
        mc = ffi.new("ctpsa_t*[]", [component.ptr for component in outs])
        lib.mad_ctpsa_compose(len(self), ma, len(inner), mb, mc)
        return CTPSAMap(outs)

    def translate(self, shifts: Sequence[complex | float]) -> "CTPSAMap":
        lib = load_library()
        ma = ffi.new("ctpsa_t*[]", [component.ptr for component in self])
        tb = ffi.new("cpx_t[]", [complex(v) for v in shifts])
        outs = [component.copy() for component in self]
        mc = ffi.new("ctpsa_t*[]", [component.ptr for component in outs])
        lib.mad_ctpsa_translate(len(self), ma, len(shifts), tb, mc)
        return CTPSAMap(outs)

    def evaluate(self, values: Sequence[complex | float]) -> tuple[complex, ...]:
        lib = load_library()
        ma = ffi.new("ctpsa_t*[]", [component.ptr for component in self])
        tb = ffi.new("cpx_t[]", [complex(v) for v in values])
        out = ffi.new("cpx_t[]", len(self))
        lib.mad_ctpsa_eval(len(self), ma, len(values), tb, out)
        return tuple(complex(out[i]) for i in range(len(self)))

    def inverse(self, size: int | None = None) -> "CTPSAMap":
        if size is None:
            size = len(self)
        lib = load_library()
        ma = ffi.new("ctpsa_t*[]", [component.ptr for component in self])
        outs = [CTPSA.constant(self.descriptor, 0.0, order=self[0].order) for _ in range(int(size))]
        mc = ffi.new("ctpsa_t*[]", [component.ptr for component in outs])
        lib.mad_ctpsa_minv(len(self), ma, int(size), mc)
        return CTPSAMap(outs)

    def norm(self) -> float:
        ma = ffi.new("ctpsa_t*[]", [component.ptr for component in self])
        return float(load_library().mad_ctpsa_mnrm(len(self), ma))

    def __len__(self) -> int:
        return len(self._components)

    def __getitem__(self, index):
        return self._components[index]

    def __iter__(self) -> Iterator[CTPSA]:
        return iter(self._components)

    def _binary(self, other, op: str, *, reflected: bool = False):
        if isinstance(other, CTPSAMap):
            if len(self) != len(other):
                raise ValueError("CTPSAMap sizes differ")
            if reflected:
                return CTPSAMap(getattr(b, op)(a) for a, b in zip(self, other))
            return CTPSAMap(getattr(a, op)(b) for a, b in zip(self, other))
        if isinstance(other, CTPSA) or isinstance(other, TPSA) or _is_scalar(other):
            if reflected:
                return CTPSAMap(getattr(component, _reflected_name(op))(other) for component in self)
            return CTPSAMap(getattr(component, op)(other) for component in self)
        return NotImplemented

    def __matmul__(self, inner: "CTPSAMap") -> "CTPSAMap":
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
        return CTPSAMap(-component for component in self)

    def __pos__(self):
        return self.copy()

    def __repr__(self) -> str:
        return f"CTPSAMap(size={len(self)}, descriptor=0x{self.descriptor.address:x})"


def _reflected_name(name: str) -> str:
    return {
        "__add__": "__radd__",
        "__sub__": "__rsub__",
        "__mul__": "__rmul__",
        "__truediv__": "__rtruediv__",
        "__pow__": "__rpow__",
    }[name]
        """Return a complex identity map."""

        """Return the descriptor shared by all complex map components."""

        """Return a component-wise copy of this complex map."""

        """Apply a function to each component and return a new complex map."""

        """Compose this complex map with an inner map."""

        """Translate/evaluate this complex map around a numeric vector."""

        """Numerically evaluate this complex TPSA map."""

        """Return the formal inverse complex map using MAD-NG CTPSA inversion."""

        """Return the MAD-NG norm of the complex map."""

        """Return the number of complex map components."""

        """Return one component or a slice of components."""

        """Iterate over complex map components."""

        """Compose complex maps using the @ operator."""

        """Return a component-wise complex map sum."""

        """Return a reflected component-wise complex map sum."""

        """Return a component-wise complex map difference."""

        """Return a reflected component-wise complex map difference."""

        """Return a component-wise complex map product or scaling."""

        """Return a reflected component-wise complex map product or scaling."""

        """Return a component-wise complex map quotient."""

        """Return a reflected component-wise complex map quotient."""

        """Raise each complex map component to the given power."""

        """Return the component-wise additive inverse."""

        """Return a copy of this complex map."""

        """Return a concise representation of this complex map."""

