"""TPSA map wrapper."""
from __future__ import annotations

import numbers
from collections.abc import Iterable, Iterator, Sequence

from ._lib import ffi, lib
from .descriptor import Descriptor
from .tpsa import TPSA, _ensure_compatible

_Number = numbers.Real


class TPSAMap(Sequence[TPSA]):
    """Vector/map of TPSA components.

    ``a @ b`` is map composition: substitute map ``b`` into map ``a``.  All
    components must share one MAD-NG descriptor object.
    """

    def __init__(self, components: Iterable[TPSA]) -> None:
        comps = tuple(components)
        if not comps:
            raise ValueError("TPSAMap requires at least one component")
        if not all(isinstance(c, TPSA) for c in comps):
            raise TypeError("all map components must be TPSA values")
        first = comps[0]
        for comp in comps[1:]:
            _ensure_compatible(first, comp)
        self._components = comps
        self.descriptor = first.descriptor

    @classmethod
    def identity(
        cls,
        descriptor: Descriptor,
        n: int | None = None,
        *,
        start: int = 0,
        names: Sequence[str | None] | None = None,
    ) -> "TPSAMap":
        """Return an identity map over descriptor variables."""

        return cls(descriptor.variables(n, start=start, names=names))

    @classmethod
    def zeros(cls, descriptor: Descriptor, n: int, *, mo: int | None = None) -> "TPSAMap":
        """Return a map with ``n`` zero TPSA components."""

        if n <= 0:
            raise ValueError("n must be positive")
        return cls(descriptor.zero(mo=mo) for _ in range(n))

    @classmethod
    def constants(cls, descriptor: Descriptor, values: Sequence[float], *, mo: int | None = None) -> "TPSAMap":
        """Return a constant map."""

        if not values:
            raise ValueError("values must not be empty")
        return cls(descriptor.constant(float(v), mo=mo) for v in values)

    @property
    def components(self) -> tuple[TPSA, ...]:
        return self._components

    def __len__(self) -> int:
        return len(self._components)

    def __iter__(self) -> Iterator[TPSA]:
        return iter(self._components)

    def __getitem__(self, item):
        result = self._components[item]
        if isinstance(item, slice):
            return TPSAMap(result)
        return result

    @property
    def mo(self) -> int:
        return int(lib().mad_tpsa_mord(len(self), _const_ptr_array(self._components), False))

    @property
    def hi(self) -> int:
        return int(lib().mad_tpsa_mord(len(self), _const_ptr_array(self._components), True))

    def norm(self) -> float:
        return float(lib().mad_tpsa_mnrm(len(self), _const_ptr_array(self._components)))

    def copy(self) -> "TPSAMap":
        return TPSAMap(c.copy() for c in self._components)

    def eval(self, point: Sequence[float]) -> tuple[float, ...]:
        """Evaluate the map at a numeric point."""

        if not point:
            raise ValueError("point must contain at least one coordinate")
        if len(point) > self.descriptor.nn:
            raise ValueError("point length must be <= descriptor.nv + descriptor.np")
        f = ffi()
        tb = f.new("num_t []", [float(x) for x in point])
        tc = f.new("num_t []", len(self))
        lib().mad_tpsa_eval(len(self), _const_ptr_array(self._components), len(point), tb, tc)
        return tuple(float(tc[i]) for i in range(len(self)))

    def compose(self, other: "TPSAMap") -> "TPSAMap":
        """Substitute map ``other`` into ``self`` and return the resulting map."""

        if not isinstance(other, TPSAMap):
            raise TypeError("compose expects a TPSAMap")
        _ensure_same_descriptor(self, other)
        out = self._empty_outputs(len(self), mo=max(self.mo, other.mo))
        lib().mad_tpsa_compose(
            len(self),
            _const_ptr_array(self._components),
            len(other),
            _const_ptr_array(other._components),
            _ptr_array(out),
        )
        return TPSAMap(out)

    def translate(self, offsets: Sequence[float]) -> "TPSAMap":
        """Translate variables by numeric offsets using MAD-NG's map routine."""

        if not offsets:
            raise ValueError("offsets must contain at least one coordinate")
        if len(offsets) > self.descriptor.nn:
            raise ValueError("offset length must be <= descriptor.nv + descriptor.np")
        f = ffi()
        tb = f.new("num_t []", [float(x) for x in offsets])
        out = self._empty_outputs(len(self), mo=self.mo)
        lib().mad_tpsa_translate(len(self), _const_ptr_array(self._components), len(offsets), tb, _ptr_array(out))
        return TPSAMap(out)

    def inverse(self, nb: int | None = None) -> "TPSAMap":
        """Return MAD-NG's inverse map approximation."""

        nb = len(self) if nb is None else int(nb)
        if nb <= 0:
            raise ValueError("nb must be positive")
        out = self._empty_outputs(nb, mo=self.mo)
        lib().mad_tpsa_minv(len(self), _const_ptr_array(self._components), nb, _ptr_array(out))
        return TPSAMap(out)

    def partial_inverse(self, select: Sequence[int], nb: int | None = None) -> "TPSAMap":
        """Return MAD-NG's selected partial inverse map."""

        nb = len(self) if nb is None else int(nb)
        if len(select) < nb:
            raise ValueError("select must contain at least nb entries")
        f = ffi()
        sel = f.new("idx_t []", [int(x) for x in select])
        out = self._empty_outputs(nb, mo=self.mo)
        lib().mad_tpsa_pminv(len(self), _const_ptr_array(self._components), nb, _ptr_array(out), sel)
        return TPSAMap(out)

    def liebra(self, other: "TPSAMap") -> "TPSAMap":
        """Return MAD-NG's Lie bracket of two maps."""

        other = self._require_map(other)
        out = self._empty_outputs(len(self), mo=max(self.mo, other.mo))
        lib().mad_tpsa_liebra(
            len(self), _const_ptr_array(self._components), _const_ptr_array(other._components), _ptr_array(out)
        )
        return TPSAMap(out)

    def exppb(self, other: "TPSAMap") -> "TPSAMap":
        """Return ``exp(:self:) other`` using MAD-NG's map routine."""

        other = self._require_map(other)
        out = self._empty_outputs(len(self), mo=max(self.mo, other.mo))
        lib().mad_tpsa_exppb(
            len(self), _const_ptr_array(self._components), len(other), _const_ptr_array(other._components), _ptr_array(out)
        )
        return TPSAMap(out)

    def logpb(self, other: "TPSAMap") -> "TPSAMap":
        """Return MAD-NG's logarithmic Poisson-bracket map helper."""

        other = self._require_map(other)
        out = self._empty_outputs(len(self), mo=max(self.mo, other.mo))
        lib().mad_tpsa_logpb(
            len(self), _const_ptr_array(self._components), _const_ptr_array(other._components), _ptr_array(out)
        )
        return TPSAMap(out)

    def _empty_outputs(self, n: int, *, mo: int | None = None) -> tuple[TPSA, ...]:
        order = self.descriptor.mo if mo is None else min(int(mo), self.descriptor.mo)
        return tuple(TPSA(self.descriptor, 0.0, mo=order) for _ in range(n))

    def _require_map(self, other: "TPSAMap") -> "TPSAMap":
        if not isinstance(other, TPSAMap):
            raise TypeError("operand must be a TPSAMap")
        if len(self) != len(other):
            raise ValueError("maps must have the same length for this operation")
        _ensure_same_descriptor(self, other)
        return other

    def __call__(self, *args):
        if len(args) == 1 and isinstance(args[0], TPSAMap):
            return self.compose(args[0])
        if len(args) == 1 and isinstance(args[0], Sequence) and not isinstance(args[0], TPSA):
            return self.eval(args[0])
        return self.eval(args)

    def __matmul__(self, other: "TPSAMap") -> "TPSAMap":
        return self.compose(other)

    def __pos__(self) -> "TPSAMap":
        return self.copy()

    def __neg__(self) -> "TPSAMap":
        return TPSAMap(-c for c in self._components)

    def __add__(self, other):
        if isinstance(other, TPSAMap):
            other = self._require_map(other)
            return TPSAMap(a + b for a, b in zip(self, other))
        if isinstance(other, (TPSA, _Number)):
            if isinstance(other, TPSA):
                _ensure_compatible(self[0], other)
            return TPSAMap(a + other for a in self)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, TPSAMap):
            other = self._require_map(other)
            return TPSAMap(a - b for a, b in zip(self, other))
        if isinstance(other, (TPSA, _Number)):
            if isinstance(other, TPSA):
                _ensure_compatible(self[0], other)
            return TPSAMap(a - other for a in self)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (TPSA, _Number)):
            if isinstance(other, TPSA):
                _ensure_compatible(self[0], other)
            return TPSAMap(other - a for a in self)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, TPSAMap):
            other = self._require_map(other)
            return TPSAMap(a * b for a, b in zip(self, other))
        if isinstance(other, (TPSA, _Number)):
            if isinstance(other, TPSA):
                _ensure_compatible(self[0], other)
            return TPSAMap(a * other for a in self)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, TPSAMap):
            other = self._require_map(other)
            return TPSAMap(a / b for a, b in zip(self, other))
        if isinstance(other, (TPSA, _Number)):
            if isinstance(other, TPSA):
                _ensure_compatible(self[0], other)
            return TPSAMap(a / other for a in self)
        return NotImplemented

    def __repr__(self) -> str:
        return f"TPSAMap(len={len(self)}, mo={self.mo}, descriptor={self.descriptor!r})"


def _ensure_same_descriptor(a: TPSAMap, b: TPSAMap) -> None:
    if a.descriptor is not b.descriptor and a.descriptor.ptr != b.descriptor.ptr:
        raise ValueError("TPSA maps must use the same MAD-NG descriptor object")


def _ptr_array(values: Sequence[TPSA]):
    return ffi().new("tpsa_t *[]", [v.ptr for v in values])


def _const_ptr_array(values: Sequence[TPSA]):
    return ffi().new("tpsa_t *[]", [v.ptr for v in values])
