"""Coupled 6D parametric normal form, including a nonlinear RF cavity.

Run ``python examples/parametric_normal_form_6d.py`` after installing
``madng-tpsa`` and NumPy. See ``examples/parametric_normal_form_6d.md`` for
conventions, equations, coefficient access, and the limits of this example.

Coordinates are dimensionless canonical pairs (x, px, y, py, z, delta).
The example calculates the closed orbit, a symplectic linear A(p), a canonical
nonlinear H(p), and all nine leading action-detuning coefficients. The phase
map is cubic; parameter order is independently selectable (default: one).

Parameter Taylor coefficients are *native six-variable CTPSA objects*, not
pairs of real series. A small outer parameter convolution keeps the two
truncation orders separate and avoids large dense combined descriptors during
normalization. Result.native() exports ordinary parametric CTPSA/CTPSAMap
objects. No finite differences are used to calculate parameter coefficients.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import permutations
from math import factorial, pi
from typing import Mapping, Sequence

import numpy as np

import madng_tpsa as mt

ND = 6
PHASE_ORDER = 3
ZERO = (0,) * ND
UNITS = tuple(tuple(int(i == j) for i in range(ND)) for j in range(ND))
J = np.kron(np.eye(3), [[0.0, 1.0], [-1.0, 0.0]])
DEFAULTS = dict(kf=0.86, kd=-0.74, ks=0.06, ko=0.025,
                kxy=0.035, vrf=0.18, kb=0.0, phirf=0.0)
PARAMETERS = tuple(DEFAULTS)


class ParameterRing:
    """Hold the native phase descriptor and independent parameter truncation."""

    def __init__(self, names: Sequence[str], order: int):
        """Create a ring with cubic phase order and parameter order 0, 1, or 2."""
        if order not in (0, 1, 2):
            raise ValueError("parameter order must be 0, 1, or 2")
        if len(set(names)) != len(names) or set(names) - set(PARAMETERS):
            raise ValueError(f"parameters must be distinct names from {PARAMETERS}")
        self.names = tuple(names)
        self.order = order
        self.zero = (0,) * len(names)
        self.desc = mt.descriptor(ND, PHASE_ORDER)

    def constant(self, value: complex = 0.0) -> Jet:
        """Return a constant with the full working phase and parameter orders."""
        return Jet(self, {self.zero: mt.CTPSA.constant(self.desc, complex(value))})

    def coordinate(self, index: int) -> Jet:
        """Return a zero-based phase coordinate as a native complex series."""
        return Jet(self, {self.zero: mt.CTPSA.variable(self.desc, index + 1)})

    def knob(self, name: str, nominal: float) -> Jet:
        """Return nominal plus its independent parameter increment, when active."""
        out = self.constant(nominal)
        if name in self.names and self.order:
            exponent = tuple(int(i == self.names.index(name)) for i in range(len(self.names)))
            out.data[exponent] = mt.CTPSA.constant(self.desc, 1.0)
        return out


class Jet:
    """A parameter polynomial whose coefficients are native complex 6D TPSAs.

    This is an example-local tensor-product container, not a replacement for
    the library's TPSA algebra. All phase multiplication, differentiation,
    elementary functions, and coefficient access use madng_tpsa.CTPSA.
    """

    def __init__(self, ring: ParameterRing, data: Mapping[tuple[int, ...], mt.CTPSA]):
        """Store nonzero parameter coefficients, without mutating input series."""
        self.ring = ring
        self.data = {a: f for a, f in data.items() if not f.is_null()}

    def _coerce(self, other):
        if isinstance(other, Jet):
            if other.ring is not self.ring:
                raise ValueError("parameter rings differ")
            return other
        return self.ring.constant(complex(other))

    def __add__(self, other):
        """Add coefficients with equal parameter exponents."""
        other = self._coerce(other)
        data = dict(self.data)
        for a, f in other.data.items():
            data[a] = data[a] + f if a in data else f
        return Jet(self.ring, data)

    __radd__ = __add__

    def __neg__(self):
        """Return the additive inverse."""
        return Jet(self.ring, {a: -f for a, f in self.data.items()})

    def __sub__(self, other):
        """Subtract another parameter jet or scalar."""
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        """Subtract this jet from a scalar."""
        return self._coerce(other) - self

    def __mul__(self, other):
        """Convolve parameter exponents, multiplying phase series in C."""
        if not isinstance(other, Jet):
            return Jet(self.ring, {a: f * complex(other) for a, f in self.data.items()})
        other = self._coerce(other)
        data = {}
        for a, f in self.data.items():
            for b, g in other.data.items():
                ab = tuple(x + y for x, y in zip(a, b))
                if sum(ab) <= self.ring.order:
                    product = f * g
                    data[ab] = data[ab] + product if ab in data else product
        return Jet(self.ring, data)

    __rmul__ = __mul__

    def __pow__(self, power: int):
        """Raise a jet to an integer power, including negative powers of units."""
        if not isinstance(power, int):
            raise TypeError("Jet powers must be integers")
        if power < 0:
            return self.reciprocal() ** (-power)
        out, base = self.ring.constant(1), self
        while power:
            if power & 1:
                out = out * base
            power >>= 1
            if power:
                base = base * base
        return out

    def reciprocal(self):
        """Invert a jet with nonzero nominal constant term by a finite series."""
        a0 = self.data.get(self.ring.zero)
        if a0 is None or a0.constant_term == 0:
            raise ZeroDivisionError("jet has zero nominal constant term")
        inv = Jet(self.ring, {self.ring.zero: 1.0 / a0})
        epsilon = (self - Jet(self.ring, {self.ring.zero: a0})) * inv
        return inv * sum(((-epsilon) ** k for k in range(self.ring.order + 1)),
                         self.ring.constant())

    def __truediv__(self, other):
        """Divide by a scalar or an invertible parameter jet."""
        if not isinstance(other, Jet):
            if other == 0:
                raise ZeroDivisionError("division by zero")
            return self * (1.0 / complex(other))
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other):
        """Divide a scalar by this jet."""
        return self.reciprocal() * other

    def analytic(self, name: str):
        """Evaluate sin, sqrt, or log, expanding only the nilpotent parameters."""
        a0 = self.data.get(self.ring.zero, mt.CTPSA.constant(self.ring.desc))
        base = Jet(self.ring, {self.ring.zero: a0})
        epsilon = self - base
        if name == "sin":
            derivatives = (mt.sin(a0), mt.cos(a0), -mt.sin(a0))
            return sum((Jet(self.ring, {self.ring.zero: derivatives[k]}) * epsilon**k / factorial(k)
                        for k in range(self.ring.order + 1)), self.ring.constant())
        if a0.constant_term == 0:
            raise ValueError(f"{name} requires a nonzero nominal constant term")
        h = epsilon / base
        if name == "sqrt":
            return Jet(self.ring, {self.ring.zero: mt.sqrt(a0)}) * (1 + h / 2 - h*h / 8)
        if name == "log":
            return Jet(self.ring, {self.ring.zero: mt.log(a0)}) + h - h*h / 2
        raise ValueError(f"unsupported analytic function {name}")

    def real(self):
        """Project coefficient values onto their real parts (variables unchanged)."""
        return Jet(self.ring, {a: mt.CTPSA.from_tpsa(f.real) for a, f in self.data.items()})

    def imag(self):
        """Project coefficient values onto their imaginary parts."""
        return Jet(self.ring, {a: mt.CTPSA.from_tpsa(f.imag) for a, f in self.data.items()})

    def conjugate(self):
        """Conjugate coefficients, without exchanging formal z and zbar variables."""
        return Jet(self.ring, {a: f.conjugate() for a, f in self.data.items()})

    def degree(self, n: int):
        """Select homogeneous *phase* degree n, retaining parameter coefficients."""
        return Jet(self.ring, {a: f.get_order(n) for a, f in self.data.items()})

    def derivative(self, index: int):
        """Differentiate a zero-based phase coordinate in each native coefficient."""
        return Jet(self.ring, {a: f.derivative(index + 1) for a, f in self.data.items()})

    def coefficient(self, monomial=ZERO):
        """Extract a phase monomial as a parameter-only jet."""
        return Jet(self.ring, {a: mt.CTPSA.constant(self.ring.desc, f[monomial])
                               for a, f in self.data.items()})

    def nominal(self) -> complex:
        """Return the constant coefficient in both phase and parameter variables."""
        f = self.data.get(self.ring.zero)
        return f.constant_term if f is not None else 0j

    def norm(self) -> float:
        """Return the largest absolute Taylor coefficient, including parameters."""
        return max((abs(c) for f in self.data.values() for _, c in f.coefficients()), default=0.0)

    def native(self, descriptor: mt.Descriptor) -> mt.CTPSA:
        """Flatten the two orders into a library CTPSA with explicit parameters."""
        if descriptor.n_variables != ND or descriptor.n_parameters != len(self.ring.names):
            raise ValueError("export descriptor has incompatible dimensions")
        out = mt.CTPSA.constant(descriptor)
        for parameter, f in self.data.items():
            for monomial, coefficient in f.coefficients():
                out[monomial + parameter] = coefficient
        return out


def matvec(matrix, vector):
    """Multiply a small scalar/jet matrix by a jet vector."""
    return [sum((entry * value for entry, value in zip(row, vector)), 0)
            for row in matrix]


def matmul(a, b):
    """Multiply matrices whose entries may be parameter jets."""
    return [list(row) for row in zip(*(matvec(a, column) for column in zip(*b)))]


def transpose(a):
    """Return a list-based transpose without NumPy object-array coercion."""
    return [list(column) for column in zip(*a)]


def matrix_nominal(a):
    """Extract a dense complex nominal matrix for numerical linear algebra."""
    return np.array([[entry.nominal() for entry in row] for row in a], dtype=complex)


def maximum(values):
    """Return the largest Taylor-coefficient norm in a vector."""
    return max((x.norm() for x in values), default=0.0)


def track(coordinates, knobs):
    """Track the exactly symplectic thin-lens toy ring, then truncate its series.

    Each drift is the flow of
    T=(px^2+py^2+eta*delta^2)/2 + dispersion*px*delta.
    The companion z += L*dispersion*px term is essential for symplecticity.
    RF is delta -= vrf*sin(k_rf*z+phirf), including its cubic nonlinearity.
    These are scaled illustrative coordinates, not a calibrated machine model.
    """
    x, px, y, py, z, delta = coordinates
    eta, dispersion, k_rf = 0.42, 0.13, 1.1

    def drift(length, state):
        x, px, y, py, z, delta = state
        return [x + length*(px + dispersion*delta), px,
                y + length*py, py, z + length*(eta*delta + dispersion*px), delta]

    x, px, y, py, z, delta = drift(0.5, [x, px, y, py, z, delta])
    px, py = px - knobs['kf']*x, py + knobs['kf']*y
    # Normal octupole and skew quadrupole colocated at QF.
    px, py = (px - knobs['ko']*(x**3 - 3*x*y*y)/6 - knobs['kxy']*y,
              py - knobs['ko']*(y**3 - 3*x*x*y)/6 - knobs['kxy']*x)
    x, px, y, py, z, delta = drift(1.0, [x, px, y, py, z, delta])
    px, py = (px - knobs['kd']*x - knobs['ks']*(x*x-y*y)/2 + knobs['kb'],
              py + knobs['kd']*y + knobs['ks']*x*y)
    x, px, y, py, z, delta = drift(0.5, [x, px, y, py, z, delta])
    phase = k_rf*z + knobs['phirf']
    sine = phase.analytic('sin') if isinstance(phase, Jet) else np.sin(phase)
    delta = delta - knobs['vrf']*sine
    return [x, px, y, py, z, delta]


def closed_orbit(ring, knobs):
    """Solve the nominal and parameter-dependent fixed point with a frozen Jacobian."""
    zero = [ring.constant() for _ in range(ND)]
    variables = [ring.coordinate(i) for i in range(ND)]
    origin_map = track(variables, knobs)
    r0 = matrix_nominal([[f.coefficient(m) for m in UNITS] for f in origin_map])
    if np.linalg.cond(r0 - np.eye(ND)) > 1e10:
        raise ValueError("closed-orbit equation is singular or ill-conditioned")
    inverse = np.linalg.inv(r0 - np.eye(ND))
    orbit = zero
    for _ in range(ring.order + 3):
        residual = [a-b for a, b in zip(track(orbit, knobs), orbit)]
        if maximum(residual) < 2e-13:
            break
        correction = matvec(inverse, residual)
        orbit = [a-b for a, b in zip(orbit, correction)]
    error = maximum([a-b for a, b in zip(track(orbit, knobs), orbit)])
    if error > 2e-10:
        raise RuntimeError(f"parameter closed orbit did not converge: {error:g}")
    return orbit, error


def linear_normalizer(matrix, ring):
    """Solve parameter eigenvectors and symplectically normalize three stable modes.

    A bordered solve fixes l0*v(p)=1, where l0 is a nominal left eigenvector.
    Parameter coefficients are obtained algebraically, not by finite differences.
    The returned real A obeys A.T*J*A=J and A^-1*M*A=diag(R(mu_j)).
    """
    r0 = matrix_nominal(matrix)
    if np.max(np.abs(r0.imag)) > 1e-10 or np.max(np.abs(r0.T @ J @ r0 - J)) > 1e-9:
        raise ValueError("nominal linear map is not real symplectic")
    eigenvalues, eigenvectors = np.linalg.eig(r0.real)
    if np.max(np.abs(np.abs(eigenvalues)-1)) > 1e-9:
        raise ValueError(f"unstable nominal map: eigenvalues={eigenvalues}")
    separation = np.abs(eigenvalues[:, None] - eigenvalues[None, :]) + np.eye(ND)*10
    if separation.min() < 1e-7:
        raise ValueError("degenerate modes or integer/half-integer tune")
    left = np.linalg.inv(eigenvectors)
    selected = [i for i in range(ND)
                if float(eigenvectors[:, i].real @ J @ eigenvectors[:, i].imag) > 1e-10]
    if len(selected) != 3:
        raise ValueError("could not identify three positive-signature stable modes")
    # Keep the nominal x-like, y-like, synchrotron-like ordering continuous in p.
    selected = max(permutations(selected), key=lambda inds: sum(
        np.linalg.norm(eigenvectors[2*j:2*j+2, inds[j]])**2
        / np.linalg.norm(eigenvectors[:, inds[j]])**2 for j in range(3)))
    columns, lambdas = [], []
    for index in selected:
        v0, lam0 = eigenvectors[:, index], eigenvalues[index]
        bordered = np.zeros((ND+1, ND+1), dtype=complex)
        bordered[:ND, :ND] = r0 - lam0*np.eye(ND)
        bordered[:ND, ND] = -v0
        bordered[ND, :ND] = left[index]
        if np.linalg.cond(bordered) > 1e10:
            raise ValueError("ill-conditioned eigenvector continuation")
        inverse = np.linalg.inv(bordered)
        v, lam = [ring.constant(x) for x in v0], ring.constant(lam0)
        for _ in range(ring.order + 1):
            residual = [a-lam*b for a, b in zip(matvec(matrix, v), v)]
            gauge = sum((left[index, j]*v[j] for j in range(ND)), ring.constant()) - 1
            correction = matvec(inverse, [-a for a in residual] + [-gauge])
            v = [a+b for a, b in zip(v, correction[:ND])]
            lam = lam + correction[ND]
        re, im = [a.real() for a in v], [a.imag() for a in v]
        area = sum((a*b for a, b in zip(re, matvec(J, im))), ring.constant())
        scale = area.analytic('sqrt')
        columns.extend([[a/scale for a in re], [a/scale for a in im]])
        lambdas.extend([lam.conjugate(), lam])  # z=(Q+iP)/sqrt(2)
    a = transpose(columns)
    ainv = matmul(matmul(-J, transpose(a)), J)
    defect = matmul(matmul(transpose(a), J), a)
    error = maximum([defect[i][j]-J[i, j] for i in range(ND) for j in range(ND)])
    if error > 2e-9:
        raise RuntimeError(f"parameter linear normalizer is not symplectic: {error:g}")
    return a, ainv, lambdas, error


def phase_monomials(degree):
    """Yield all exponent tuples of a fixed six-variable phase degree."""
    def recurse(prefix, remaining):
        if len(prefix) == ND-1:
            yield (*prefix, remaining)
        else:
            for n in range(remaining+1):
                yield from recurse((*prefix, n), remaining-n)
    yield from recurse((), degree)


def monomial_value(variables, exponents):
    """Form a monomial with native CTPSA products and parameter convolution."""
    out = variables[0].ring.constant(1)
    for x, power in zip(variables, exponents):
        if power:
            out = out * x**power
    return out


def rotate(vector, lambdas, variables, degree):
    """Evaluate a homogeneous map at the diagonal rotation without composition."""
    out = [variables[0].ring.constant() for _ in vector]
    for m in phase_monomials(degree):
        factor = monomial_value(lambdas, m) * monomial_value(variables, m)
        for i, f in enumerate(vector):
            out[i] = out[i] + f.coefficient(m)*factor
    return out


def directional(vector, direction):
    """Return D(vector)*direction, truncated by the native phase algebra."""
    return [sum((f.derivative(i)*direction[i] for i in range(ND)), f.ring.constant())
            for f in vector]


def homological(vector, lambdas, variables, degree, resonance_tol=1e-7):
    """Remove nonresonant monomials; retain only structural action resonances.

    For G=Lw+g_n and H=w+u_n, H^-1 G H has g_n+L*u_n-u_n(Lw).
    Thus u[i,m]=g[i,m]/(lambda**m-lambda[i]). A small nonstructural
    denominator raises an error; this example is not a resonant normal form.
    """
    ring = variables[0].ring
    u, resonant = [ring.constant() for _ in vector], [ring.constant() for _ in vector]
    minimum = float('inf')
    for m in phase_monomials(degree):
        harmonic = tuple(m[2*j]-m[2*j+1] for j in range(3))
        product = monomial_value(lambdas, m)
        monomial = monomial_value(variables, m)
        for i, g in enumerate(vector):
            target = tuple((1 if i % 2 == 0 else -1) if j == i//2 else 0 for j in range(3))
            c = g.coefficient(m)
            if harmonic == target:
                resonant[i] = resonant[i] + c*monomial
            else:
                denominator = product - lambdas[i]
                minimum = min(minimum, abs(denominator.nominal()))
                if abs(denominator.nominal()) < resonance_tol:
                    raise ValueError(f"near resonance at component {i}, monomial {m}")
                if c.data:
                    u[i] = u[i] + (c/denominator)*monomial
    return u, resonant, minimum


def symplectic_defect(h):
    """Check canonical brackets of H through degree two, including parameters.

    In the formal (z,zbar) coordinates the Poisson tensor is -i*J. A cubic
    Taylor map determines its symplecticity defect only through degree two.
    """
    derivatives = [[f.derivative(i) for i in range(ND)] for f in h]
    error = 0.0
    for i in range(ND):
        for j in range(i+1, ND):
            bracket = sum((-1j*(derivatives[i][2*k]*derivatives[j][2*k+1]
                                - derivatives[i][2*k+1]*derivatives[j][2*k])
                           for k in range(3)), h[0].ring.constant())
            defect = bracket + 1j*J[i, j]
            error = max(error, *(defect.degree(n).norm() for n in range(3)))
    return error


@dataclass
class Result:
    """Store the parametric orbit, normalizers, tunes, detuning, and residuals."""

    ring: ParameterRing
    orbit: list[Jet]
    a: list[list[Jet]]
    h: list[Jet]
    normal: list[Jet]
    physical_normalizer: list[Jet]
    tunes: list[Jet]
    detuning: list[list[Jet]]
    errors: dict[str, float]

    def native(self):
        """Return native parametric CTPSA values/maps, with phase variables first."""
        d = mt.descriptor(ND, PHASE_ORDER+self.ring.order,
                          n_parameters=len(self.ring.names),
                          parameter_order=max(1, self.ring.order))
        def convert(f):
            return f.native(d)
        return dict(descriptor=d, parameters=self.ring.names,
                    closed_orbit=mt.CTPSAMap(map(convert, self.orbit)),
                    A=[[convert(x) for x in row] for row in self.a],
                    H=mt.CTPSAMap(map(convert, self.h)),
                    N=mt.CTPSAMap(map(convert, self.normal)),
                    nonlinear_A=mt.CTPSAMap(map(convert, self.physical_normalizer)),
                    tunes=[convert(x) for x in self.tunes],
                    detuning=[[convert(x) for x in row] for row in self.detuning])


def analyze(*, parameter_order=1, parameters=PARAMETERS, nominal=None) -> Result:
    """Compute a nonresonant cubic 6D normal form with parameter Taylor coefficients.

    Active parameters are additive increments about DEFAULTS. Parameter order
    two includes mixed/square sensitivities; order zero is a numerical baseline.
    Nominal dipole kick and RF phase must be zero; their increments are supported.
    """
    settings = dict(DEFAULTS)
    if nominal is not None:
        if set(nominal) - set(DEFAULTS):
            raise ValueError("unknown nominal setting")
        settings.update(nominal)
    if settings['kb'] != 0 or settings['phirf'] != 0:
        raise ValueError("this example requires zero nominal kb and phirf; their increments are supported")
    ring = ParameterRing(parameters, parameter_order)
    knobs = {name: ring.knob(name, value) for name, value in settings.items()}
    w = [ring.coordinate(i) for i in range(ND)]
    orbit, orbit_error = closed_orbit(ring, knobs)
    centered = [a-b for a, b in zip(track([x+c for x, c in zip(w, orbit)], knobs), orbit)]
    matrix = [[f.coefficient(m) for m in UNITS] for f in centered]
    a, ainv, lambdas, a_error = linear_normalizer(matrix, ring)
    # S maps the formal (z,zbar) pairs to real (Q,P) pairs.
    s = np.kron(np.eye(3), np.array([[1, 1], [-1j, 1j]])/np.sqrt(2))
    b = matmul(a, s)
    binv = matmul(np.linalg.inv(s), ainv)
    physical = [c+x for c, x in zip(orbit, matvec(b, w))]
    g = matvec(binv, [x-c for x, c in zip(track(physical, knobs), orbit)])
    linear_error = maximum([x.degree(1)-lam*v for x, lam, v in zip(g, lambdas, w)])
    centered_error = maximum([x.degree(0) for x in g])
    if max(linear_error, centered_error) > 2e-9:
        raise RuntimeError("linear/closed-orbit normalization failed")
    g2, g3 = [x.degree(2) for x in g], [x.degree(3) for x in g]
    u2, resonance2, gap2 = homological(g2, lambdas, w, 2)
    if maximum(resonance2) > 2e-12:
        raise RuntimeError("unexpected quadratic action resonance")
    duu = directional(u2, u2)
    dg_u = directional(g2, u2)
    rotated_duu = rotate(duu, lambdas, w, 3)
    # Exact cubic formula for exp(-u2) o G o exp(u2). The half-flow
    # correction is crucial: a bare w+u2 substitution is not canonical.
    cubic = [c+d+(lam*t-rt)/2 for c, d, lam, t, rt
             in zip(g3, dg_u, lambdas, duu, rotated_duu)]
    u3, n3, gap3 = homological(cubic, lambdas, w, 3)
    h3 = [v+t/2 for v, t in zip(u3, duu)]
    h = [x+u+v for x, u, v in zip(w, u2, h3)]
    normal = [lam*x+n for lam, x, n in zip(lambdas, w, n3)]
    # Coefficient-level conjugacy G o H = H o N through cubic phase order.
    ru2 = rotate(u2, lambdas, w, 2)
    rh3 = rotate(h3, lambdas, w, 3)
    e2 = [lam*u+c-r for lam, u, c, r in zip(lambdas, u2, g2, ru2)]
    e3 = [lam*v+d+c-r-n for lam, v, d, c, r, n
          in zip(lambdas, h3, dg_u, g3, rh3, n3)]
    tunes = [(-lambdas[2*j].analytic('log').imag()/(2*pi)) for j in range(3)]
    detuning = []
    for j in range(3):
        row = []
        for k in range(3):
            m = list(UNITS[2*j])
            m[2*k] += 1
            m[2*k+1] += 1
            ratio = n3[2*j].coefficient(tuple(m))/lambdas[2*j]
            if ratio.real().norm() > 2e-8:
                raise RuntimeError("resonant coefficient contains non-Hamiltonian amplitude growth")
            row.append(-ratio.imag()/(2*pi))
        detuning.append(row)
    symmetry = maximum([detuning[i][j]-detuning[j][i] for i in range(3) for j in range(3)])
    physical_normalizer = [c+x for c, x in zip(orbit, matvec(b, h))]
    errors = dict(closed_orbit=orbit_error, centered_map=centered_error,
                  linear_normalization=linear_error, symplectic_A=a_error,
                  quadratic_conjugacy=maximum(e2), cubic_conjugacy=maximum(e3),
                  detuning_symmetry=symmetry, symplectic_H=symplectic_defect(h),
                  minimum_nonresonant_denominator=min(gap2, gap3))
    for key, value in errors.items():
        if key != 'minimum_nonresonant_denominator' and value > 2e-8:
            raise RuntimeError(f"normal-form validation failed: {key}={value:g}")
    return Result(ring, orbit, a, h, normal, physical_normalizer, tunes, detuning, errors)


def compose_native(outer: mt.CTPSAMap, inner: mt.CTPSAMap) -> mt.CTPSAMap:
    """Compose exported maps while explicitly preserving every parameter.

    The current bundled backend interprets missing substitutions as zero. Pass
    parameter identity series as well as the six phase substitutions, rather
    than silently losing parametric coefficients through ``outer @ inner``.
    """
    d = outer.descriptor
    if d.n_variables != ND or d.address != inner.descriptor.address or len(inner) != ND:
        raise ValueError("expected six substitutions with the same descriptor")
    substitutions = list(inner)
    for p in range(d.n_parameters):
        identity = mt.CTPSA.constant(d)
        monomial = [0]*d.n_total
        monomial[ND+p] = 1
        identity[tuple(monomial)] = 1
        substitutions.append(identity)
    return outer.compose(mt.CTPSAMap(substitutions))


def scalar_terms(value):
    """Format parameter coefficients of a phase-independent result."""
    terms = []
    for powers, coefficient in sorted(value.data.items(), key=lambda item: (sum(item[0]), item[0])):
        c = coefficient.constant_term
        if abs(c) < 1e-11:
            continue
        label = '*'.join(f'd{name}' + (f'^{power}' if power != 1 else '')
                         for name, power in zip(value.ring.names, powers) if power) or '1'
        terms.append(f'{c.real:+.8g} {label}')
    return ' '.join(terms) or '0'


def main(argv=None):
    """Run the example and print matrices, parameter series, and residual checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameter-order', type=int, choices=(0, 1, 2), default=1)
    parser.add_argument('--parameters', nargs='+', choices=PARAMETERS, default=list(PARAMETERS))
    args = parser.parse_args(argv)
    result = analyze(parameter_order=args.parameter_order, parameters=args.parameters)
    np.set_printoptions(precision=7, suppress=True)
    print('Coupled 6D FODO + dispersive transport + sinusoidal RF cavity')
    print('Native CTPSA backend:', mt.loaded_library_path())
    print('Phase order: 3; parameter order:', args.parameter_order)
    print('Additive parameter increments:', result.ring.names)
    print('\nParameter-dependent closed orbit (x, px, y, py, z, delta):')
    for name, value in zip(('x', 'px', 'y', 'py', 'z', 'delta'), result.orbit):
        print(f'  {name}: {scalar_terms(value)}')
    print('\nA(0), columns (Q1,P1,Q2,P2,Q3,P3):\n', matrix_nominal(result.a).real)
    for name in result.ring.names if args.parameter_order else ():
        key = tuple(int(n == name) for n in result.ring.names)
        derivative = np.array([[x.data[key].constant_term.real if key in x.data else 0
                                for x in row] for row in result.a])
        print(f'\ndA/d{name} at nominal:\n', derivative)
    print('\nMode tunes (x-like, y-like, synchrotron-like):')
    for j, tune in enumerate(result.tunes):
        print(f'  Q{j+1}: {scalar_terms(tune)}')
    print('\nD[j,k] in Q_j(J,p)=Q_j(0,p)+sum_k D[j,k](p)*J_k; J_k=z_k*zbar_k:')
    for j in range(3):
        for k in range(3):
            print(f'  D[{j+1},{k+1}]: {scalar_terms(result.detuning[j][k])}')
    print('\nNonlinear H = identity + h2 + h3, max Taylor coefficients by component:')
    for j, f in enumerate(result.h):
        print(f'  H[{j}]: quadratic={f.degree(2).norm():.6g}, cubic={f.degree(3).norm():.6g}')
    print('\nValidation (all retained parameter coefficients):')
    for name, error in result.errors.items():
        print(f'  {name}: {error:.3e}')
    print('\nUse analyze(...).native() for native CTPSA/CTPSAMap orbit, A, H, N and detuning.')
    return result


if __name__ == '__main__':
    main()
