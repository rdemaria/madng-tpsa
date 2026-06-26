"""Parametric normal form of a one-degree-of-freedom FODO cell.

This example is intentionally written as a small, readable normal-form driver. It
uses the real MAD-NG-compatible TPSA backend provided by ``madng_tpsa`` and adds
a tiny complex-TPSA layer in Python by storing complex series as real/imaginary
TPSA pairs. That is enough for complex Courant-Snyder coordinates and for the
homological equations used by normal-form term removal.

Model
-----
The one-turn map is a thin-lens FODO cell with five parameters:

``dkd``
    defocusing quadrupole strength error added to the nominal QD strength.
``dkf``
    focusing quadrupole strength error added to the nominal QF strength.
``ks``
    thin sextupole strength at QD.
``ko``
    thin octupole strength at QF.
``kb``
    thin dipole kick at QD.

The script demonstrates the standard workflow:

1. solve the parameter-dependent closed orbit;
2. compute the parameter-dependent linear normalizing matrix ``A``;
3. transform the map to complex normalized coordinates;
4. remove non-resonant monomials order by order;
5. print the resulting amplitude detuning coefficients, including their
   parametric dependence.

Run with:
    python examples/parametric_normal_form.py
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi, sqrt
from typing import Iterable, Sequence

import madng_tpsa as mt


# Descriptor layout: variables first, parameters second.
N_VAR = 2
PARAMETER_NAMES = ("dkd", "dkf", "ks", "ko", "kb")
DKD, DKF, KS, KO, KB = range(5)


@dataclass(frozen=True)
class CTPSA:
    """A complex TPSA represented by two real TPSA objects."""

    re: mt.TPSA
    im: mt.TPSA

    @classmethod
    def zero(cls, desc: mt.Descriptor) -> "CTPSA":
        return cls(desc.constant(0.0), desc.constant(0.0))

    @classmethod
    def real(cls, value: mt.TPSA | float, desc: mt.Descriptor | None = None) -> "CTPSA":
        if isinstance(value, mt.TPSA):
            return cls(value, value.descriptor.constant(0.0))
        if desc is None:
            raise TypeError("desc is required when promoting a Python scalar")
        return cls(desc.constant(float(value)), desc.constant(0.0))

    @classmethod
    def complex(cls, re: mt.TPSA | float, im: mt.TPSA | float, desc: mt.Descriptor | None = None) -> "CTPSA":
        if isinstance(re, mt.TPSA):
            descriptor = re.descriptor
            real = re
        else:
            if desc is None:
                raise TypeError("desc is required when promoting scalar real part")
            descriptor = desc
            real = desc.constant(float(re))
        if isinstance(im, mt.TPSA):
            if im.descriptor.address != descriptor.address:
                raise ValueError("real and imaginary TPSA descriptors differ")
            imag = im
        else:
            imag = descriptor.constant(float(im))
        return cls(real, imag)

    @property
    def descriptor(self) -> mt.Descriptor:
        return self.re.descriptor

    def conj(self) -> "CTPSA":
        return CTPSA(self.re, -self.im)

    def variable_coefficient(self, j: int, k: int) -> "CTPSA":
        """Coefficient of z^j zbar^k as a parameter-only complex TPSA."""

        return CTPSA(
            variable_coefficient(self.re, (j, k)),
            variable_coefficient(self.im, (j, k)),
        )

    def without_variable_order_below(self, order: int) -> "CTPSA":
        return CTPSA(drop_variable_orders_below(self.re, order), drop_variable_orders_below(self.im, order))

    def __add__(self, other: object) -> "CTPSA":
        other = promote_complex(other, self.descriptor)
        return CTPSA(self.re + other.re, self.im + other.im)

    def __radd__(self, other: object) -> "CTPSA":
        return self.__add__(other)

    def __sub__(self, other: object) -> "CTPSA":
        other = promote_complex(other, self.descriptor)
        return CTPSA(self.re - other.re, self.im - other.im)

    def __rsub__(self, other: object) -> "CTPSA":
        other = promote_complex(other, self.descriptor)
        return CTPSA(other.re - self.re, other.im - self.im)

    def __neg__(self) -> "CTPSA":
        return CTPSA(-self.re, -self.im)

    def __mul__(self, other: object) -> "CTPSA":
        other = promote_complex(other, self.descriptor)
        return CTPSA(self.re * other.re - self.im * other.im, self.re * other.im + self.im * other.re)

    def __rmul__(self, other: object) -> "CTPSA":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "CTPSA":
        other = promote_complex(other, self.descriptor)
        den = other.re * other.re + other.im * other.im
        return CTPSA((self.re * other.re + self.im * other.im) / den, (self.im * other.re - self.re * other.im) / den)

    def __rtruediv__(self, other: object) -> "CTPSA":
        return promote_complex(other, self.descriptor).__truediv__(self)

    def __pow__(self, power: int) -> "CTPSA":
        if not isinstance(power, int) or power < 0:
            raise TypeError("CTPSA only implements non-negative integer powers in this example")
        out = CTPSA.real(1.0, self.descriptor)
        base = self
        n = power
        while n:
            if n & 1:
                out = out * base
            n >>= 1
            if n:
                base = base * base
        return out


def promote_complex(value: object, desc: mt.Descriptor) -> CTPSA:
    if isinstance(value, CTPSA):
        if value.descriptor.address != desc.address:
            raise ValueError("complex TPSA descriptors differ")
        return value
    if isinstance(value, mt.TPSA):
        if value.descriptor.address != desc.address:
            raise ValueError("TPSA descriptors differ")
        return CTPSA.real(value)
    if isinstance(value, complex):
        return CTPSA.complex(float(value.real), float(value.imag), desc)
    if isinstance(value, (int, float)):
        return CTPSA.real(float(value), desc)
    raise TypeError(f"cannot promote {type(value).__name__} to CTPSA")


def monomial_tpsa(desc: mt.Descriptor, monomial: Sequence[int], value: float) -> mt.TPSA:
    out = desc.constant(0.0)
    out[tuple(monomial)] = float(value)
    return out


def variable_coefficient(f: mt.TPSA, variable_powers: tuple[int, int]) -> mt.TPSA:
    """Return the parameter TPSA multiplying x^j p^k in ``f``."""

    out = f.descriptor.constant(0.0)
    for monomial, value in f.coefficients():
        if monomial[:N_VAR] == variable_powers:
            out[(0, 0, *monomial[N_VAR:])] = value
    return out


def drop_variable_orders_below(f: mt.TPSA, order: int) -> mt.TPSA:
    out = f.descriptor.constant(0.0)
    for monomial, value in f.coefficients():
        if sum(monomial[:N_VAR]) >= order:
            out[monomial] = value
    return out


def tpsa_variable_order_terms(f: mt.TPSA, order: int) -> Iterable[tuple[tuple[int, ...], float]]:
    for monomial, value in f.coefficients():
        if sum(monomial[:N_VAR]) == order:
            yield monomial, value


def complex_substitute(f: mt.TPSA, substitutions: Sequence[CTPSA]) -> CTPSA:
    """Evaluate a real TPSA in complex substitutions for the two variables."""

    desc = f.descriptor
    out = CTPSA.zero(desc)
    for monomial, value in f.coefficients():
        j, k = monomial[:N_VAR]
        param_monomial = (0, 0, *monomial[N_VAR:])
        term = CTPSA.real(monomial_tpsa(desc, param_monomial, value))
        if j:
            term = term * (substitutions[0] ** j)
        if k:
            term = term * (substitutions[1] ** k)
        out = out + term
    return out


def substitute_real(f: mt.TPSA, substitutions: Sequence[mt.TPSA]) -> mt.TPSA:
    """Evaluate a real TPSA in real substitutions for x,p while preserving parameters.

    This mirrors MAD-NG map composition for the small examples here. Keeping it
    explicit makes the parameter flow visible and also lets the example run with
    any backend that implements scalar TPSA algebra, even if its map-composition
    shortcut has not implemented parameter propagation yet.
    """

    desc = f.descriptor
    out = desc.constant(0.0)
    for monomial, value in f.coefficients():
        j, k = monomial[:N_VAR]
        term = monomial_tpsa(desc, (0, 0, *monomial[N_VAR:]), value)
        if j:
            term = term * (substitutions[0] ** j)
        if k:
            term = term * (substitutions[1] ** k)
        out = out + term
    return out


def compose_real_map(outer: mt.TPSAMap, inner: mt.TPSAMap) -> mt.TPSAMap:
    return mt.TPSAMap(substitute_real(component, inner) for component in outer)


def chain_maps(*maps: mt.TPSAMap) -> mt.TPSAMap:
    """Compose maps in application order: chain_maps(a, b, c) means c after b after a."""

    if not maps:
        raise ValueError("at least one map is required")
    out = maps[0]
    for nxt in maps[1:]:
        out = compose_real_map(nxt, out)
    return out


def compose_complex_map(outer: Sequence[CTPSA], inner: Sequence[CTPSA]) -> tuple[CTPSA, CTPSA]:
    return tuple(complex_substitute(component.re, inner) + 1j * complex_substitute(component.im, inner) for component in outer)  # type: ignore[return-value]


def complex_map_from_real_map(real_map: mt.TPSAMap) -> tuple[CTPSA, CTPSA]:
    """Return z' and zbar' for a real normalized map [X', P']."""

    desc = real_map.descriptor
    root2 = sqrt(2.0)
    z = CTPSA.complex(real_map[0] / root2, real_map[1] / root2, desc)
    zbar = z.conj()
    return z, zbar


def real_from_complex_pair(z: CTPSA, zbar: CTPSA) -> tuple[mt.TPSA, mt.TPSA]:
    """Convert z,zbar to real normalized X,P, assuming zbar is the conjugate map."""

    root2 = sqrt(2.0)
    x = (z + zbar).re / root2
    p = ((z - zbar) / 1j).re / root2
    return x, p


def constant_plus_parameter(desc: mt.Descriptor, nominal: float, index: int) -> mt.TPSA:
    # Parameter TPSA values are order-1 in the C API. Placing the full-order
    # constant on the left promotes the resulting parameter expression to the
    # descriptor's full map order.
    return desc.constant(nominal) + desc.parameter(index + 1)


def build_fodo_map(desc: mt.Descriptor) -> mt.TPSAMap:
    x, p = desc.variables()
    kf = constant_plus_parameter(desc, 0.86, DKF)
    kd = constant_plus_parameter(desc, -0.74, DKD)
    ks = desc.constant(0.0) + desc.parameter(KS + 1)
    ko = desc.constant(0.0) + desc.parameter(KO + 1)
    kb = desc.constant(0.0) + desc.parameter(KB + 1)

    def drift(length: float) -> mt.TPSAMap:
        x0, p0 = desc.variables()
        return mt.TPSAMap([x0 + length * p0, p0])

    def quad(k: mt.TPSA) -> mt.TPSAMap:
        x0, p0 = desc.variables()
        return mt.TPSAMap([x0, p0 - x0 * k])

    def sextupole(k: mt.TPSA) -> mt.TPSAMap:
        x0, p0 = desc.variables()
        return mt.TPSAMap([x0, p0 - 0.5 * (x0 * x0) * k])

    def octupole(k: mt.TPSA) -> mt.TPSAMap:
        x0, p0 = desc.variables()
        return mt.TPSAMap([x0, p0 - (x0**3) * (k / 6.0)])

    def dipole(k: mt.TPSA) -> mt.TPSAMap:
        x0, p0 = desc.variables()
        return mt.TPSAMap([x0, p0 + k])

    qf = chain_maps(quad(kf), octupole(ko))
    qd = chain_maps(quad(kd), sextupole(ks), dipole(kb))

    # Start just before QF. The order below is the order experienced by a particle.
    return chain_maps(drift(0.5), qf, drift(1.0), qd, drift(0.5))


def solve_closed_orbit(one_turn: mt.TPSAMap, *, iterations: int) -> mt.TPSAMap:
    desc = one_turn.descriptor
    xco = desc.constant(0.0)
    pco = desc.constant(0.0)

    for _ in range(iterations):
        z = mt.TPSAMap([xco, pco])
        fz = compose_real_map(one_turn, z)
        residual_x = fz[0] - xco
        residual_p = fz[1] - pco

        j11 = substitute_real(one_turn[0].derivative(1), z) - 1.0
        j12 = substitute_real(one_turn[0].derivative(2), z)
        j21 = substitute_real(one_turn[1].derivative(1), z)
        j22 = substitute_real(one_turn[1].derivative(2), z) - 1.0

        det = j11 * j22 - j12 * j21
        dx = (-residual_x * j22 + j12 * residual_p) / det
        dp = (j21 * residual_x - j11 * residual_p) / det
        xco = xco + dx
        pco = pco + dp

    return mt.TPSAMap([xco, pco])


def translate_to_closed_orbit(one_turn: mt.TPSAMap, closed_orbit: mt.TPSAMap) -> mt.TPSAMap:
    desc = one_turn.descriptor
    x, p = desc.variables()
    shifted_in = mt.TPSAMap([x + closed_orbit[0], p + closed_orbit[1]])
    shifted_out = compose_real_map(one_turn, shifted_in)
    return mt.TPSAMap([shifted_out[0] - closed_orbit[0], shifted_out[1] - closed_orbit[1]])


def linear_normalizing_matrix(local_map: mt.TPSAMap) -> tuple[tuple[mt.TPSA, mt.TPSA], tuple[mt.TPSA, mt.TPSA], mt.TPSA, mt.TPSA]:
    """Return A, A inverse, cos(mu), sin(mu) as parameter TPSA values."""

    m11 = variable_coefficient(local_map[0], (1, 0))
    m12 = variable_coefficient(local_map[0], (0, 1))
    m21 = variable_coefficient(local_map[1], (1, 0))
    m22 = variable_coefficient(local_map[1], (0, 1))

    cos_mu = 0.5 * (m11 + m22)
    sin_mu = mt.sqrt(1.0 - cos_mu * cos_mu)
    beta = m12 / sin_mu
    alpha = (m11 - m22) / (2.0 * sin_mu)
    sqrt_beta = mt.sqrt(beta)

    # Physical coordinates = A * normalized real coordinates.
    a11 = sqrt_beta
    a12 = local_map.descriptor.constant(0.0)
    a21 = -alpha / sqrt_beta
    a22 = 1.0 / sqrt_beta

    inv11 = a22
    inv12 = local_map.descriptor.constant(0.0)
    inv21 = -a21
    inv22 = a11
    return ((a11, a12), (a21, a22)), ((inv11, inv12), (inv21, inv22)), cos_mu, sin_mu


def apply_linear_matrix(matrix: tuple[tuple[mt.TPSA, mt.TPSA], tuple[mt.TPSA, mt.TPSA]], vector: mt.TPSAMap) -> mt.TPSAMap:
    x, p = vector
    return mt.TPSAMap([
        x * matrix[0][0] + p * matrix[0][1],
        x * matrix[1][0] + p * matrix[1][1],
    ])


def normalize_linear(local_map: mt.TPSAMap, a: tuple[tuple[mt.TPSA, mt.TPSA], tuple[mt.TPSA, mt.TPSA]], ainv: tuple[tuple[mt.TPSA, mt.TPSA], tuple[mt.TPSA, mt.TPSA]]) -> mt.TPSAMap:
    desc = local_map.descriptor
    normalized_variables = mt.TPSAMap.identity(desc)
    to_physical = apply_linear_matrix(a, normalized_variables)
    to_normal = apply_linear_matrix(ainv, local_map)
    return compose_real_map(to_normal, to_physical)


def to_complex_coordinates(normalized_map: mt.TPSAMap) -> tuple[CTPSA, CTPSA]:
    """Rewrite a real normalized map in complex variables z,zbar."""

    desc = normalized_map.descriptor
    z, zbar = desc.variables()
    z = CTPSA.real(z)
    zbar = CTPSA.real(zbar)
    root2 = sqrt(2.0)
    x_sub = (z + zbar) / root2
    p_sub = (z - zbar) / (1j * root2)
    x1 = complex_substitute(normalized_map[0], [x_sub, p_sub])
    p1 = complex_substitute(normalized_map[1], [x_sub, p_sub])
    return (x1 + 1j * p1) / root2, (x1 - 1j * p1) / root2


def remove_nonresonant_terms(
    complex_map: tuple[CTPSA, CTPSA],
    lam: CTPSA,
    *,
    max_order: int,
) -> tuple[tuple[CTPSA, CTPSA], tuple[CTPSA, CTPSA]]:
    """Compute nonlinear normalizing map H and normal form N = H^-1 F H."""

    desc = lam.descriptor
    z_real, zbar_real = desc.variables()
    identity = (CTPSA.real(z_real), CTPSA.real(zbar_real))
    h_total = identity
    current = complex_map
    lambar = lam.conj()

    for order in range(2, max_order + 1):
        h_z = CTPSA.zero(desc)
        h_zbar = CTPSA.zero(desc)

        for j in range(order + 1):
            k = order - j
            monomial_eigenvalue = (lam**j) * (lambar**k)

            cz = current[0].variable_coefficient(j, k)
            if list(cz.re.coefficients()) or list(cz.im.coefficients()):
                resonant_for_z = j == k + 1
                if not resonant_for_z:
                    h_z = h_z + cz / (monomial_eigenvalue - lam) * (identity[0] ** j) * (identity[1] ** k)

            czbar = current[1].variable_coefficient(j, k)
            if list(czbar.re.coefficients()) or list(czbar.im.coefficients()):
                resonant_for_zbar = k == j + 1
                if not resonant_for_zbar:
                    h_zbar = h_zbar + czbar / (monomial_eigenvalue - lambar) * (identity[0] ** j) * (identity[1] ** k)

        if not (list(h_z.re.coefficients()) or list(h_z.im.coefficients()) or list(h_zbar.re.coefficients()) or list(h_zbar.im.coefficients())):
            continue

        h_order = (identity[0] + h_z, identity[1] + h_zbar)
        h_order_inverse = (identity[0] - h_z, identity[1] - h_zbar)

        # Order-by-order near-identity inverse is sufficient for removing the
        # selected order. Full composition keeps lower-order removals intact up
        # to truncation order.
        current = compose_complex_map(h_order_inverse, compose_complex_map(current, h_order))
        h_total = compose_complex_map(h_total, h_order)

    return h_total, current


def detuning_terms(normal_form: tuple[CTPSA, CTPSA], lam: CTPSA, *, max_order: int) -> list[tuple[int, mt.TPSA, mt.TPSA]]:
    """Return [(q, dmu_q, growth_q)] for z' = lambda z exp(i sum dmu_q J^q)."""

    out: list[tuple[int, mt.TPSA, mt.TPSA]] = []
    for q in range(1, (max_order - 1) // 2 + 1):
        coeff = normal_form[0].variable_coefficient(q + 1, q)
        phase_coeff = coeff / lam
        out.append((q, phase_coeff.im, phase_coeff.re))
    return out



def acos_parameter_series(cos_mu: mt.TPSA) -> mt.TPSA:
    """Parameter TPSA for acos(cos_mu) through the descriptor parameter order.

    Some minimal TPSA backends implement the elementary functions for variable
    series first and not for pure-parameter series. The normal-form example only
    needs a short tune diagnostic, so this explicit Taylor expansion keeps the
    example backend-independent.
    """

    c0 = cos_mu[()]
    s0 = sqrt(1.0 - c0 * c0)
    delta = cos_mu - c0
    mu = cos_mu.descriptor.constant(acos(c0)) - delta / s0
    if cos_mu.descriptor.parameter_order >= 2:
        mu = mu - (c0 / (2.0 * s0**3)) * (delta * delta)
    return mu

def format_series(f: mt.TPSA, *, eps: float = 1e-12, max_terms: int = 10) -> str:
    terms: list[str] = []
    for monomial, value in f.coefficients():
        if abs(value) < eps:
            continue
        pieces = []
        for name, power in zip(PARAMETER_NAMES, monomial[N_VAR:]):
            if power == 1:
                pieces.append(name)
            elif power:
                pieces.append(f"{name}^{power}")
        basis = "*".join(pieces) if pieces else "1"
        terms.append(f"{value:+.6e}*{basis}")
        if len(terms) >= max_terms:
            break
    return " ".join(terms) if terms else "0"


def print_matrix(name: str, matrix: tuple[tuple[mt.TPSA, mt.TPSA], tuple[mt.TPSA, mt.TPSA]]) -> None:
    print(name)
    for row in matrix:
        print("  [", format_series(row[0]), ",", format_series(row[1]), "]")


def main() -> None:
    map_order = 5
    parameter_order = 2
    desc = mt.DescriptorBuilder().variables(2).order(map_order).parameters(len(PARAMETER_NAMES), order=parameter_order).build()

    one_turn = build_fodo_map(desc)
    closed_orbit = solve_closed_orbit(one_turn, iterations=map_order + parameter_order)
    local_map = translate_to_closed_orbit(one_turn, closed_orbit)

    a, ainv, cos_mu, sin_mu = linear_normalizing_matrix(local_map)
    normalized_map = normalize_linear(local_map, a, ainv)

    # With z = (X + i P)/sqrt(2) and R = [[cos, sin], [-sin, cos]],
    # the linear complex eigenvalue is exp(-i*mu).
    lam = CTPSA.complex(cos_mu, -sin_mu)
    complex_map = to_complex_coordinates(normalized_map)
    nonlinear_a, normal_form = remove_nonresonant_terms(complex_map, lam, max_order=map_order)

    print("Parameter-dependent closed orbit")
    print("  x_co =", format_series(closed_orbit[0]))
    print("  p_co =", format_series(closed_orbit[1]))
    print()

    print_matrix("Parameter-dependent linear normalizing matrix A", a)
    print()

    linear_tune = acos_parameter_series(cos_mu) / (2.0 * pi)
    print("Linear tune series Q = mu/(2*pi)")
    print("  Q =", format_series(linear_tune))
    print("  cos(mu) =", format_series(cos_mu))
    print("  sin(mu) =", format_series(sin_mu))
    print()

    print("Nonlinear complex normalizing map H: z_old = H_z(z, zbar; parameters)")
    for order in range(2, map_order + 1):
        hz_terms = [item for item in tpsa_variable_order_terms(nonlinear_a[0].re, order)]
        hi_terms = [item for item in tpsa_variable_order_terms(nonlinear_a[0].im, order)]
        if hz_terms or hi_terms:
            print(f"  H_z order {order}: {len(hz_terms)} real terms, {len(hi_terms)} imaginary terms")
    print()

    print("Detuning coefficients for z' = lambda*z*exp(i*(phi1*J + phi2*J^2 + ...))")
    print("where lambda = exp(-i*mu) and J = z*zbar = (X^2 + P^2)/2.")
    print("The conventional tune shift is dQ/dJ^q = -phi_q/(2*pi) for this z convention.")
    for q, phi_q, growth_q in detuning_terms(normal_form, lam, max_order=map_order):
        print(f"  dQ/dJ^{q} = ({format_series(-phi_q / (2.0 * pi))})")
        if any(abs(v) > 1e-10 for _, v in growth_q.coefficients()):
            print(f"    diagnostic real part c/lambda = {format_series(growth_q)}")


if __name__ == "__main__":
    main()
