from __future__ import annotations

import math

import pytest

import madng_tpsa as mt

pytestmark = pytest.mark.skipif(
    not mt.is_available(),
    reason=mt.availability_error() or "MAD-NG TPSA shared library is not available",
)


def test_polynomial_algebra_and_coefficients():
    desc = mt.DescriptorBuilder().variables(2).order(5).build()
    x, y = desc.variables()

    f = mt.sin(x) + y**2 + 3.0 * x * y + 2.0

    assert f[0, 0] == pytest.approx(2.0)
    assert f[1, 0] == pytest.approx(1.0)
    assert f[3, 0] == pytest.approx(-1.0 / 6.0)
    assert f[0, 2] == pytest.approx(1.0)
    assert f[1, 1] == pytest.approx(3.0)


def test_derivative_and_evaluation():
    desc = mt.descriptor(2, 4)
    x, y = desc.variables()
    f = x**2 + 2 * x * y + mt.exp(y)

    dfdx = f.derivative(1)
    assert dfdx[1, 0] == pytest.approx(2.0)
    assert dfdx[0, 1] == pytest.approx(2.0)
    assert f.evaluate([0.2, 0.1]) == pytest.approx(0.2**2 + 2 * 0.2 * 0.1 + math.exp(0.1))


def test_map_identity_composition_and_evaluation():
    desc = mt.descriptor(2, 4)
    x, y = desc.variables()
    identity = mt.TPSAMap.identity(desc)
    outer = mt.TPSAMap([x + y**2, y + 1.0])

    composed = outer @ identity
    assert composed[0][1, 0] == pytest.approx(1.0)
    assert composed[0][0, 2] == pytest.approx(1.0)
    assert composed[1][0, 0] == pytest.approx(1.0)
    assert composed.evaluate([0.2, 0.3]) == pytest.approx((0.2 + 0.3**2, 1.3))


def test_lapack_backed_c_map_inverse():
    desc = mt.descriptor(2, 4)
    x, p = desc.variables()
    mapping = mt.TPSAMap([x + 1.2 * p, p - 0.05 * x**2])
    inv = mapping.inverse()
    roundtrip = inv @ mapping
    assert roundtrip.evaluate([0.001, 0.002]) == pytest.approx((0.001, 0.002))
    assert roundtrip[0][1, 0] == pytest.approx(1.0)
    assert roundtrip[0][0, 1] == pytest.approx(0.0)
    assert roundtrip[1][1, 0] == pytest.approx(0.0)
    assert roundtrip[1][0, 1] == pytest.approx(1.0)
