from __future__ import annotations

import cmath

import pytest

import madng_tpsa as mt


def test_complex_coefficients_and_real_imag_projection():
    d = mt.descriptor(2, 4)
    x, y = d.variables()
    z = mt.CTPSA.from_tpsa(x) + 1j * mt.CTPSA.from_tpsa(y) + (2.0 - 3.0j)

    assert z[()] == pytest.approx(2.0 - 3.0j)
    assert z[1, 0] == pytest.approx(1.0 + 0.0j)
    assert z[0, 1] == pytest.approx(0.0 + 1.0j)
    assert z.real[1, 0] == pytest.approx(1.0)
    assert z.real[0, 1] == pytest.approx(0.0)
    assert z.imag[0, 1] == pytest.approx(1.0)


def test_complex_algebra_and_math_functions():
    d = mt.descriptor(1, 5)
    (x,) = d.variables()
    z = mt.CTPSA.from_tpsa(x) + (0.3 + 0.2j)

    f = mt.exp(mt.log(z))
    assert f[()] == pytest.approx(0.3 + 0.2j)
    assert f[1] == pytest.approx(1.0 + 0.0j)

    s, c = mt.sincos(z)
    assert s[()] == pytest.approx(cmath.sin(0.3 + 0.2j))
    assert c[()] == pytest.approx(cmath.cos(0.3 + 0.2j))


def test_complex_map_composition_evaluation_and_inverse():
    d = mt.descriptor(2, 4)
    x, p = d.variables()
    xc = mt.CTPSA.from_tpsa(x)
    pc = mt.CTPSA.from_tpsa(p)

    mapping = mt.CTPSAMap([xc + (1.0 + 0.5j) * pc, pc - 0.05j * xc**2])
    identity = mt.CTPSAMap.identity(d)
    composed = mapping @ identity
    assert composed.evaluate([0.001, 0.002]) == pytest.approx(mapping.evaluate([0.001, 0.002]))

    inv = mapping.inverse()
    roundtrip = inv @ mapping
    assert roundtrip.evaluate([0.001, 0.002]) == pytest.approx((0.001 + 0j, 0.002 + 0j))
