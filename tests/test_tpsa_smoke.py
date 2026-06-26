import math

import madng_tpsa as mt

from conftest import requires_madng


@requires_madng
def test_operator_overloads_and_functions():
    d = mt.desc(2, 5)
    x, y = d.variables()
    f = mt.sin(x) + x * y + 2.0
    assert math.isclose(f.constant, 2.0)
    assert math.isclose(f.coeff([1, 1]), 1.0)
    assert math.isclose(f.deriv(1).coeff([1, 0]), 1.0)

    g = (x + 1) ** 3
    assert math.isclose(g.coeff([0, 0]), 1.0)
    assert math.isclose(g.coeff([1, 0]), 3.0)
    assert math.isclose(g.coeff([2, 0]), 3.0)
    assert math.isclose(g.coeff([3, 0]), 1.0)


@requires_madng
def test_copy_and_mutation_are_independent():
    d = mt.desc(1, 3)
    (x,) = d.variables()
    y = x.copy()
    y += 2.0
    assert x.constant == 0.0
    assert y.constant == 2.0
