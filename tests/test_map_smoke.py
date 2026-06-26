import math

import madng_tpsa as mt

from conftest import requires_madng


@requires_madng
def test_map_eval_and_compose():
    d = mt.desc(2, 5)
    x, px = d.variables()
    kick = mt.TPSAMap([x, px + 0.1 * x**2])
    assert kick.eval([1.0, 0.0]) == (1.0, 0.1)

    twice = kick @ kick
    assert math.isclose(twice[1].coeff([2, 0]), 0.2)
