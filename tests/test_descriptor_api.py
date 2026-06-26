import madng_tpsa as mt

from conftest import requires_madng


@requires_madng
def test_descriptor_builder_and_variables():
    d = mt.DescriptorBuilder().variables(2).order(3).parameters(1, order=1).build()
    assert d.nv == 2
    assert d.np == 1
    assert d.mo == 3
    x, y = d.variables()
    k, = d.parameters()
    assert x.coeff([1, 0, 0]) == 1.0
    assert y.coeff([0, 1, 0]) == 1.0
    assert k.coeff([0, 0, 1]) == 1.0


@requires_madng
def test_descriptor_index_roundtrip():
    d = mt.desc(2, 3)
    idx = d.index([2, 1])
    assert d.mono(idx) == (2, 1)
