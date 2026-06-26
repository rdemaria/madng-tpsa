from __future__ import annotations

import madng_tpsa as mt


def test_bundled_backend_is_loadable():
    assert mt.is_available(), mt.availability_error()
    assert mt.loaded_library_path() is not None
    lib = mt.load_library()
    assert hasattr(lib, "mad_tpsa_newd")
    assert hasattr(lib, "mad_ctpsa_newd")
    assert hasattr(lib, "mad_ctpsa_compose")


def test_descriptor_builder_repr_and_validation():
    builder = mt.DescriptorBuilder().variables(2).order(4).parameters(1, order=1).orders(4, 4, 1)
    text = repr(builder)
    assert "n_variables=2" in text
    assert "max_order=4" in text
    assert "orders=(4, 4, 1)" in text


def test_numeric_math_fallbacks():
    assert mt.sin(0.0) == 0.0
    assert mt.cos(0.0) == 1.0
    assert mt.sinc(0.0) == 1.0
    assert mt.conj(1 + 2j) == 1 - 2j
