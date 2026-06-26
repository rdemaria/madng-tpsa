from __future__ import annotations

import madng_tpsa as mt


def test_import_is_lazy_and_availability_has_bool_contract():
    available = mt.is_available()
    assert isinstance(available, bool)
    if available:
        assert mt.loaded_library_path() is not None
    else:
        assert mt.availability_error() is not None


def test_descriptor_builder_repr_and_validation_without_library():
    builder = mt.DescriptorBuilder().variables(2).order(4).parameters(1, order=1).orders(4, 4, 1)
    text = repr(builder)
    assert "n_variables=2" in text
    assert "max_order=4" in text
    assert "orders=(4, 4, 1)" in text


def test_numeric_math_fallbacks():
    assert mt.sin(0.0) == 0.0
    assert mt.cos(0.0) == 1.0
    assert mt.sinc(0.0) == 1.0
