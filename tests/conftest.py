import pytest

import madng_tpsa as mt


def madng_available() -> bool:
    try:
        mt.load_library()
    except mt.LibraryNotFoundError:
        return False
    return True


requires_madng = pytest.mark.skipif(
    not madng_available(), reason="requires a loadable MAD-NG shared library"
)
