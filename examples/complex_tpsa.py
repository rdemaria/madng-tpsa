"""Complex TPSA example using the mad_ctpsa C API wrapper.

Run with:
    python examples/complex_tpsa.py
"""

from __future__ import annotations

import madng_tpsa as mt


def main() -> None:
    desc = mt.descriptor(2, 5)
    x, y = desc.variables()

    z = mt.CTPSA.from_tpsa(x) + 1j * mt.CTPSA.from_tpsa(y)
    f = mt.exp(1j * z) + z**2

    print("f constant:", f[0, 0])
    print("f[x]:", f[1, 0])
    print("f[y]:", f[0, 1])
    print("real(f)[x^2]:", f.real[2, 0])
    print("imag(f)[x*y]:", f.imag[1, 1])

    mapping = mt.CTPSAMap([z, mt.conj(z)])
    print("map(0.1, 0.2):", mapping.evaluate([0.1, 0.2]))


if __name__ == "__main__":
    main()
