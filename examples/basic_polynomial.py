"""Basic TPSA algebra example.

Run with:
    python examples/basic_polynomial.py
"""

from __future__ import annotations

import madng_tpsa as mt


def main() -> None:
    desc = mt.DescriptorBuilder().variables(2).order(5).build()
    x, y = desc.variables()

    f = mt.sin(x) + y**2 + 3.0 * x * y + 2.0

    print("f = sin(x) + y^2 + 3xy + 2")
    print("constant:", f[0, 0])
    print("x coefficient:", f[1, 0])
    print("x^3 coefficient:", f[3, 0])
    print("y^2 coefficient:", f[0, 2])
    print("f(0.2, 0.1):", f.evaluate([0.2, 0.1]))


if __name__ == "__main__":
    main()
