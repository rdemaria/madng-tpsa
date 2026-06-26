"""Descriptor parameters example.

Run with:
    python examples/parameters.py
"""

from __future__ import annotations

import madng_tpsa as mt


def main() -> None:
    desc = (
        mt.DescriptorBuilder()
        .variables(2)
        .order(4)
        .parameters(1, order=1)
        .build()
    )
    x, y = desc.variables()
    k = desc.parameter(1)

    # Dense monomials include variables followed by parameters: (x, y, k).
    f = mt.sin(x) + y**2 + x * k + 2.0

    print("descriptor dimensions:", desc.n_total)
    print("coefficient of x*k:", f[1, 0, 1])
    print("f(x=0.2, y=0.1, k=3.0):", f.evaluate([0.2, 0.1, 3.0]))


if __name__ == "__main__":
    main()
