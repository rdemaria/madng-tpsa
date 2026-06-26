"""TPSA map composition and inverse example.

Run with:
    python examples/maps.py
"""

from __future__ import annotations

import madng_tpsa as mt


def drift(length: float, desc: mt.Descriptor) -> mt.TPSAMap:
    x, px = desc.variables()
    return mt.TPSAMap([x + length * px, px])


def kick(strength: float, desc: mt.Descriptor) -> mt.TPSAMap:
    x, px = desc.variables()
    return mt.TPSAMap([x, px - strength * x**2])


def main() -> None:
    desc = mt.descriptor(2, 5)

    one_turn = drift(1.0, desc) @ kick(0.2, desc) @ drift(1.0, desc)
    point = [1e-3, 0.0]
    image = one_turn.evaluate(point)

    inv = one_turn.inverse()
    recovered = inv.evaluate(image)

    print("one-turn map at x=1e-3, px=0:", image)
    print("inverse applied to that image:", recovered)


if __name__ == "__main__":
    main()
