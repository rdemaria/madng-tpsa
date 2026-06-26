"""TPSA map example.

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
    print("one-turn map at x=1e-3, px=0:", one_turn.evaluate([1e-3, 0.0]))

    inv = one_turn.inverse()
    recovered = inv @ one_turn
    print("inverse composed with map at x=1e-3, px=0:", recovered.evaluate([1e-3, 0.0]))


if __name__ == "__main__":
    main()
