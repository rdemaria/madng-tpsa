"""TPSA map evaluation and composition example."""

import madng_tpsa as mt


def main() -> None:
    d = mt.desc(nv=2, order=5)
    x, px = d.variables(names=["x", "px"])

    identity = mt.TPSAMap.identity(d, 2)
    kick = mt.TPSAMap([x, px + 0.1 * x**2])
    twice = kick @ kick

    print("identity at [1, 2]:", identity.eval([1.0, 2.0]))
    print("kick at [1, 0]:", kick.eval([1.0, 0.0]))
    print("twice second component:", twice[1].coefficients())


if __name__ == "__main__":
    mt.load_library()
    main()
