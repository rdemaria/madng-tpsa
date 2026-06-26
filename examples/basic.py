"""Basic scalar TPSA example."""

import madng_tpsa as mt
from madng_tpsa import cos, exp, sin


def main() -> None:
    d = mt.desc(nv=2, order=5)
    x, y = d.variables(names=["x", "y"])

    f = sin(x) + x * y + 2.0
    g = exp(x + y) * cos(x)

    print("f constant:", f.constant)
    print("f[x*y]:", f.coeff([1, 1]))
    print("df/dx coefficients:", f.deriv(0).coefficients())
    print("g non-zero coefficients:", g.coefficients())


if __name__ == "__main__":
    mt.load_library()  # Optional: fail early with a useful error message.
    main()
