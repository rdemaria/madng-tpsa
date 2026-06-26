"""Variables plus parameter TPSA values."""

import madng_tpsa as mt


def main() -> None:
    d = mt.DescriptorBuilder().variables(2).order(4).parameters(2, order=1).build()
    x, y = d.variables(names=["x", "y"])
    kx, ky = d.parameters(names=["kx", "ky"])

    h = 0.5 * (x**2 + y**2) + kx * x**2 + ky * y**2
    print("Hamiltonian coefficients:")
    for mono, value in sorted(h.coefficients().items()):
        print(mono, value)


if __name__ == "__main__":
    mt.load_library()
    main()
