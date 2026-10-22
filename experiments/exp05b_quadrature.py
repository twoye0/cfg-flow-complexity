import math

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

import experiments.exp05b_velocity_coupling as exp05b
from experiments.exp05b_velocity_coupling import integrate_sample


torch.set_default_dtype(torch.float64)

# RK4 convergence was already verified.
N_STEPS = 100

# Gauss-Hermite convergence check.
QUADRATURE_ORDERS = [
    10,
    20,
    30,
]

BETAS = [
    0.0,
    math.pi / 3.0,   # 60 degrees
]


def gauss_hermite_2d(order):
    """
    Construct tensor-product Gauss-Hermite quadrature
    for z ~ N(0, I_2).

    numpy.hermgauss approximates

        integral exp(-x^2) f(x) dx

    while for Z ~ N(0,1),

        E[f(Z)]
        =
        1/sqrt(pi)
        integral exp(-x^2) f(sqrt(2) x) dx.

    Therefore in 2D,

        E[f(Z1,Z2)]
        ≈
        1/pi sum_i sum_j
        w_i w_j
        f(sqrt(2)x_i, sqrt(2)x_j).
    """

    nodes_1d, weights_1d = hermgauss(order)

    points = []
    weights = []

    for i in range(order):
        for j in range(order):
            z = np.sqrt(2.0) * np.array([
                nodes_1d[i],
                nodes_1d[j],
            ])

            weight = (
                weights_1d[i]
                * weights_1d[j]
                / math.pi
            )

            points.append(z)
            weights.append(weight)

    points = np.asarray(points)
    weights = np.asarray(weights)

    return points, weights


def compute_complexity(beta, order):
    points, weights = gauss_hermite_2d(order)

    weighted_complexity = 0.0

    n_points = len(points)

    for index, (z_np, weight) in enumerate(
        zip(points, weights)
    ):
        if (
            index == 0
            or (index + 1) % 100 == 0
            or index + 1 == n_points
        ):
            print(
                f"    point {index + 1}/{n_points}"
            )

        z = torch.tensor(
            z_np,
            dtype=torch.get_default_dtype(),
        )

        _, J1 = integrate_sample(
            z,
            beta,
        )

        complexity_i = torch.sum(
            J1 * J1
        ).item()

        weighted_complexity += (
            weight * complexity_i
        )

    return weighted_complexity


def main():
    # Use the RK4 resolution already validated in the
    # previous convergence experiment.
    exp05b.N_STEPS = N_STEPS

    print(
        "Exp 5-B — Gauss-Hermite Population Check"
    )
    print("=" * 72)

    print(f"RK4 steps: {N_STEPS}")
    print(
        "Quadrature orders: "
        f"{QUADRATURE_ORDERS}"
    )
    print()

    results = {}

    for order in QUADRATURE_ORDERS:
        print("=" * 72)
        print(
            f"Gauss-Hermite order = {order} "
            f"({order ** 2} latent points)"
        )
        print("=" * 72)

        order_results = {}

        for beta in BETAS:
            beta_deg = round(
                beta * 180.0 / math.pi,
                10,
            )

            print()
            print(
                f"beta = {beta_deg:.1f} deg"
            )

            complexity = compute_complexity(
                beta,
                order,
            )

            order_results[beta_deg] = complexity

            print(
                f"    C1 = {complexity:.12f}"
            )

        c0 = order_results[0.0]
        c60 = order_results[60.0]

        delta = c60 - c0

        results[order] = {
            "C0": c0,
            "C60": c60,
            "delta": delta,
        }

        print()
        print(
            f"Delta C = C60 - C0 "
            f"= {delta:+.12f}"
        )

    # ========================================================
    # Convergence summary
    # ========================================================

    print()
    print("=" * 72)
    print("Quadrature convergence summary")
    print("=" * 72)

    print(
        f"{'order':>8}  "
        f"{'points':>8}  "
        f"{'C0':>14}  "
        f"{'C60':>14}  "
        f"{'Delta C':>14}"
    )

    for order in QUADRATURE_ORDERS:
        r = results[order]

        print(
            f"{order:8d}  "
            f"{order ** 2:8d}  "
            f"{r['C0']:14.10f}  "
            f"{r['C60']:14.10f}  "
            f"{r['delta']:14.10f}"
        )

    print()
    print(
        "Successive changes in Delta C"
    )

    for previous_order, current_order in zip(
        QUADRATURE_ORDERS[:-1],
        QUADRATURE_ORDERS[1:],
    ):
        previous = results[
            previous_order
        ]["delta"]

        current = results[
            current_order
        ]["delta"]

        change = abs(
            current - previous
        )

        print(
            f"{previous_order:2d} -> "
            f"{current_order:2d}: "
            f"|DeltaC_curr - DeltaC_prev| "
            f"= {change:.3e}"
        )

    print()
    print("=" * 72)

    final_order = QUADRATURE_ORDERS[-1]
    final_delta = results[
        final_order
    ]["delta"]

    print(
        f"Highest-order Delta C "
        f"= {final_delta:+.12f}"
    )

    if final_delta > 0:
        print(
            "Highest-order quadrature gives "
            "C60 > C0."
        )
    elif final_delta < 0:
        print(
            "Highest-order quadrature gives "
            "C60 < C0."
        )
    else:
        print(
            "Highest-order quadrature gives "
            "C60 = C0."
        )


if __name__ == "__main__":
    main()