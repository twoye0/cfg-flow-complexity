import math

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss

import experiments.exp05b_velocity_coupling as exp05b


torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

N_STEPS = 100
QUADRATURE_ORDER = 20

BETAS = [
    0.0,
    math.pi / 12.0,   # 15 deg
    math.pi / 6.0,    # 30 deg
    math.pi / 4.0,    # 45 deg
    math.pi / 3.0,    # 60 deg
]


# ------------------------------------------------------------
# 2D Gauss-Hermite quadrature for z ~ N(0, I)
# ------------------------------------------------------------

def gauss_hermite_2d(order):
    nodes, weights = hermgauss(order)

    points = []
    quad_weights = []

    for i in range(order):
        for j in range(order):
            z = np.sqrt(2.0) * np.array(
                [nodes[i], nodes[j]]
            )

            weight = (
                weights[i]
                * weights[j]
                / math.pi
            )

            points.append(z)
            quad_weights.append(weight)

    return (
        np.asarray(points),
        np.asarray(quad_weights),
    )


# ------------------------------------------------------------
# Low-information velocity statistics
# ------------------------------------------------------------

def velocity_statistics(x, t, beta):
    """
    Compute instantaneous scalar statistics of the
    symmetric velocity Jacobian

        S = (A + A^T) / 2,

    where A = grad_x v_t(x).

    Returns
    -------
    trace_s:
        tr(S), preserving signed total stretching.

    frob_sq_s:
        ||S||_F^2, measuring stretching magnitude
        while discarding its sign.
    """

    A = exp05b.velocity_jacobian(
        x,
        t,
        beta,
    )

    S = 0.5 * (A + A.T)

    trace_s = torch.trace(S)

    frob_sq_s = torch.sum(S * S)

    return trace_s, frob_sq_s


# ------------------------------------------------------------
# Joint trajectory integration
#
# We integrate:
#
#   dx/dt = v_t(x)
#   dJ/dt = A_t J
#
# together with two scalar accumulators:
#
#   dT/dt = 2 tr(S_t)
#   dF/dt = ||S_t||_F^2
#
# Therefore at t=1:
#
#   T_1 = 2 integral tr(S_t) dt
#   F_1 = integral ||S_t||_F^2 dt
#
# for this individual trajectory.
# ------------------------------------------------------------

def augmented_dynamics(x, J, t, beta):
    dx, dJ = exp05b.joint_dynamics(
        x,
        J,
        t,
        beta,
    )

    trace_s, frob_sq_s = velocity_statistics(
        x,
        t,
        beta,
    )

    d_trace = 2.0 * trace_s
    d_frob = frob_sq_s

    return dx, dJ, d_trace, d_frob


def augmented_rk4_step(
    x,
    J,
    trace_acc,
    frob_acc,
    t,
    dt,
    beta,
):
    k1_x, k1_J, k1_trace, k1_frob = (
        augmented_dynamics(
            x,
            J,
            t,
            beta,
        )
    )

    k2_x, k2_J, k2_trace, k2_frob = (
        augmented_dynamics(
            x + 0.5 * dt * k1_x,
            J + 0.5 * dt * k1_J,
            t + 0.5 * dt,
            beta,
        )
    )

    k3_x, k3_J, k3_trace, k3_frob = (
        augmented_dynamics(
            x + 0.5 * dt * k2_x,
            J + 0.5 * dt * k2_J,
            t + 0.5 * dt,
            beta,
        )
    )

    k4_x, k4_J, k4_trace, k4_frob = (
        augmented_dynamics(
            x + dt * k3_x,
            J + dt * k3_J,
            t + dt,
            beta,
        )
    )

    x_new = x + (dt / 6.0) * (
        k1_x
        + 2.0 * k2_x
        + 2.0 * k3_x
        + k4_x
    )

    J_new = J + (dt / 6.0) * (
        k1_J
        + 2.0 * k2_J
        + 2.0 * k3_J
        + k4_J
    )

    trace_new = trace_acc + (dt / 6.0) * (
        k1_trace
        + 2.0 * k2_trace
        + 2.0 * k3_trace
        + k4_trace
    )

    frob_new = frob_acc + (dt / 6.0) * (
        k1_frob
        + 2.0 * k2_frob
        + 2.0 * k3_frob
        + k4_frob
    )

    return (
        x_new,
        J_new,
        trace_new,
        frob_new,
    )


def integrate_augmented_sample(z, beta):
    x = z.clone()

    J = torch.eye(
        2,
        dtype=z.dtype,
        device=z.device,
    )

    trace_acc = torch.zeros(
        (),
        dtype=z.dtype,
        device=z.device,
    )

    frob_acc = torch.zeros(
        (),
        dtype=z.dtype,
        device=z.device,
    )

    dt = 1.0 / N_STEPS
    t = 0.0

    for _ in range(N_STEPS):
        (
            x,
            J,
            trace_acc,
            frob_acc,
        ) = augmented_rk4_step(
            x,
            J,
            trace_acc,
            frob_acc,
            t,
            dt,
            beta,
        )

        t += dt

    return (
        x,
        J,
        trace_acc,
        frob_acc,
    )


# ------------------------------------------------------------
# Population quantities using Gauss-Hermite quadrature
# ------------------------------------------------------------

def compute_population_quantities(beta):
    points, weights = gauss_hermite_2d(
        QUADRATURE_ORDER
    )

    complexity = 0.0
    p_trace = 0.0
    p_frob = 0.0

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

        (
            _,
            J1,
            trace_acc,
            frob_acc,
        ) = integrate_augmented_sample(
            z,
            beta,
        )

        complexity_i = torch.sum(
            J1 * J1
        ).item()

        complexity += (
            weight * complexity_i
        )

        p_trace += (
            weight * trace_acc.item()
        )

        p_frob += (
            weight * frob_acc.item()
        )

    return complexity, p_trace, p_frob


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # Keep the imported module consistent with this experiment.
    exp05b.N_STEPS = N_STEPS

    print(
        "Exp 5-C — Low-Information Velocity Baselines"
    )
    print("=" * 78)

    print(f"RK4 steps: {N_STEPS}")
    print(
        f"Gauss-Hermite order: "
        f"{QUADRATURE_ORDER}"
    )
    print(
        f"Latent points per beta: "
        f"{QUADRATURE_ORDER ** 2}"
    )

    print()

    results = []

    for beta in BETAS:
        beta_deg = round(
            beta * 180.0 / math.pi,
            10,
        )

        print("=" * 78)
        print(
            f"beta = {beta_deg:.1f} deg"
        )
        print("=" * 78)

        (
            complexity,
            p_trace,
            p_frob,
        ) = compute_population_quantities(
            beta
        )

        results.append(
            {
                "beta_deg": beta_deg,
                "C1": complexity,
                "P_trace": p_trace,
                "P_frob": p_frob,
            }
        )

        print()
        print(
            f"C1      = {complexity:.12f}"
        )
        print(
            f"P_trace = {p_trace:.12f}"
        )
        print(
            f"P_frob  = {p_frob:.12f}"
        )
        print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)

    print(
        f"{'beta':>8}  "
        f"{'C1':>16}  "
        f"{'P_trace':>16}  "
        f"{'P_frob':>16}"
    )

    for r in results:
        print(
            f"{r['beta_deg']:8.1f}  "
            f"{r['C1']:16.12f}  "
            f"{r['P_trace']:16.12f}  "
            f"{r['P_frob']:16.12f}"
        )

    # --------------------------------------------------------
    # Changes relative to beta = 0
    # --------------------------------------------------------

    baseline = results[0]

    print()
    print(
        "Changes relative to beta = 0"
    )
    print("-" * 78)

    print(
        f"{'beta':>8}  "
        f"{'Delta C1':>16}  "
        f"{'Delta P_trace':>18}  "
        f"{'Delta P_frob':>18}"
    )

    for r in results:
        delta_c = (
            r["C1"]
            - baseline["C1"]
        )

        delta_trace = (
            r["P_trace"]
            - baseline["P_trace"]
        )

        delta_frob = (
            r["P_frob"]
            - baseline["P_frob"]
        )

        print(
            f"{r['beta_deg']:8.1f}  "
            f"{delta_c:16.12f}  "
            f"{delta_trace:18.12f}  "
            f"{delta_frob:18.12f}"
        )


if __name__ == "__main__":
    main()