import math

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from torch.func import jacrev


torch.set_default_dtype(torch.float64)


# ============================================================
# Configuration
# ============================================================

N_STEPS = 100
QUADRATURE_ORDER = 20

TARGET_CLASS = 1
OMEGA = 2.0

ALPHAS = [
    -0.50,
    -0.25,
    0.00,
    0.25,
    0.50,
]


# ============================================================
# Exp2 distributions
# ============================================================

CLASS_MEANS = torch.tensor([
    [-2.0, 0.0],
    [ 2.0, 0.0],
])

CLASS_COVS = torch.stack([
    torch.diag(torch.tensor([0.25, 1.0])),
    torch.diag(torch.tensor([1.0, 0.25])),
])


# ============================================================
# Genuine interpolation family
#
# x_t = a_alpha(t) z + b(t) x
#
# a_alpha(t)
#   = (1-t) [1 + alpha t(1-t)]
#
# b(t) = t
#
# All alpha values have the same endpoints:
#
#   a(0)=1, b(0)=0
#   a(1)=0, b(1)=1
#
# But for alpha != 0, the intermediate class-conditional
# covariance changes:
#
#   C_t,c = a(t)^2 I + b(t)^2 Sigma_c
#
# Therefore this is a genuinely different probability path,
# not merely a different transport coupling for the same p_t.
# ============================================================

def interpolation_coefficients(t, alpha):
    t = torch.as_tensor(
        t,
        dtype=torch.get_default_dtype(),
    )

    alpha = torch.as_tensor(
        alpha,
        dtype=torch.get_default_dtype(),
    )

    a = (
        (1.0 - t)
        * (
            1.0
            + alpha * t * (1.0 - t)
        )
    )

    b = t

    # d/dt [(1-t)(1 + alpha t(1-t))]
    #
    # = -1 + alpha (1-t)(1-3t)
    a_dot = (
        -1.0
        + alpha
        * (1.0 - t)
        * (1.0 - 3.0 * t)
    )

    b_dot = torch.ones_like(t)

    return a, b, a_dot, b_dot


# ============================================================
# Conditional marginal
#
# If
#
#   z ~ N(0, I)
#   x | c ~ N(mu_c, Sigma_c)
#
# and
#
#   x_t = a z + b x,
#
# then
#
#   mean = b mu_c
#
#   cov  = a^2 I + b^2 Sigma_c
#
# and
#
#   K = Cov(dot{x_t}, x_t)
#     = a_dot a I + b_dot b Sigma_c.
# ============================================================

def conditional_marginal(t, c, alpha):
    a, b, a_dot, b_dot = (
        interpolation_coefficients(
            t,
            alpha,
        )
    )

    mu = CLASS_MEANS[c]
    Sigma = CLASS_COVS[c]

    I = torch.eye(
        2,
        dtype=torch.get_default_dtype(),
    )

    mean = b * mu

    cov = (
        (a * a) * I
        + (b * b) * Sigma
    )

    mean_dot = b_dot * mu

    K = (
        (a_dot * a) * I
        + (b_dot * b) * Sigma
    )

    return mean, cov, mean_dot, K


# ============================================================
# Conditional marginal velocity
#
# v_t(y | c)
# =
# mean_dot
# +
# K C^{-1} (y - mean)
# ============================================================

def conditional_velocity(y, t, c, alpha):
    mean, cov, mean_dot, K = (
        conditional_marginal(
            t,
            c,
            alpha,
        )
    )

    centered = y - mean

    q = torch.linalg.solve(
        cov,
        centered,
    )

    return mean_dot + K @ q


# ============================================================
# Conditional log density
#
# Used for mixture responsibilities.
# ============================================================

def conditional_log_prob(y, t, c, alpha):
    mean, cov, _, _ = (
        conditional_marginal(
            t,
            c,
            alpha,
        )
    )

    centered = y - mean

    q = torch.linalg.solve(
        cov,
        centered,
    )

    quadratic = centered @ q

    _, logdet = torch.linalg.slogdet(
        cov
    )

    d = y.numel()

    return -0.5 * (
        quadratic
        + logdet
        + d * math.log(2.0 * math.pi)
    )


# ============================================================
# Unconditional mixture velocity
#
# Equal class priors.
# ============================================================

def unconditional_velocity(y, t, alpha):
    log_probs = torch.stack([
        conditional_log_prob(
            y,
            t,
            c,
            alpha,
        )
        for c in range(2)
    ])

    responsibilities = torch.softmax(
        log_probs,
        dim=0,
    )

    velocities = torch.stack([
        conditional_velocity(
            y,
            t,
            c,
            alpha,
        )
        for c in range(2)
    ])

    return torch.sum(
        responsibilities[:, None]
        * velocities,
        dim=0,
    )


# ============================================================
# CFG velocity
# ============================================================

def cfg_velocity(y, t, alpha):
    v_uncond = unconditional_velocity(
        y,
        t,
        alpha,
    )

    v_cond = conditional_velocity(
        y,
        t,
        TARGET_CLASS,
        alpha,
    )

    return (
        v_uncond
        + OMEGA
        * (v_cond - v_uncond)
    )


# ============================================================
# Joint flow + sensitivity dynamics
#
# dx/dt = v_t(x)
#
# dJ/dt = [nabla_x v_t(x)] J
# ============================================================

def velocity_jacobian(x, t, alpha):
    return jacrev(
        lambda y: cfg_velocity(
            y,
            t,
            alpha,
        )
    )(x)


def joint_dynamics(x, J, t, alpha):
    v = cfg_velocity(
        x,
        t,
        alpha,
    )

    A = velocity_jacobian(
        x,
        t,
        alpha,
    )

    J_dot = A @ J

    return v, J_dot


def joint_rk4_step(
    x,
    J,
    t,
    dt,
    alpha,
):
    k1_x, k1_J = joint_dynamics(
        x,
        J,
        t,
        alpha,
    )

    k2_x, k2_J = joint_dynamics(
        x + 0.5 * dt * k1_x,
        J + 0.5 * dt * k1_J,
        t + 0.5 * dt,
        alpha,
    )

    k3_x, k3_J = joint_dynamics(
        x + 0.5 * dt * k2_x,
        J + 0.5 * dt * k2_J,
        t + 0.5 * dt,
        alpha,
    )

    k4_x, k4_J = joint_dynamics(
        x + dt * k3_x,
        J + dt * k3_J,
        t + dt,
        alpha,
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

    return x_new, J_new


def integrate_sample(z, alpha):
    x = z.clone()

    J = torch.eye(
        2,
        dtype=z.dtype,
        device=z.device,
    )

    dt = 1.0 / N_STEPS
    t = 0.0

    for _ in range(N_STEPS):
        x, J = joint_rk4_step(
            x,
            J,
            t,
            dt,
            alpha,
        )

        t += dt

    return x, J


# ============================================================
# 2D Gauss-Hermite quadrature
#
# For z ~ N(0, I):
#
# E[f(z)]
#
# ~= (1/pi) sum_i sum_j
#       w_i w_j
#       f(sqrt(2) x_i, sqrt(2) x_j)
# ============================================================

def gauss_hermite_2d(order):
    nodes, weights = hermgauss(order)

    points = []
    quad_weights = []

    for i in range(order):
        for j in range(order):
            z = np.sqrt(2.0) * np.array([
                nodes[i],
                nodes[j],
            ])

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


# ============================================================
# Population mapping complexity
# ============================================================

def population_complexity(alpha):
    points, weights = gauss_hermite_2d(
        QUADRATURE_ORDER
    )

    complexity = 0.0
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
                f"    point "
                f"{index + 1}/{n_points}"
            )

        z = torch.tensor(
            z_np,
            dtype=torch.get_default_dtype(),
        )

        _, J1 = integrate_sample(
            z,
            alpha,
        )

        complexity_i = torch.sum(
            J1 * J1
        ).item()

        complexity += (
            weight * complexity_i
        )

    return complexity


# ============================================================
# Basic endpoint / path sanity checks
# ============================================================

def sanity_checks():
    print("Sanity checks")
    print("-" * 78)

    for alpha in ALPHAS:
        a0, b0, _, _ = (
            interpolation_coefficients(
                0.0,
                alpha,
            )
        )

        a1, b1, _, _ = (
            interpolation_coefficients(
                1.0,
                alpha,
            )
        )

        print(
            f"alpha={alpha:+.2f}: "
            f"a(0)={a0.item():.6f}, "
            f"b(0)={b0.item():.6f}, "
            f"a(1)={a1.item():.6f}, "
            f"b(1)={b1.item():.6f}"
        )

    # Show explicitly that intermediate marginals differ.
    print()
    print("Conditional covariance at t=0.5, class=1")

    for alpha in ALPHAS:
        _, cov, _, _ = conditional_marginal(
            0.5,
            TARGET_CLASS,
            alpha,
        )

        print(
            f"alpha={alpha:+.2f}: "
            f"diag(cov)="
            f"{torch.diag(cov).tolist()}"
        )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "Exp 5-D — Genuine Interpolation Family"
    )
    print("=" * 78)

    print(f"CFG omega: {OMEGA}")
    print(f"RK4 steps: {N_STEPS}")
    print(
        f"Gauss-Hermite order: "
        f"{QUADRATURE_ORDER}"
    )
    print(
        f"Latent points per alpha: "
        f"{QUADRATURE_ORDER ** 2}"
    )
    print()

    sanity_checks()

    print()
    print("=" * 78)
    print("Population mapping complexity")
    print("=" * 78)

    results = []

    for alpha in ALPHAS:
        print()
        print(
            f"alpha = {alpha:+.2f}"
        )
        print("-" * 78)

        C1 = population_complexity(
            alpha
        )

        results.append({
            "alpha": alpha,
            "C1": C1,
        })

        print()
        print(
            f"C1 = {C1:.12f}"
        )

    # --------------------------------------------------------
    # Find alpha = 0 baseline
    # --------------------------------------------------------

    baseline = next(
        r
        for r in results
        if abs(r["alpha"]) < 1e-12
    )

    C0 = baseline["C1"]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)

    print(
        f"{'alpha':>10}  "
        f"{'C1':>16}  "
        f"{'Delta C1':>16}"
    )

    for r in results:
        delta = r["C1"] - C0

        print(
            f"{r['alpha']:10.2f}  "
            f"{r['C1']:16.12f}  "
            f"{delta:16.12f}"
        )

    print()
    print(
        "Baseline sanity target:"
    )
    print(
        "alpha = 0 should give "
        "C1 ~= 0.77461145"
    )


if __name__ == "__main__":
    main()