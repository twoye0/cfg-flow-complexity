import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev

from src.distributions import sample_base


torch.set_default_dtype(torch.float64)

# ============================================================
# Configuration
# ============================================================

SEED = 0
N_SAMPLES = 200
N_STEPS = 200

TARGET_CLASS = 1
OMEGA = 2.0

BETAS = [
    0.0,
    math.pi / 12.0,   # 15 deg
    math.pi / 6.0,    # 30 deg
    math.pi / 4.0,    # 45 deg
    math.pi / 3.0,    # 60 deg
]

RESULTS_DIR = "results"
FIGURE_DIR = "results/figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


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
# Rotation
# ============================================================

def rotation(theta):
    c = torch.cos(theta)
    s = torch.sin(theta)

    return torch.stack([
        torch.stack([c, -s]),
        torch.stack([s,  c]),
    ])


def rotation_derivative(theta):
    """
    dR(theta) / dtheta
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    return torch.stack([
        torch.stack([-s, -c]),
        torch.stack([ c, -s]),
    ])


# ============================================================
# Matrix interpolation
#
# x_t = A_t z + B_t x
#
# A_t = (1-t) R(theta_beta(t))
# B_t = t I
#
# theta_beta(t) = beta sin(pi t)
#
# Important:
#
# A_t A_t^T = (1-t)^2 I
#
# Therefore the class-conditional marginals are exactly the
# same as in Exp2 for every beta:
#
# mean = t mu_c
# cov  = (1-t)^2 I + t^2 Sigma_c
#
# Only the transport coupling / velocity field changes.
# ============================================================

def interpolation_matrices(t, beta):
    t = torch.as_tensor(
        t,
        dtype=torch.get_default_dtype(),
    )

    beta = torch.as_tensor(
        beta,
        dtype=torch.get_default_dtype(),
    )

    theta = beta * torch.sin(math.pi * t)

    theta_dot = (
        beta
        * math.pi
        * torch.cos(math.pi * t)
    )

    R = rotation(theta)
    R_theta = rotation_derivative(theta)

    R_dot = theta_dot * R_theta

    I = torch.eye(
        2,
        dtype=torch.get_default_dtype(),
    )

    A = (1.0 - t) * R
    B = t * I

    A_dot = -R + (1.0 - t) * R_dot
    B_dot = I

    return A, B, A_dot, B_dot


# ============================================================
# Conditional marginal quantities
# ============================================================

def conditional_marginal(t, c, beta):
    A, B, A_dot, B_dot = interpolation_matrices(
        t,
        beta,
    )

    mu = CLASS_MEANS[c]
    Sigma = CLASS_COVS[c]

    mean = B @ mu

    cov = (
        A @ A.T
        + B @ Sigma @ B.T
    )

    mean_dot = B_dot @ mu

    # Cross-covariance:
    #
    # Cov(dot{x_t}, x_t)
    #
    K = (
        A_dot @ A.T
        + B_dot @ Sigma @ B.T
    )

    return mean, cov, mean_dot, K


# ============================================================
# Conditional velocity
#
# v_t(y | c)
# =
# mean_dot
# +
# Cov(dot{x_t}, x_t)
# Cov(x_t)^{-1}
# (y - mean)
# ============================================================

def conditional_velocity(y, t, c, beta):
    mean, cov, mean_dot, K = conditional_marginal(
        t,
        c,
        beta,
    )

    centered = y - mean

    # Solve cov * q = centered instead of explicitly
    # constructing cov^{-1}.
    q = torch.linalg.solve(
        cov,
        centered,
    )

    return mean_dot + K @ q


# ============================================================
# Class log density
#
# Used only for mixture responsibilities.
#
# Since the marginals are beta-independent, these
# responsibilities should also be beta-independent.
# ============================================================

def conditional_log_prob(y, t, c, beta):
    mean, cov, _, _ = conditional_marginal(
        t,
        c,
        beta,
    )

    centered = y - mean

    q = torch.linalg.solve(
        cov,
        centered,
    )

    quadratic = centered @ q

    _, logdet = torch.linalg.slogdet(cov)

    d = y.numel()

    return -0.5 * (
        quadratic
        + logdet
        + d * math.log(2.0 * math.pi)
    )


# ============================================================
# Unconditional mixture velocity
#
# Equal class prior: p(c=0) = p(c=1) = 1/2.
# ============================================================

def unconditional_velocity(y, t, beta):
    log_probs = torch.stack([
        conditional_log_prob(
            y,
            t,
            c,
            beta,
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
            beta,
        )
        for c in range(2)
    ])

    return torch.sum(
        responsibilities[:, None] * velocities,
        dim=0,
    )


# ============================================================
# CFG velocity
# ============================================================

def cfg_velocity(y, t, beta):
    v_uncond = unconditional_velocity(
        y,
        t,
        beta,
    )

    v_cond = conditional_velocity(
        y,
        t,
        TARGET_CLASS,
        beta,
    )

    return (
        v_uncond
        + OMEGA * (v_cond - v_uncond)
    )


# ============================================================
# Joint dynamics
#
# dx/dt = v_t(x)
#
# dJ/dt = [nabla_x v_t(x)] J
#
# This directly integrates the flow and its sensitivity.
# ============================================================

def velocity_jacobian(x, t, beta):
    return jacrev(
        lambda y: cfg_velocity(
            y,
            t,
            beta,
        )
    )(x)


def joint_dynamics(x, J, t, beta):
    v = cfg_velocity(
        x,
        t,
        beta,
    )

    A = velocity_jacobian(
        x,
        t,
        beta,
    )

    J_dot = A @ J

    return v, J_dot


def joint_rk4_step(x, J, t, dt, beta):
    k1_x, k1_J = joint_dynamics(
        x,
        J,
        t,
        beta,
    )

    k2_x, k2_J = joint_dynamics(
        x + 0.5 * dt * k1_x,
        J + 0.5 * dt * k1_J,
        t + 0.5 * dt,
        beta,
    )

    k3_x, k3_J = joint_dynamics(
        x + 0.5 * dt * k2_x,
        J + 0.5 * dt * k2_J,
        t + 0.5 * dt,
        beta,
    )

    k4_x, k4_J = joint_dynamics(
        x + dt * k3_x,
        J + dt * k3_J,
        t + dt,
        beta,
    )

    x_next = x + (dt / 6.0) * (
        k1_x
        + 2.0 * k2_x
        + 2.0 * k3_x
        + k4_x
    )

    J_next = J + (dt / 6.0) * (
        k1_J
        + 2.0 * k2_J
        + 2.0 * k3_J
        + k4_J
    )

    return x_next, J_next


def integrate_sample(z, beta):
    dt = 1.0 / N_STEPS

    x = z.clone()

    J = torch.eye(
        2,
        dtype=torch.get_default_dtype(),
    )

    for step in range(N_STEPS):
        t = step * dt

        x, J = joint_rk4_step(
            x,
            J,
            t,
            dt,
            beta,
        )

    return x, J


# ============================================================
# Sanity check:
# marginals should be beta-independent
# ============================================================

def marginal_invariance_check():
    test_times = [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]

    baseline_beta = 0.0

    max_mean_error = 0.0
    max_cov_error = 0.0

    for beta in BETAS:
        for t in test_times:
            for c in range(2):
                mean_0, cov_0, _, _ = conditional_marginal(
                    t,
                    c,
                    baseline_beta,
                )

                mean_b, cov_b, _, _ = conditional_marginal(
                    t,
                    c,
                    beta,
                )

                mean_error = torch.linalg.norm(
                    mean_b - mean_0
                ).item()

                cov_error = torch.linalg.norm(
                    cov_b - cov_0,
                    ord="fro",
                ).item()

                max_mean_error = max(
                    max_mean_error,
                    mean_error,
                )

                max_cov_error = max(
                    max_cov_error,
                    cov_error,
                )

    return max_mean_error, max_cov_error


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED)

    print(
        "Exp 5-B — Same Marginals, "
        "Different Velocity Couplings"
    )
    print("=" * 70)

    print(f"Samples: {N_SAMPLES}")
    print(f"RK4 steps: {N_STEPS}")
    print(f"Target class: {TARGET_CLASS}")
    print(f"CFG omega: {OMEGA}")
    print()

    # --------------------------------------------------------
    # Marginal invariance sanity check
    # --------------------------------------------------------

    mean_error, cov_error = marginal_invariance_check()

    print("Marginal invariance sanity check")
    print(
        "max mean difference across beta "
        f"= {mean_error:.3e}"
    )
    print(
        "max covariance difference across beta "
        f"= {cov_error:.3e}"
    )
    print()

    # --------------------------------------------------------
    # Use exactly the same base samples for every beta.
    # --------------------------------------------------------

    z_samples = sample_base(N_SAMPLES)

    beta_values = []
    beta_degrees = []

    complexities = []
    complexity_stds = []

    mean_det_j = []
    min_det_j = []

    all_sample_complexities = []

    for beta_index, beta in enumerate(BETAS):
        beta_deg = beta * 180.0 / math.pi

        print("-" * 70)
        print(
            f"beta = {beta:.6f} rad "
            f"({beta_deg:.1f} deg)"
        )

        sample_complexities = []
        sample_determinants = []

        for i, z in enumerate(z_samples):
            if (
                i == 0
                or (i + 1) % 50 == 0
                or i + 1 == N_SAMPLES
            ):
                print(
                    f"Processing sample "
                    f"{i + 1}/{N_SAMPLES}"
                )

            _, J1 = integrate_sample(
                z,
                beta,
            )

            complexity_i = torch.sum(
                J1 * J1
            ).item()

            determinant_i = torch.det(
                J1
            ).item()

            sample_complexities.append(
                complexity_i
            )

            sample_determinants.append(
                determinant_i
            )

        sample_complexities = np.asarray(
            sample_complexities
        )

        sample_determinants = np.asarray(
            sample_determinants
        )

        complexity = np.mean(
            sample_complexities
        )

        complexity_std = np.std(
            sample_complexities
        )

        beta_values.append(beta)
        beta_degrees.append(beta_deg)

        complexities.append(complexity)
        complexity_stds.append(
            complexity_std
        )

        mean_det_j.append(
            np.mean(sample_determinants)
        )

        min_det_j.append(
            np.min(sample_determinants)
        )

        all_sample_complexities.append(
            sample_complexities
        )

        print()
        print(
            "Mapping complexity "
            f"E[||J_1||_F^2] = "
            f"{complexity:.9f}"
        )

        print(
            "Std across samples "
            f"= {complexity_std:.9f}"
        )

        print(
            "Mean det(J_1) "
            f"= {np.mean(sample_determinants):.9f}"
        )

        print(
            "Min det(J_1) "
            f"= {np.min(sample_determinants):.9f}"
        )

        print()

    beta_values = np.asarray(beta_values)
    beta_degrees = np.asarray(beta_degrees)

    complexities = np.asarray(complexities)
    complexity_stds = np.asarray(
        complexity_stds
    )

    mean_det_j = np.asarray(mean_det_j)
    min_det_j = np.asarray(min_det_j)

    all_sample_complexities = np.asarray(
        all_sample_complexities
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    baseline_complexity = complexities[0]

    for beta_deg, complexity in zip(
        beta_degrees,
        complexities,
    ):
        delta = (
            complexity
            - baseline_complexity
        )

        print(
            f"beta={beta_deg:5.1f} deg  "
            f"C1={complexity:.9f}  "
            f"delta_from_beta0={delta:+.9f}"
        )

    print()
    print(
        "Complexity range across couplings:"
    )

    print(
        f"max(C1) - min(C1) "
        f"= "
        f"{np.max(complexities) - np.min(complexities):.9f}"
    )

    print()
    print(
        "Baseline reproduction check:"
    )

    print(
        "Expected previous omega=2 result "
        "~ 0.761131"
    )

    print(
        f"Current beta=0 result "
        f"= {complexities[0]:.9f}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        RESULTS_DIR,
        "exp05b_velocity_coupling.npz",
    )

    np.savez(
        output_path,
        beta=beta_values,
        beta_degrees=beta_degrees,
        complexity=complexities,
        complexity_std=complexity_stds,
        sample_complexity=all_sample_complexities,
        mean_det_j=mean_det_j,
        min_det_j=min_det_j,
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        beta_degrees,
        complexities,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        r"Maximum coupling rotation "
        r"$\beta$ (degrees)"
    )

    plt.ylabel(
        r"Mapping complexity "
        r"$\mathbb{E}\|J_1\|_F^2$"
    )

    plt.title(
        "Exp 5-B: Same Marginals, "
        "Different Velocity Couplings"
    )

    plt.tight_layout()

    figure_path = os.path.join(
        FIGURE_DIR,
        "exp05b_velocity_coupling_complexity.png",
    )

    plt.savefig(
        figure_path,
        dpi=200,
    )

    plt.close()

    print()
    print("Saved raw results to:")
    print(output_path)

    print()
    print("Saved figure to:")
    print(figure_path)


if __name__ == "__main__":
    main()