import os
import math

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# Configuration
# ============================================================

torch.set_default_dtype(torch.float64)

N_STEPS = 1000

BETAS = [
    0.0,
    math.pi / 12.0,   # 15 degrees
    math.pi / 6.0,    # 30 degrees
    math.pi / 4.0,    # 45 degrees
    math.pi / 3.0,    # 60 degrees
]

RESULTS_DIR = "results"
FIGURE_DIR = "results/figures"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Basic matrices
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
    dR/dtheta
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    return torch.stack([
        torch.stack([-s, -c]),
        torch.stack([ c, -s]),
    ])


# ============================================================
# Covariance interpolation
#
# eigenvalues:
#   a(t) = 4^t
#   b(t) = 4^{-t}
#
# orientation:
#   theta(t) = beta sin(pi t)
#
# Hence:
#   Sigma(0) = I
#   Sigma(1) = diag(4, 1/4)
# for every beta.
# ============================================================

def covariance_and_derivative(t, beta):
    t = torch.as_tensor(
        t,
        dtype=torch.get_default_dtype(),
    )

    beta = torch.as_tensor(
        beta,
        dtype=torch.get_default_dtype(),
    )

    log4 = math.log(4.0)

    # Eigenvalues
    a = torch.exp(log4 * t)
    b = torch.exp(-log4 * t)

    da = log4 * a
    db = -log4 * b

    D = torch.diag(
        torch.stack([a, b])
    )

    D_dot = torch.diag(
        torch.stack([da, db])
    )

    # Rotation schedule
    theta = beta * torch.sin(math.pi * t)

    theta_dot = (
        beta
        * math.pi
        * torch.cos(math.pi * t)
    )

    R = rotation(theta)
    R_theta = rotation_derivative(theta)

    R_dot = theta_dot * R_theta

    # Sigma = R D R^T
    Sigma = R @ D @ R.T

    # d/dt (R D R^T)
    Sigma_dot = (
        R_dot @ D @ R.T
        + R @ D_dot @ R.T
        + R @ D @ R_dot.T
    )

    return Sigma, Sigma_dot


# ============================================================
# Canonical symmetric velocity Jacobian
#
# We require:
#
#   Sigma_dot = S Sigma + Sigma S
#
# with S symmetric.
#
# In the eigenbasis of Sigma:
#
#   S_ij = Sigma_dot_ij / (lambda_i + lambda_j)
#
# ============================================================

def symmetric_velocity_jacobian(t, beta):
    Sigma, Sigma_dot = covariance_and_derivative(
        t,
        beta,
    )

    eigvals, U = torch.linalg.eigh(Sigma)

    Sigma_dot_eig = U.T @ Sigma_dot @ U

    denom = (
        eigvals[:, None]
        + eigvals[None, :]
    )

    S_eig = Sigma_dot_eig / denom

    S = U @ S_eig @ U.T

    # Numerical symmetrization
    S = 0.5 * (S + S.T)

    return S


# ============================================================
# Jacobian dynamics
#
#   dJ/dt = S_t J
#   J(0) = I
#
# ============================================================

def jacobian_rhs(J, t, beta):
    S = symmetric_velocity_jacobian(
        t,
        beta,
    )

    return S @ J


def rk4_step(J, t, dt, beta):
    k1 = jacobian_rhs(
        J,
        t,
        beta,
    )

    k2 = jacobian_rhs(
        J + 0.5 * dt * k1,
        t + 0.5 * dt,
        beta,
    )

    k3 = jacobian_rhs(
        J + 0.5 * dt * k2,
        t + 0.5 * dt,
        beta,
    )

    k4 = jacobian_rhs(
        J + dt * k3,
        t + dt,
        beta,
    )

    return J + (dt / 6.0) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


def integrate_jacobian(beta):
    dt = 1.0 / N_STEPS

    J = torch.eye(
        2,
        dtype=torch.get_default_dtype(),
    )

    for step in range(N_STEPS):
        t = step * dt

        J = rk4_step(
            J,
            t,
            dt,
            beta,
        )

    return J


# ============================================================
# Main
# ============================================================

def main():
    print("Exp 5-A — Rotating Gaussian Interpolation")
    print("=" * 65)

    print(f"RK4 steps: {N_STEPS}")
    print()

    target_covariance = torch.tensor(
        [
            [4.0, 0.0],
            [0.0, 0.25],
        ],
        dtype=torch.get_default_dtype(),
    )

    beta_values = []
    beta_degrees = []

    complexities = []
    covariance_errors = []
    determinant_values = []

    final_jacobians = []

    for beta in BETAS:
        beta_deg = beta * 180.0 / math.pi

        J1 = integrate_jacobian(beta)

        M1 = J1 @ J1.T

        complexity = torch.trace(M1).item()

        covariance_error = torch.linalg.norm(
            M1 - target_covariance,
            ord="fro",
        ).item()

        determinant = torch.det(J1).item()

        beta_values.append(beta)
        beta_degrees.append(beta_deg)

        complexities.append(complexity)
        covariance_errors.append(covariance_error)
        determinant_values.append(determinant)

        final_jacobians.append(
            J1.detach().cpu().numpy()
        )

        print("-" * 65)
        print(
            f"beta = {beta:.6f} rad "
            f"({beta_deg:.1f} deg)"
        )

        print()
        print("J_1 =")
        print(
            J1.detach().cpu().numpy()
        )

        print()
        print("J_1 J_1^T =")
        print(
            M1.detach().cpu().numpy()
        )

        print()
        print(
            f"C_1 = ||J_1||_F^2 "
            f"= {complexity:.12f}"
        )

        print(
            "endpoint covariance error "
            f"= {covariance_error:.3e}"
        )

        print(
            f"det(J_1) = "
            f"{determinant:.12f}"
        )

        print()

    beta_values = np.asarray(beta_values)
    beta_degrees = np.asarray(beta_degrees)

    complexities = np.asarray(complexities)
    covariance_errors = np.asarray(
        covariance_errors
    )
    determinant_values = np.asarray(
        determinant_values
    )

    final_jacobians = np.asarray(
        final_jacobians
    )

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 65)
    print("Summary")
    print("=" * 65)

    for beta_deg, complexity, error in zip(
        beta_degrees,
        complexities,
        covariance_errors,
    ):
        print(
            f"beta={beta_deg:5.1f} deg  "
            f"C1={complexity:.12f}  "
            f"cov_error={error:.3e}"
        )

    complexity_range = (
        np.max(complexities)
        - np.min(complexities)
    )

    print()
    print(
        "Complexity range across paths:"
    )
    print(
        f"max(C1) - min(C1) "
        f"= {complexity_range:.12e}"
    )

    print()
    print(
        "Maximum endpoint covariance error:"
    )
    print(
        f"{np.max(covariance_errors):.3e}"
    )

    # ========================================================
    # Save results
    # ========================================================

    output_path = os.path.join(
        RESULTS_DIR,
        "exp05a_rotating_gaussian.npz",
    )

    np.savez(
        output_path,
        beta=beta_values,
        beta_degrees=beta_degrees,
        complexity=complexities,
        covariance_error=covariance_errors,
        determinant=determinant_values,
        final_jacobian=final_jacobians,
    )

    # ========================================================
    # Figure
    # ========================================================

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
        r"Maximum rotation amplitude $\beta$ (degrees)"
    )

    plt.ylabel(
        r"Final mapping complexity "
        r"$\|J_1\|_F^2$"
    )

    plt.title(
        "Exp 5-A: Same Gaussian Endpoints, "
        "Different Interpolation Geometry"
    )

    plt.tight_layout()

    figure_path = os.path.join(
        FIGURE_DIR,
        "exp05a_rotating_gaussian_complexity.png",
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