import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev

from src.flow import exp02_cfg_flow


torch.set_default_dtype(torch.float64)

SEED = 0
N_SAMPLES = 200
N_STEPS = 200
TARGET_CLASS = 1

OMEGAS = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
)


def guided_map(z, omega):
    return exp02_cfg_flow(
        z,
        t_start=0.0,
        t_end=1.0,
        omega=omega,
        c=TARGET_CLASS,
        n_steps=N_STEPS,
    )


def map_jacobian(z, omega):
    return jacrev(
        lambda z_: guided_map(z_, omega)
    )(z)


def main():
    torch.manual_seed(SEED)

    # Use the same latent sampling protocol as Exp 3.
    z_samples = torch.randn(N_SAMPLES, 2)

    # Reference Jacobians J_f = J_{f^(1)}.
    reference_jacobians = []

    for z in z_samples:
        j_f = map_jacobian(z, 1.0)
        reference_jacobians.append(j_f)

    reference_jacobians = torch.stack(reference_jacobians)

    mean_sigma_sq = []
    mean_weights = []
    mean_contributions = []

    mean_actual_ratio = []
    mean_decomposed_ratio = []
    max_identity_error = []

    all_sigma_sq = []
    all_weights = []
    all_contributions = []
    all_actual_ratio = []
    all_decomposed_ratio = []

    print("Exp 3 Diagnostic — Directional Decomposition")
    print("=" * 70)
    print(f"Number of latent samples: {N_SAMPLES}")
    print(f"RK4 steps: {N_STEPS}")
    print()

    for omega in OMEGAS:
        sigma_sq_values = []
        weight_values = []
        contribution_values = []

        actual_ratio_values = []
        decomposed_ratio_values = []

        for i, z in enumerate(z_samples):
            j_f = reference_jacobians[i]

            # J_{f^(omega)}
            j_f_omega = map_jacobian(
                z,
                float(omega),
            )

            # J_g = J_{f^(omega)} J_f^{-1}
            j_g = torch.linalg.solve(
                j_f.T,
                j_f_omega.T,
            ).T

            # Full SVD:
            # J_g = U diag(S) V^T
            _, singular_values, vh = torch.linalg.svd(
                j_g,
                full_matrices=False,
            )

            # Rows of V^T are the right singular vectors v_i^T.
            #
            # Directional reference energy:
            # ||v_i^T J_f||_2^2
            projected_reference = vh @ j_f

            directional_energy = torch.sum(
                projected_reference**2,
                dim=1,
            )

            reference_frob_sq = torch.sum(j_f**2)

            weights = (
                directional_energy
                / reference_frob_sq
            )

            sigma_sq = singular_values**2

            # Exact normalized complexity decomposition:
            #
            # ||J_g J_f||_F^2 / ||J_f||_F^2
            #     = sum_i w_i sigma_i^2
            decomposed_ratio = torch.sum(
                weights * sigma_sq
            )

            actual_ratio = (
                torch.sum(j_f_omega**2)
                / reference_frob_sq
            )

            # Contribution to complexity change:
            #
            # actual_ratio - 1
            #     = sum_i w_i (sigma_i^2 - 1)
            contributions = (
                weights
                * (sigma_sq - 1.0)
            )

            sigma_sq_values.append(
                sigma_sq.detach().cpu().numpy()
            )

            weight_values.append(
                weights.detach().cpu().numpy()
            )

            contribution_values.append(
                contributions.detach().cpu().numpy()
            )

            actual_ratio_values.append(
                actual_ratio.item()
            )

            decomposed_ratio_values.append(
                decomposed_ratio.item()
            )

        sigma_sq_values = np.asarray(
            sigma_sq_values
        )

        weight_values = np.asarray(
            weight_values
        )

        contribution_values = np.asarray(
            contribution_values
        )

        actual_ratio_values = np.asarray(
            actual_ratio_values
        )

        decomposed_ratio_values = np.asarray(
            decomposed_ratio_values
        )

        identity_error = np.abs(
            actual_ratio_values
            - decomposed_ratio_values
        )

        mean_sigma_sq.append(
            sigma_sq_values.mean(axis=0)
        )

        mean_weights.append(
            weight_values.mean(axis=0)
        )

        mean_contributions.append(
            contribution_values.mean(axis=0)
        )

        mean_actual_ratio.append(
            actual_ratio_values.mean()
        )

        mean_decomposed_ratio.append(
            decomposed_ratio_values.mean()
        )

        max_identity_error.append(
            identity_error.max()
        )

        all_sigma_sq.append(
            sigma_sq_values
        )

        all_weights.append(
            weight_values
        )

        all_contributions.append(
            contribution_values
        )

        all_actual_ratio.append(
            actual_ratio_values
        )

        all_decomposed_ratio.append(
            decomposed_ratio_values
        )

        print(
            f"omega={omega:.2f} | "
            f"sigma^2=({sigma_sq_values[:, 0].mean():.6f}, "
            f"{sigma_sq_values[:, 1].mean():.6f}) | "
            f"weights=({weight_values[:, 0].mean():.6f}, "
            f"{weight_values[:, 1].mean():.6f}) | "
            f"contrib=({contribution_values[:, 0].mean():.6f}, "
            f"{contribution_values[:, 1].mean():.6f}) | "
            f"ratio={actual_ratio_values.mean():.6f} | "
            f"max err={identity_error.max():.3e}"
        )

    mean_sigma_sq = np.asarray(
        mean_sigma_sq
    )

    mean_weights = np.asarray(
        mean_weights
    )

    mean_contributions = np.asarray(
        mean_contributions
    )

    mean_actual_ratio = np.asarray(
        mean_actual_ratio
    )

    mean_decomposed_ratio = np.asarray(
        mean_decomposed_ratio
    )

    max_identity_error = np.asarray(
        max_identity_error
    )

    all_sigma_sq = np.stack(
        all_sigma_sq
    )

    all_weights = np.stack(
        all_weights
    )

    all_contributions = np.stack(
        all_contributions
    )

    all_actual_ratio = np.stack(
        all_actual_ratio
    )

    all_decomposed_ratio = np.stack(
        all_decomposed_ratio
    )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    os.makedirs(
        "results/figures",
        exist_ok=True,
    )

    np.savez(
        "results/exp03_directional_decomposition.npz",
        omegas=OMEGAS,
        mean_sigma_sq=mean_sigma_sq,
        mean_weights=mean_weights,
        mean_contributions=mean_contributions,
        mean_actual_ratio=mean_actual_ratio,
        mean_decomposed_ratio=mean_decomposed_ratio,
        max_identity_error=max_identity_error,
        all_sigma_sq=all_sigma_sq,
        all_weights=all_weights,
        all_contributions=all_contributions,
        all_actual_ratio=all_actual_ratio,
        all_decomposed_ratio=all_decomposed_ratio,
        z_samples=z_samples.detach().cpu().numpy(),
    )

    # Figure 1:
    # squared singular values of J_g
    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_sigma_sq[:, 0],
        marker="o",
        label=r"$\mathbb{E}[\sigma_1^2]$",
    )

    plt.plot(
        OMEGAS,
        mean_sigma_sq[:, 1],
        marker="o",
        label=r"$\mathbb{E}[\sigma_2^2]$",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel("Mean squared singular value")
    plt.title(
        "Exp 3 Diagnostic: "
        "Directional Contraction"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_directional_singular_values.png",
        dpi=200,
    )

    plt.close()

    # Figure 2:
    # reference-complexity weights
    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_weights[:, 0],
        marker="o",
        label=r"$\mathbb{E}[w_1]$",
    )

    plt.plot(
        OMEGAS,
        mean_weights[:, 1],
        marker="o",
        label=r"$\mathbb{E}[w_2]$",
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel("Mean reference-energy weight")
    plt.title(
        "Exp 3 Diagnostic: "
        "Reference Geometry Weights"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_directional_weights.png",
        dpi=200,
    )

    plt.close()

    # Figure 3:
    # directional contributions to complexity change
    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_contributions[:, 0],
        marker="o",
        label=r"$\mathbb{E}[w_1(\sigma_1^2-1)]$",
    )

    plt.plot(
        OMEGAS,
        mean_contributions[:, 1],
        marker="o",
        label=r"$\mathbb{E}[w_2(\sigma_2^2-1)]$",
    )

    plt.axhline(
        0.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel("Mean contribution")
    plt.title(
        "Exp 3 Diagnostic: "
        "Directional Complexity Change"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_directional_contributions.png",
        dpi=200,
    )

    plt.close()

    # Figure 4:
    # exact decomposition sanity check
    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_actual_ratio,
        marker="o",
        label="Actual normalized complexity",
    )

    plt.plot(
        OMEGAS,
        mean_decomposed_ratio,
        marker="x",
        linestyle="--",
        label=r"$\sum_i w_i \sigma_i^2$",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel("Normalized complexity")
    plt.title(
        "Exp 3 Diagnostic: "
        "Exact Directional Decomposition"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_directional_decomposition_check.png",
        dpi=200,
    )

    plt.close()

    print()
    print("Saved:")
    print(
        "  results/"
        "exp03_directional_decomposition.npz"
    )
    print(
        "  results/figures/"
        "exp03_directional_singular_values.png"
    )
    print(
        "  results/figures/"
        "exp03_directional_weights.png"
    )
    print(
        "  results/figures/"
        "exp03_directional_contributions.png"
    )
    print(
        "  results/figures/"
        "exp03_directional_decomposition_check.png"
    )


if __name__ == "__main__":
    main()