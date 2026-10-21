import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev

from experiments.exp04_decoupling import (
    N_SAMPLES,
    N_STEPS,
    OMEGAS,
    SEED,
    guided_map,
)


torch.set_default_dtype(torch.float64)


def map_jacobian(z, omega):
    return jacrev(
        lambda z_: guided_map(
            z_,
            omega,
        )
    )(z)


def main():
    torch.manual_seed(SEED)

    # Exactly the same latent sampling protocol as Phase A.
    z_samples = torch.randn(
        N_SAMPLES,
        2,
    )

    # Reference Jacobians J_f = J_{f^(1)}.
    reference_jacobians = []

    for z in z_samples:
        j_f = map_jacobian(
            z,
            1.0,
        )
        reference_jacobians.append(
            j_f
        )

    reference_jacobians = torch.stack(
        reference_jacobians
    )

    print(
        "Exp 4 Phase B — "
        "Directional Diagnostic of Decoupled Samples"
    )
    print("=" * 76)
    print(
        f"Number of latent samples: {N_SAMPLES}"
    )
    print(
        f"RK4 steps: {N_STEPS}"
    )
    print()

    summary_rows = []

    all_records = {}

    for omega in OMEGAS:
        if omega <= 1.0:
            continue

        records = []

        for i, z in enumerate(z_samples):
            j_f = reference_jacobians[i]

            j_f_omega = map_jacobian(
                z,
                float(omega),
            )

            # J_f^(omega) = J_g J_f
            # => J_g = J_f^(omega) J_f^{-1}
            j_g = torch.linalg.solve(
                j_f.T,
                j_f_omega.T,
            ).T

            # J_g = U diag(sigma) V^T
            _, singular_values, vh = torch.linalg.svd(
                j_g,
                full_matrices=False,
            )

            sigma_sq = singular_values**2

            reference_frob_sq = torch.sum(
                j_f**2
            )

            # Rows of V^T are v_i^T.
            projected_reference = vh @ j_f

            directional_energy = torch.sum(
                projected_reference**2,
                dim=1,
            )

            weights = (
                directional_energy
                / reference_frob_sq
            )

            contributions = (
                weights
                * (sigma_sq - 1.0)
            )

            local_ratio = (
                torch.sum(j_f_omega**2)
                / reference_frob_sq
            )

            g_op = singular_values[0]

            # Exact decomposition check.
            decomposed_ratio = torch.sum(
                weights * sigma_sq
            )

            decomposition_error = torch.abs(
                local_ratio
                - decomposed_ratio
            )

            is_decoupled = bool(
                (g_op > 1.0)
                and (local_ratio < 1.0)
            )

            records.append(
                {
                    "sample_index": i,
                    "g_op": g_op.item(),
                    "local_ratio": local_ratio.item(),
                    "sigma_sq_1": sigma_sq[0].item(),
                    "sigma_sq_2": sigma_sq[1].item(),
                    "weight_1": weights[0].item(),
                    "weight_2": weights[1].item(),
                    "contrib_1": contributions[0].item(),
                    "contrib_2": contributions[1].item(),
                    "decomposition_error":
                        decomposition_error.item(),
                    "is_decoupled": is_decoupled,
                }
            )

        all_records[float(omega)] = records

        decoupled_records = [
            r for r in records
            if r["is_decoupled"]
        ]

        n_decoupled = len(
            decoupled_records
        )

        fraction_decoupled = (
            n_decoupled / N_SAMPLES
        )

        max_error = max(
            r["decomposition_error"]
            for r in records
        )

        print(
            f"omega={omega:.2f} | "
            f"decoupled={n_decoupled}/{N_SAMPLES} "
            f"({fraction_decoupled:.3f}) | "
            f"max decomposition err="
            f"{max_error:.3e}"
        )

        if n_decoupled == 0:
            print()
            continue

        sigma_sq_1 = np.array(
            [
                r["sigma_sq_1"]
                for r in decoupled_records
            ]
        )

        sigma_sq_2 = np.array(
            [
                r["sigma_sq_2"]
                for r in decoupled_records
            ]
        )

        weight_1 = np.array(
            [
                r["weight_1"]
                for r in decoupled_records
            ]
        )

        weight_2 = np.array(
            [
                r["weight_2"]
                for r in decoupled_records
            ]
        )

        contrib_1 = np.array(
            [
                r["contrib_1"]
                for r in decoupled_records
            ]
        )

        contrib_2 = np.array(
            [
                r["contrib_2"]
                for r in decoupled_records
            ]
        )

        local_ratio = np.array(
            [
                r["local_ratio"]
                for r in decoupled_records
            ]
        )

        # Since torch.linalg.svd sorts singular values descending,
        # direction 1 is the expanding / least-contracted direction
        # for these decoupled samples.
        print(
            "  decoupled means:"
        )

        print(
            f"    sigma^2 = "
            f"({sigma_sq_1.mean():.6f}, "
            f"{sigma_sq_2.mean():.6f})"
        )

        print(
            f"    weights = "
            f"({weight_1.mean():.6f}, "
            f"{weight_2.mean():.6f})"
        )

        print(
            f"    contributions = "
            f"({contrib_1.mean():.6f}, "
            f"{contrib_2.mean():.6f})"
        )

        print(
            f"    local complexity ratio = "
            f"{local_ratio.mean():.6f}"
        )

        # For the decoupled set, direction 1 must have
        # sigma_1 > 1 because sigma_1 is the largest singular value.
        #
        # Test whether its positive expansion cost is outweighed
        # by contraction in direction 2.
        expansion_cost = np.maximum(
            contrib_1,
            0.0,
        )

        contraction_benefit = np.maximum(
            -contrib_2,
            0.0,
        )

        mechanism_mask = (
            (contrib_1 > 0.0)
            & (contrib_2 < 0.0)
            & (
                contraction_benefit
                > expansion_cost
            )
        )

        mechanism_fraction = (
            mechanism_mask.mean()
        )

        print(
            "    low-weight-expansion / "
            "high-weight-contraction pattern: "
            f"{mechanism_fraction:.3f}"
        )

        print()

        summary_rows.append(
            [
                omega,
                n_decoupled,
                fraction_decoupled,
                sigma_sq_1.mean(),
                sigma_sq_2.mean(),
                weight_1.mean(),
                weight_2.mean(),
                contrib_1.mean(),
                contrib_2.mean(),
                local_ratio.mean(),
                mechanism_fraction,
            ]
        )

    # Convert all records to arrays for saving.
    omega_values = []
    sample_indices = []
    g_ops = []
    local_ratios = []
    sigma_sq_values = []
    weight_values = []
    contribution_values = []
    decoupled_flags = []
    decomposition_errors = []

    for omega, records in all_records.items():
        for r in records:
            omega_values.append(
                omega
            )

            sample_indices.append(
                r["sample_index"]
            )

            g_ops.append(
                r["g_op"]
            )

            local_ratios.append(
                r["local_ratio"]
            )

            sigma_sq_values.append(
                [
                    r["sigma_sq_1"],
                    r["sigma_sq_2"],
                ]
            )

            weight_values.append(
                [
                    r["weight_1"],
                    r["weight_2"],
                ]
            )

            contribution_values.append(
                [
                    r["contrib_1"],
                    r["contrib_2"],
                ]
            )

            decoupled_flags.append(
                r["is_decoupled"]
            )

            decomposition_errors.append(
                r["decomposition_error"]
            )

    omega_values = np.asarray(
        omega_values
    )

    sample_indices = np.asarray(
        sample_indices
    )

    g_ops = np.asarray(
        g_ops
    )

    local_ratios = np.asarray(
        local_ratios
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

    decoupled_flags = np.asarray(
        decoupled_flags
    )

    decomposition_errors = np.asarray(
        decomposition_errors
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
        "results/"
        "exp04_directional_diagnostic.npz",
        omega=omega_values,
        sample_index=sample_indices,
        g_op=g_ops,
        local_ratio=local_ratios,
        sigma_sq=sigma_sq_values,
        weights=weight_values,
        contributions=contribution_values,
        is_decoupled=decoupled_flags,
        decomposition_error=decomposition_errors,
        z_samples=z_samples.detach().cpu().numpy(),
    )

    # Figure 1:
    # Every omega > 1 sample.
    # Highlight decoupled samples.
    plt.figure(
        figsize=(7, 5)
    )

    regular = ~decoupled_flags

    plt.scatter(
        g_ops[regular],
        local_ratios[regular],
        alpha=0.35,
        label="Other samples",
    )

    plt.scatter(
        g_ops[decoupled_flags],
        local_ratios[decoupled_flags],
        marker="x",
        s=60,
        label="Decoupled samples",
    )

    plt.axvline(
        1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.axhline(
        1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.xlabel(
        r"$\|J_g\|_{\mathrm{op}}$"
    )

    plt.ylabel(
        "Local normalized mapping complexity"
    )

    plt.title(
        "Exp 4: Local Contraction–Complexity "
        "Decoupling"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp04_local_decoupling_scatter.png",
        dpi=200,
    )

    plt.close()

    # Figure 2:
    # Mean directional contributions among
    # decoupled samples, by omega.
    if len(summary_rows) > 0:
        summary = np.asarray(
            summary_rows,
            dtype=float,
        )

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            summary[:, 0],
            summary[:, 7],
            marker="o",
            label="Direction 1 contribution",
        )

        plt.plot(
            summary[:, 0],
            summary[:, 8],
            marker="o",
            label="Direction 2 contribution",
        )

        plt.axhline(
            0.0,
            linestyle="--",
            alpha=0.6,
        )

        plt.xlabel(
            r"CFG scale $\omega$"
        )

        plt.ylabel(
            "Mean contribution "
            "among decoupled samples"
        )

        plt.title(
            "Exp 4: Directional Complexity "
            "Contributions"
        )

        plt.grid(
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            "results/figures/"
            "exp04_decoupled_contributions.png",
            dpi=200,
        )

        plt.close()

    print(
        "Saved:"
    )

    print(
        "  results/"
        "exp04_directional_diagnostic.npz"
    )

    print(
        "  results/figures/"
        "exp04_local_decoupling_scatter.png"
    )

    if len(summary_rows) > 0:
        print(
            "  results/figures/"
            "exp04_decoupled_contributions.png"
        )


if __name__ == "__main__":
    main()