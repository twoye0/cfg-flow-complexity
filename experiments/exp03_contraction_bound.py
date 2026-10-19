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

    # Same latent samples for every omega.
    z_samples = torch.randn(N_SAMPLES, 2)

    # Reference Jacobians J_f = J_{f^(1)}
    reference_jacobians = []

    for z in z_samples:
        j_f = map_jacobian(z, 1.0)
        reference_jacobians.append(j_f)

    reference_jacobians = torch.stack(reference_jacobians)

    mean_actual = []
    mean_bound = []
    mean_tightness = []
    max_violation = []

    all_actual = []
    all_bound = []
    all_tightness = []
    all_g_opnorm = []

    print("Exp 3 Diagnostic — Contraction Bound")
    print("=" * 60)
    print(f"Number of latent samples: {N_SAMPLES}")
    print(f"RK4 steps: {N_STEPS}")
    print()

    for omega in OMEGAS:
        actual_values = []
        bound_values = []
        tightness_values = []
        g_opnorm_values = []

        for i, z in enumerate(z_samples):
            j_f = reference_jacobians[i]

            # J_{f^(omega)}
            j_f_omega = map_jacobian(
                z,
                float(omega),
            )

            # From:
            # J_{f^(omega)} = J_g J_f
            #
            # therefore:
            # J_g = J_{f^(omega)} J_f^{-1}
            #
            # Use solve instead of explicitly forming inverse:
            # J_g^T = solve(J_f^T, J_f_omega^T)
            j_g = torch.linalg.solve(
                j_f.T,
                j_f_omega.T,
            ).T

            # Actual mapping-complexity contribution
            actual = torch.sum(j_f_omega**2)

            # ||J_g||_op
            singular_values = torch.linalg.svdvals(j_g)
            g_opnorm = singular_values.max()

            # Contraction upper bound
            reference_frob_sq = torch.sum(j_f**2)

            bound = (
                g_opnorm**2
                * reference_frob_sq
            )

            tightness = actual / bound

            actual_values.append(actual.item())
            bound_values.append(bound.item())
            tightness_values.append(tightness.item())
            g_opnorm_values.append(g_opnorm.item())

        actual_values = np.asarray(actual_values)
        bound_values = np.asarray(bound_values)
        tightness_values = np.asarray(tightness_values)
        g_opnorm_values = np.asarray(g_opnorm_values)

        # Numerical inequality check:
        # actual <= bound
        violation = actual_values - bound_values

        mean_actual.append(actual_values.mean())
        mean_bound.append(bound_values.mean())
        mean_tightness.append(tightness_values.mean())
        max_violation.append(violation.max())

        all_actual.append(actual_values)
        all_bound.append(bound_values)
        all_tightness.append(tightness_values)
        all_g_opnorm.append(g_opnorm_values)

        print(
            f"omega={omega:.2f} | "
            f"actual={actual_values.mean():.6f} | "
            f"bound={bound_values.mean():.6f} | "
            f"tightness={tightness_values.mean():.6f} | "
            f"g_op={g_opnorm_values.mean():.6f} | "
            f"max(actual-bound)={violation.max():.3e}"
        )

    mean_actual = np.asarray(mean_actual)
    mean_bound = np.asarray(mean_bound)
    mean_tightness = np.asarray(mean_tightness)
    max_violation = np.asarray(max_violation)

    all_actual = np.stack(all_actual)
    all_bound = np.stack(all_bound)
    all_tightness = np.stack(all_tightness)
    all_g_opnorm = np.stack(all_g_opnorm)

    # ---------------------------------------------------------
    # Save numerical results
    # ---------------------------------------------------------

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    np.savez(
        "results/exp03_contraction_bound.npz",
        omegas=OMEGAS,
        mean_actual=mean_actual,
        mean_bound=mean_bound,
        mean_tightness=mean_tightness,
        max_violation=max_violation,
        all_actual=all_actual,
        all_bound=all_bound,
        all_tightness=all_tightness,
        all_g_opnorm=all_g_opnorm,
        z_samples=z_samples.detach().cpu().numpy(),
    )

    # ---------------------------------------------------------
    # Figure 1: actual complexity vs contraction bound
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_actual,
        marker="o",
        label="Actual mapping complexity",
    )

    plt.plot(
        OMEGAS,
        mean_bound,
        marker="o",
        label="Contraction upper bound",
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel("Mean sample-wise quantity")
    plt.title("Exp 3 Diagnostic: Contraction Bound")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_actual_vs_contraction_bound.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Figure 2: bound tightness
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_tightness,
        marker="o",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        alpha=0.6,
        label="Tight bound",
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel(r"Mean actual / bound")
    plt.title("Exp 3 Diagnostic: Bound Tightness")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_contraction_bound_tightness.png",
        dpi=200,
    )

    plt.close()

    print()
    print("Saved:")
    print("  results/exp03_contraction_bound.npz")
    print(
        "  results/figures/"
        "exp03_actual_vs_contraction_bound.png"
    )
    print(
        "  results/figures/"
        "exp03_contraction_bound_tightness.png"
    )


if __name__ == "__main__":
    main()