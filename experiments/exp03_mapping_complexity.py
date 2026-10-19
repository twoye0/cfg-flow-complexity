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
    """
    f^{(omega)} : z -> x

    Construct the guided flow map directly by integrating
    the Exp 2 CFG velocity field from t=0 to t=1.
    """
    return exp02_cfg_flow(
        z,
        t_start=0.0,
        t_end=1.0,
        omega=omega,
        c=TARGET_CLASS,
        n_steps=N_STEPS,
    )


def map_jacobian(z, omega):
    """
    Jacobian of f^{(omega)} with respect to the latent input z.
    """
    return jacrev(lambda z_: guided_map(z_, omega))(z)


def sample_mapping_complexity(z, omega):
    """
    Sample-wise mapping complexity:

        || J_{f^{(omega)}}(z) ||_F^2
    """
    jac = map_jacobian(z, omega)
    return torch.sum(jac**2)


def main():
    torch.manual_seed(SEED)

    # Use the same latent samples for every omega.
    z_samples = torch.randn(N_SAMPLES, 2)

    mean_complexities = []
    median_complexities = []
    std_complexities = []

    all_complexities = []

    print("Toy Exp 3 — Mapping Complexity over the CFG Family")
    print("=" * 60)
    print(f"Number of latent samples: {N_SAMPLES}")
    print(f"RK4 steps: {N_STEPS}")
    print()

    for omega in OMEGAS:
        sample_values = []

        for z in z_samples:
            complexity = sample_mapping_complexity(
                z,
                float(omega),
            )
            sample_values.append(complexity.item())

        sample_values = np.asarray(sample_values)

        mean_complexity = sample_values.mean()
        median_complexity = np.median(sample_values)
        std_complexity = sample_values.std()

        mean_complexities.append(mean_complexity)
        median_complexities.append(median_complexity)
        std_complexities.append(std_complexity)

        all_complexities.append(sample_values)

        print(
            f"omega={omega:.2f} | "
            f"mean={mean_complexity:.6f} | "
            f"median={median_complexity:.6f} | "
            f"std={std_complexity:.6f} | "
            f"min={sample_values.min():.6f} | "
            f"max={sample_values.max():.6f}"
        )

    mean_complexities = np.asarray(mean_complexities)
    median_complexities = np.asarray(median_complexities)
    std_complexities = np.asarray(std_complexities)
    all_complexities = np.stack(all_complexities)

    # Reference complexity C(1)
    reference_index = np.where(np.isclose(OMEGAS, 1.0))[0][0]
    reference_complexity = mean_complexities[reference_index]

    normalized_complexities = (
        mean_complexities / reference_complexity
    )

    print()
    print(f"Reference complexity C(1): {reference_complexity:.6f}")
    print()
    print("Normalized mapping complexity C(omega) / C(1):")

    for omega, ratio in zip(OMEGAS, normalized_complexities):
        print(
            f"omega={omega:.2f} | "
            f"ratio={ratio:.6f}"
        )

    # ---------------------------------------------------------
    # Save numerical results
    # ---------------------------------------------------------

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    np.savez(
        "results/exp03_mapping_complexity.npz",
        omegas=OMEGAS,
        mean_complexities=mean_complexities,
        median_complexities=median_complexities,
        std_complexities=std_complexities,
        normalized_complexities=normalized_complexities,
        all_complexities=all_complexities,
        z_samples=z_samples.detach().cpu().numpy(),
    )

    # ---------------------------------------------------------
    # Figure 1: absolute mapping complexity
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_complexities,
        marker="o",
        label=r"$\mathcal{C}(f^{(\omega)})$",
    )

    plt.axvline(
        1.0,
        linestyle="--",
        alpha=0.6,
        label=r"Reference $\omega=1$",
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel(
        r"$\mathbb{E}_z[\|\nabla f^{(\omega)}(z)\|_F^2]$"
    )
    plt.title("Exp 3: Mapping Complexity over the CFG Family")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_mapping_complexity_vs_omega.png",
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Figure 2: normalized mapping complexity
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        normalized_complexities,
        marker="o",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        alpha=0.6,
        label=r"$\mathcal{C}(1)$",
    )

    plt.axvline(
        1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel(
        r"$\mathcal{C}(f^{(\omega)})/"
        r"\mathcal{C}(f^{(1)})$"
    )
    plt.title(
        "Exp 3: Normalized Mapping Complexity"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp03_normalized_mapping_complexity_vs_omega.png",
        dpi=200,
    )

    plt.close()

    print()
    print("Saved:")
    print("  results/exp03_mapping_complexity.npz")
    print(
        "  results/figures/"
        "exp03_mapping_complexity_vs_omega.png"
    )
    print(
        "  results/figures/"
        "exp03_normalized_mapping_complexity_vs_omega.png"
    )


if __name__ == "__main__":
    main()