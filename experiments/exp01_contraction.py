import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.distributions import sample_conditional
from src.remapping import remap_jacobian


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 0
N_SAMPLES = 200
N_STEPS = 200

TARGET_CLASS = 1
SIGMA = 0.5

OMEGAS = [
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
]


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)


# ---------------------------------------------------------
# Sample from the reference conditional distribution
# ---------------------------------------------------------

x_samples = sample_conditional(
    N_SAMPLES,
    c=TARGET_CLASS,
    sigma=SIGMA,
)

print(f"Number of samples: {N_SAMPLES}")
print(f"RK4 steps: {N_STEPS}")
print()


# ---------------------------------------------------------
# Compute singular values of the re-mapping Jacobian
# ---------------------------------------------------------

sigma_max_results = {}
sigma_min_results = {}

for omega in OMEGAS:

    sigma_max_values = []
    sigma_min_values = []

    for x in x_samples:

        J = remap_jacobian(
            x,
            omega=omega,
            c=TARGET_CLASS,
            sigma=SIGMA,
            n_steps=N_STEPS,
        )

        singular_values = torch.linalg.svdvals(J)

        sigma_max = singular_values.max().item()
        sigma_min = singular_values.min().item()

        sigma_max_values.append(sigma_max)
        sigma_min_values.append(sigma_min)

    sigma_max_values = np.asarray(sigma_max_values)
    sigma_min_values = np.asarray(sigma_min_values)

    sigma_max_results[omega] = sigma_max_values
    sigma_min_results[omega] = sigma_min_values

    print(
        f"omega={omega:4.2f} | "
        f"sigma_max mean={sigma_max_values.mean():.6f} | "
        f"sigma_min mean={sigma_min_values.mean():.6f} | "
        f"sigma_min median={np.median(sigma_min_values):.6f} | "
        f"sigma_min min={sigma_min_values.min():.6f}"
    )


# ---------------------------------------------------------
# Save raw results
# ---------------------------------------------------------

save_dict = {
    "omegas": np.asarray(OMEGAS),
}

for omega in OMEGAS:
    key = str(omega).replace(".", "_")

    save_dict[f"sigma_max_{key}"] = sigma_max_results[omega]
    save_dict[f"sigma_min_{key}"] = sigma_min_results[omega]

np.savez(
    "results/exp01_singular_values.npz",
    **save_dict,
)

print()
print("Saved raw singular values to:")
print("results/exp01_singular_values.npz")


# ---------------------------------------------------------
# Compute means across samples
# ---------------------------------------------------------

mean_sigma_max = np.asarray(
    [
        sigma_max_results[omega].mean()
        for omega in OMEGAS
    ]
)

mean_sigma_min = np.asarray(
    [
        sigma_min_results[omega].mean()
        for omega in OMEGAS
    ]
)


# ---------------------------------------------------------
# Plot singular-value diagnostic
# ---------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    OMEGAS,
    mean_sigma_max,
    marker="o",
    label=r"mean $\sigma_{\max}$",
)

plt.plot(
    OMEGAS,
    mean_sigma_min,
    marker="o",
    label=r"mean $\sigma_{\min}$",
)

plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="identity",
)

plt.xlabel(r"CFG scale $\omega$")
plt.ylabel("Mean singular value")

plt.title(
    "Singular Values of the CFG-Induced Re-Mapping Jacobian"
)

plt.legend()
plt.tight_layout()

figure_path = (
    "results/figures/"
    "exp01_mean_singular_values_vs_omega.png"
)

plt.savefig(
    figure_path,
    dpi=200,
)

plt.close()

print("Saved figure to:")
print(figure_path)