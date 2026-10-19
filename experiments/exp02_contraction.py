import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.distributions import sample_conditional_exp02
from src.remapping import exp02_remap_jacobian


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 0
N_SAMPLES = 200
N_STEPS = 200
TARGET_CLASS = 1

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

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

x_samples = sample_conditional_exp02(
    N_SAMPLES,
    c=TARGET_CLASS,
)


# ---------------------------------------------------------
# Compute Jacobian singular values
# ---------------------------------------------------------

all_sigma_max = []
all_sigma_min = []

print("Toy Exp 2 — CFG-Induced Re-Mapping Contraction")
print("=" * 60)
print(f"Number of samples: {N_SAMPLES}")
print(f"RK4 steps: {N_STEPS}")
print()


for omega in OMEGAS:
    sigma_max_values = []
    sigma_min_values = []

    for x in x_samples:
        J = exp02_remap_jacobian(
            x,
            omega=omega,
            c=TARGET_CLASS,
            n_steps=N_STEPS,
        )

        singular_values = torch.linalg.svdvals(J)

        sigma_max_values.append(
            singular_values.max().item()
        )

        sigma_min_values.append(
            singular_values.min().item()
        )

    sigma_max_values = np.asarray(
        sigma_max_values,
        dtype=np.float64,
    )

    sigma_min_values = np.asarray(
        sigma_min_values,
        dtype=np.float64,
    )

    all_sigma_max.append(sigma_max_values)
    all_sigma_min.append(sigma_min_values)

    print(
        f"omega={omega:4.2f} | "
        f"op mean={sigma_max_values.mean():.6f} | "
        f"median={np.median(sigma_max_values):.6f} | "
        f"max={sigma_max_values.max():.6f} | "
        f"sigma_min mean={sigma_min_values.mean():.6f}"
    )


all_sigma_max = np.stack(
    all_sigma_max,
    axis=0,
)

all_sigma_min = np.stack(
    all_sigma_min,
    axis=0,
)

omegas_np = np.asarray(
    OMEGAS,
    dtype=np.float64,
)


# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------

mean_sigma_max = all_sigma_max.mean(axis=1)
median_sigma_max = np.median(
    all_sigma_max,
    axis=1,
)
max_sigma_max = all_sigma_max.max(axis=1)

mean_sigma_min = all_sigma_min.mean(axis=1)
median_sigma_min = np.median(
    all_sigma_min,
    axis=1,
)
min_sigma_min = all_sigma_min.min(axis=1)


# ---------------------------------------------------------
# Save raw results
# ---------------------------------------------------------

np.savez(
    "results/exp02_operator_norms.npz",
    omegas=omegas_np,
    operator_norms=all_sigma_max,
    mean=mean_sigma_max,
    median=median_sigma_max,
    max=max_sigma_max,
)

np.savez(
    "results/exp02_singular_values.npz",
    omegas=omegas_np,
    sigma_max=all_sigma_max,
    sigma_min=all_sigma_min,
    mean_sigma_max=mean_sigma_max,
    mean_sigma_min=mean_sigma_min,
    median_sigma_max=median_sigma_max,
    median_sigma_min=median_sigma_min,
    max_sigma_max=max_sigma_max,
    min_sigma_min=min_sigma_min,
)


# ---------------------------------------------------------
# Figure 1: Main contraction metric
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    omegas_np,
    mean_sigma_max,
    marker="o",
    label=r"Mean $\sigma_{\max}$",
)

plt.axhline(
    1.0,
    linestyle="--",
    label="Identity",
)

plt.xlabel(r"CFG scale $\omega$")
plt.ylabel(
    r"$\mathbb{E}_x[\|\nabla g_\omega(x)\|_{\mathrm{op}}]$"
)

plt.title(
    "Exp 2: CFG-Induced Re-Mapping Contraction"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/"
    "exp02_mean_operator_norm_vs_omega.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Figure 2: Singular-value diagnostic
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    omegas_np,
    mean_sigma_max,
    marker="o",
    label=r"Mean $\sigma_{\max}$",
)

plt.plot(
    omegas_np,
    mean_sigma_min,
    marker="o",
    label=r"Mean $\sigma_{\min}$",
)

plt.axhline(
    1.0,
    linestyle="--",
    label="Identity",
)

plt.xlabel(r"CFG scale $\omega$")
plt.ylabel("Mean singular value")

plt.title(
    "Exp 2: Singular Values of the Re-Mapping Jacobian"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/"
    "exp02_mean_singular_values_vs_omega.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print()
print("Saved raw operator norms to:")
print("results/exp02_operator_norms.npz")

print()
print("Saved raw singular values to:")
print("results/exp02_singular_values.npz")

print()
print("Saved figures to:")
print(
    "results/figures/"
    "exp02_mean_operator_norm_vs_omega.png"
)
print(
    "results/figures/"
    "exp02_mean_singular_values_vs_omega.png"
)