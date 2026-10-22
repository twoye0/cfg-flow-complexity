import math

import numpy as np
import torch

from src.distributions import sample_base
from experiments.exp05b_velocity_coupling import (
    integrate_sample,
)

import experiments.exp05b_velocity_coupling as exp05b


torch.set_default_dtype(torch.float64)

SEED = 0
N_SAMPLES = 200

STEP_COUNTS = [
    100,
    200,
    400,
]

BETAS = [
    0.0,
    math.pi / 3.0,   # 60 degrees
]


def compute_complexity(z_samples, beta, n_steps):
    # Change the global step count used by integrate_sample().
    exp05b.N_STEPS = n_steps

    sample_complexities = []

    for i, z in enumerate(z_samples):
        if (
            i == 0
            or (i + 1) % 50 == 0
            or i + 1 == N_SAMPLES
        ):
            print(
                f"    sample {i + 1}/{N_SAMPLES}"
            )

        _, J1 = integrate_sample(
            z,
            beta,
        )

        complexity_i = torch.sum(
            J1 * J1
        ).item()

        sample_complexities.append(
            complexity_i
        )

    sample_complexities = np.asarray(
        sample_complexities
    )

    return (
        np.mean(sample_complexities),
        np.std(sample_complexities),
        sample_complexities,
    )


def main():
    torch.manual_seed(SEED)

    # Same samples for every beta and every RK4 resolution.
    z_samples = sample_base(N_SAMPLES)

    print(
        "Exp 5-B — RK4 Convergence Check"
    )
    print("=" * 70)
    print(f"Samples: {N_SAMPLES}")
    print()

    results = {}

    for n_steps in STEP_COUNTS:
        print("=" * 70)
        print(
            f"N_STEPS = {n_steps}"
        )
        print("=" * 70)

        step_results = {}

        for beta in BETAS:
            beta_deg = round(
               beta * 180.0 / math.pi,
               10,
            )

            print()
            print(
                f"beta = {beta_deg:.1f} deg"
            )

            mean_c, std_c, sample_c = (
                compute_complexity(
                    z_samples,
                    beta,
                    n_steps,
                )
            )

            step_results[beta_deg] = {
                "mean": mean_c,
                "std": std_c,
                "samples": sample_c,
            }

            print(
                f"    C1 = {mean_c:.12f}"
            )

        c0 = step_results[0.0]["mean"]
        c60 = step_results[60.0]["mean"]

        delta = c60 - c0

        # Since the same z samples are used, this paired
        # difference is more informative than comparing
        # the two marginal standard deviations.
        paired_delta = (
            step_results[60.0]["samples"]
            - step_results[0.0]["samples"]
        )

        delta_se = (
            np.std(
                paired_delta,
                ddof=1,
            )
            / np.sqrt(N_SAMPLES)
        )

        results[n_steps] = {
            "C0": c0,
            "C60": c60,
            "delta": delta,
            "delta_se": delta_se,
        }

        print()
        print(
            f"Delta C = C60 - C0 "
            f"= {delta:+.12f}"
        )

        print(
            f"Paired SE(Delta C) "
            f"= {delta_se:.3e}"
        )

    # --------------------------------------------------------
    # Final convergence table
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Convergence summary")
    print("=" * 70)

    print(
        f"{'steps':>8}  "
        f"{'C0':>14}  "
        f"{'C60':>14}  "
        f"{'Delta C':>14}  "
        f"{'paired SE':>12}"
    )

    for n_steps in STEP_COUNTS:
        r = results[n_steps]

        print(
            f"{n_steps:8d}  "
            f"{r['C0']:14.10f}  "
            f"{r['C60']:14.10f}  "
            f"{r['delta']:14.10f}  "
            f"{r['delta_se']:12.3e}"
        )

    print()
    print("Successive changes in Delta C")

    for n_prev, n_curr in zip(
        STEP_COUNTS[:-1],
        STEP_COUNTS[1:],
    ):
        previous = results[n_prev]["delta"]
        current = results[n_curr]["delta"]

        change = abs(
            current - previous
        )

        print(
            f"{n_prev:3d} -> {n_curr:3d}: "
            f"|DeltaC_curr - DeltaC_prev| "
            f"= {change:.3e}"
        )


if __name__ == "__main__":
    main()