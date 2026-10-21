import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev


torch.set_default_dtype(torch.float64)

SEED = 0
N_SAMPLES = 200
N_STEPS = 200
TARGET_CLASS = 1

OMEGAS = np.array(
    [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
)

# Pre-registered Exp 4 toy geometry.
MU_0 = torch.tensor([-2.0, 0.0])
MU_1 = torch.tensor([2.0, 0.0])

SIGMA_0 = torch.diag(
    torch.tensor([0.04, 1.0])
)
SIGMA_1 = torch.diag(
    torch.tensor([1.0, 0.04])
)


def class_mean(c):
    if c == 0:
        return MU_0
    if c == 1:
        return MU_1
    raise ValueError("c must be 0 or 1")


def class_covariance(c):
    if c == 0:
        return SIGMA_0
    if c == 1:
        return SIGMA_1
    raise ValueError("c must be 0 or 1")


def interpolation_covariance(t, c):
    """
    Covariance of X_t = (1-t) Z + t X,
    where Z ~ N(0, I) and X | c ~ N(mu_c, Sigma_c).
    """
    sigma = class_covariance(c)
    identity = torch.eye(
        2,
        dtype=sigma.dtype,
        device=sigma.device,
    )

    return (
        (1.0 - t) ** 2 * identity
        + t**2 * sigma
    )


def conditional_velocity(x, t, c):
    """
    Exact marginal velocity for one Gaussian class.
    """
    mu = class_mean(c).to(
        dtype=x.dtype,
        device=x.device,
    )

    sigma = class_covariance(c).to(
        dtype=x.dtype,
        device=x.device,
    )

    identity = torch.eye(
        2,
        dtype=x.dtype,
        device=x.device,
    )

    covariance_t = (
        (1.0 - t) ** 2 * identity
        + t**2 * sigma
    )

    centered = x - t * mu

    solved = torch.linalg.solve(
        covariance_t,
        centered.unsqueeze(-1),
    ).squeeze(-1)

    coefficient = (
        t * sigma
        - (1.0 - t) * identity
    )

    return mu + coefficient @ solved


def component_log_prob(x, t, c):
    """
    Log density of the class-conditional interpolation marginal,
    up to constants shared across classes.
    """
    mu = class_mean(c).to(
        dtype=x.dtype,
        device=x.device,
    )

    sigma = class_covariance(c).to(
        dtype=x.dtype,
        device=x.device,
    )

    identity = torch.eye(
        2,
        dtype=x.dtype,
        device=x.device,
    )

    covariance_t = (
        (1.0 - t) ** 2 * identity
        + t**2 * sigma
    )

    centered = x - t * mu

    solved = torch.linalg.solve(
        covariance_t,
        centered.unsqueeze(-1),
    ).squeeze(-1)

    quadratic = torch.sum(
        centered * solved,
        dim=-1,
    )

    _, logdet = torch.linalg.slogdet(
        covariance_t
    )

    return -0.5 * (
        quadratic + logdet
    )


def responsibilities(x, t):
    """
    Equal class priors.
    Returns posterior class probabilities.
    """
    log_p0 = component_log_prob(
        x,
        t,
        0,
    )

    log_p1 = component_log_prob(
        x,
        t,
        1,
    )

    logits = torch.stack(
        [log_p0, log_p1],
        dim=-1,
    )

    return torch.softmax(
        logits,
        dim=-1,
    )


def unconditional_velocity(x, t):
    """
    Exact unconditional marginal velocity of the two-component mixture.
    """
    probs = responsibilities(
        x,
        t,
    )

    v0 = conditional_velocity(
        x,
        t,
        0,
    )

    v1 = conditional_velocity(
        x,
        t,
        1,
    )

    return (
        probs[..., 0].unsqueeze(-1) * v0
        + probs[..., 1].unsqueeze(-1) * v1
    )


def cfg_velocity(x, t, omega, c=TARGET_CLASS):
    """
    CFG convention used throughout the previous experiments:

        omega = 0 : unconditional
        omega = 1 : conditional/reference
    """
    v_uncond = unconditional_velocity(
        x,
        t,
    )

    v_cond = conditional_velocity(
        x,
        t,
        c,
    )

    return (
        v_uncond
        + omega * (v_cond - v_uncond)
    )


def rk4_step(x, t, dt, omega, c):
    k1 = cfg_velocity(
        x,
        t,
        omega,
        c,
    )

    k2 = cfg_velocity(
        x + 0.5 * dt * k1,
        t + 0.5 * dt,
        omega,
        c,
    )

    k3 = cfg_velocity(
        x + 0.5 * dt * k2,
        t + 0.5 * dt,
        omega,
        c,
    )

    k4 = cfg_velocity(
        x + dt * k3,
        t + dt,
        omega,
        c,
    )

    return x + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


def cfg_flow(
    x,
    t_start,
    t_end,
    omega,
    c=TARGET_CLASS,
    n_steps=N_STEPS,
):
    dt = (
        t_end - t_start
    ) / n_steps

    y = x
    t = float(t_start)

    for _ in range(n_steps):
        y = rk4_step(
            y,
            t,
            dt,
            omega,
            c,
        )
        t += dt

    return y


def guided_map(z, omega):
    """
    f^(omega): latent t=0 -> data t=1.
    """
    return cfg_flow(
        z,
        0.0,
        1.0,
        omega,
        TARGET_CLASS,
        N_STEPS,
    )


def remap_from_latent(z, omega):
    """
    Evaluate g_omega at x=f(z).

    Since:
        g_omega = f^(omega) o f^{-1},

    at x=f(z):
        g_omega(f(z)) = f^(omega)(z).
    """
    x_reference = guided_map(
        z,
        1.0,
    )

    return x_reference


def map_jacobian(z, omega):
    return jacrev(
        lambda z_: guided_map(
            z_,
            omega,
        )
    )(z)


def main():
    torch.manual_seed(SEED)

    z_samples = torch.randn(
        N_SAMPLES,
        2,
    )

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

    reference_complexity_samples = torch.sum(
        reference_jacobians**2,
        dim=(1, 2),
    )

    reference_complexity = (
        reference_complexity_samples
        .mean()
        .item()
    )

    # Sanity check:
    # target covariance diag(1, 0.04)
    # => reference Gaussian transport Jacobian diag(1, 0.2)
    # => ||J_f||_F^2 = 1 + 0.04 = 1.04
    print(
        "Exp 4 — Contraction / Complexity Decoupling"
    )
    print("=" * 68)
    print(
        f"Number of latent samples: {N_SAMPLES}"
    )
    print(
        f"RK4 steps: {N_STEPS}"
    )
    print(
        f"Reference complexity C(1): "
        f"{reference_complexity:.6f}"
    )
    print()

    mean_g_op = []
    median_g_op = []
    max_g_op = []

    mean_complexity = []
    complexity_ratio = []

    fraction_decoupled = []

    all_g_op = []
    all_complexity = []
    all_local_ratio = []

    for omega in OMEGAS:
        g_op_values = []
        complexity_values = []
        local_ratio_values = []

        for i, z in enumerate(z_samples):
            j_f = reference_jacobians[i]

            j_f_omega = map_jacobian(
                z,
                float(omega),
            )

            # Exact identity:
            #
            # J_f^(omega) = J_g J_f
            #
            # therefore:
            #
            # J_g = J_f^(omega) J_f^{-1}
            j_g = torch.linalg.solve(
                j_f.T,
                j_f_omega.T,
            ).T

            singular_values = (
                torch.linalg.svdvals(j_g)
            )

            g_op = singular_values[0]

            complexity = torch.sum(
                j_f_omega**2
            )

            local_ratio = (
                complexity
                / reference_complexity_samples[i]
            )

            g_op_values.append(
                g_op.item()
            )

            complexity_values.append(
                complexity.item()
            )

            local_ratio_values.append(
                local_ratio.item()
            )

        g_op_values = np.asarray(
            g_op_values
        )

        complexity_values = np.asarray(
            complexity_values
        )

        local_ratio_values = np.asarray(
            local_ratio_values
        )

        # Sample-wise decoupling:
        #
        # g is locally non-contractive
        # but local mapping complexity decreases.
        decoupled = (
            (g_op_values > 1.0)
            & (local_ratio_values < 1.0)
        )

        mean_g_op.append(
            g_op_values.mean()
        )

        median_g_op.append(
            np.median(g_op_values)
        )

        max_g_op.append(
            g_op_values.max()
        )

        mean_complexity.append(
            complexity_values.mean()
        )

        complexity_ratio.append(
            complexity_values.mean()
            / reference_complexity
        )

        fraction_decoupled.append(
            decoupled.mean()
        )

        all_g_op.append(
            g_op_values
        )

        all_complexity.append(
            complexity_values
        )

        all_local_ratio.append(
            local_ratio_values
        )

        print(
            f"omega={omega:.2f} | "
            f"g_op mean={g_op_values.mean():.6f} | "
            f"median={np.median(g_op_values):.6f} | "
            f"max={g_op_values.max():.6f} | "
            f"C ratio="
            f"{complexity_values.mean() / reference_complexity:.6f} | "
            f"sample decoupled="
            f"{decoupled.mean():.3f}"
        )

    mean_g_op = np.asarray(
        mean_g_op
    )

    median_g_op = np.asarray(
        median_g_op
    )

    max_g_op = np.asarray(
        max_g_op
    )

    mean_complexity = np.asarray(
        mean_complexity
    )

    complexity_ratio = np.asarray(
        complexity_ratio
    )

    fraction_decoupled = np.asarray(
        fraction_decoupled
    )

    all_g_op = np.stack(
        all_g_op
    )

    all_complexity = np.stack(
        all_complexity
    )

    all_local_ratio = np.stack(
        all_local_ratio
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
        "results/exp04_decoupling.npz",
        omegas=OMEGAS,
        mean_g_op=mean_g_op,
        median_g_op=median_g_op,
        max_g_op=max_g_op,
        mean_complexity=mean_complexity,
        complexity_ratio=complexity_ratio,
        fraction_decoupled=fraction_decoupled,
        all_g_op=all_g_op,
        all_complexity=all_complexity,
        all_local_ratio=all_local_ratio,
        z_samples=z_samples.detach().cpu().numpy(),
    )

    # Figure 1:
    # primary contraction statistic.
    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        mean_g_op,
        marker="o",
        label=r"$\mathbb{E}\|J_g\|_{\mathrm{op}}$",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel("Mean operator norm")
    plt.title(
        "Exp 4: Induced Re-Mapping"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp04_mean_operator_norm.png",
        dpi=200,
    )

    plt.close()

    # Figure 2:
    # primary mapping-complexity statistic.
    plt.figure(figsize=(7, 5))

    plt.plot(
        OMEGAS,
        complexity_ratio,
        marker="o",
        label=r"$\mathcal{C}(\omega)/\mathcal{C}(1)$",
    )

    plt.axhline(
        1.0,
        linestyle="--",
        alpha=0.6,
    )

    plt.xlabel(r"CFG scale $\omega$")
    plt.ylabel("Normalized mapping complexity")
    plt.title(
        "Exp 4: Mapping Complexity"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp04_mapping_complexity_ratio.png",
        dpi=200,
    )

    plt.close()

    # Figure 3:
    # Direct comparison of the two pre-registered criteria.
    fig, ax1 = plt.subplots(
        figsize=(7, 5)
    )

    line1 = ax1.plot(
        OMEGAS,
        mean_g_op,
        marker="o",
        label="Mean operator norm",
    )

    ax1.set_xlabel(
        r"CFG scale $\omega$"
    )

    ax1.set_ylabel(
        "Mean operator norm"
    )

    ax1.axhline(
        1.0,
        linestyle="--",
        alpha=0.4,
    )

    ax2 = ax1.twinx()

    line2 = ax2.plot(
        OMEGAS,
        complexity_ratio,
        marker="s",
        linestyle="--",
        label="Complexity ratio",
    )

    ax2.set_ylabel(
        "Normalized mapping complexity"
    )

    ax2.axhline(
        1.0,
        linestyle=":",
        alpha=0.4,
    )

    lines = line1 + line2
    labels = [
        line.get_label()
        for line in lines
    ]

    ax1.legend(
        lines,
        labels,
        loc="best",
    )

    ax1.set_title(
        "Exp 4: Contraction vs. Complexity"
    )

    fig.tight_layout()

    plt.savefig(
        "results/figures/"
        "exp04_contraction_vs_complexity.png",
        dpi=200,
    )

    plt.close()

    print()
    print("Pre-registered target:")
    print(
        "  mean ||J_g||_op > 1 "
        "AND C(omega)/C(1) < 1"
    )

    target_mask = (
        (mean_g_op > 1.0)
        & (complexity_ratio < 1.0)
        & (OMEGAS > 1.0)
    )

    target_omegas = OMEGAS[
        target_mask
    ]

    if len(target_omegas) == 0:
        print(
            "Result: no omega > 1 "
            "satisfied the target criterion."
        )
    else:
        print(
            "Result: target criterion satisfied at:"
        )
        for omega in target_omegas:
            print(
                f"  omega={omega:.2f}"
            )

    print()
    print("Saved:")
    print(
        "  results/exp04_decoupling.npz"
    )
    print(
        "  results/figures/"
        "exp04_mean_operator_norm.png"
    )
    print(
        "  results/figures/"
        "exp04_mapping_complexity_ratio.png"
    )
    print(
        "  results/figures/"
        "exp04_contraction_vs_complexity.png"
    )


if __name__ == "__main__":
    main()