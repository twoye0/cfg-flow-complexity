import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jacrev

from src.distributions import sample_base
from src.velocity import exp02_cfg_velocity


# ============================================================
# Configuration
# ============================================================

SEED = 0
N_SAMPLES = 200
N_STEPS = 200

TARGET_CLASS = 1

# Compare unguided conditional flow against CFG.
OMEGAS = [1.0, 2.0]

torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)

FIGURE_DIR = "results/figures"
RESULTS_DIR = "results"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Velocity and its Jacobian
# ============================================================

def velocity(x, t, omega):
    return exp02_cfg_velocity(
        x,
        t,
        omega=omega,
        c=TARGET_CLASS,
    )


def velocity_jacobian(x, t, omega):
    """
    A_t = nabla_x v_t(x)
    """
    return jacrev(
        lambda y: velocity(y, t, omega)
    )(x)


# ============================================================
# Joint dynamics
#
#   dx/dt = v_t(x)
#   dJ/dt = A_t J
# ============================================================

def joint_dynamics(x, J, t, omega):
    v = velocity(x, t, omega)
    A = velocity_jacobian(x, t, omega)

    J_dot = A @ J

    return v, J_dot


def joint_rk4_step(x, J, t, dt, omega):
    k1_x, k1_J = joint_dynamics(
        x, J, t, omega
    )

    k2_x, k2_J = joint_dynamics(
        x + 0.5 * dt * k1_x,
        J + 0.5 * dt * k1_J,
        t + 0.5 * dt,
        omega,
    )

    k3_x, k3_J = joint_dynamics(
        x + 0.5 * dt * k2_x,
        J + 0.5 * dt * k2_J,
        t + 0.5 * dt,
        omega,
    )

    k4_x, k4_J = joint_dynamics(
        x + dt * k3_x,
        J + dt * k3_J,
        t + dt,
        omega,
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


# ============================================================
# State diagnostic
#
# M_t = J_t J_t^T
#
# M_t = m1 u1 u1^T + m2 u2 u2^T
#
# q_i = 2 m_i u_i^T S_t u_i
#
# Exact:
#
# q1 + q2 = 2 tr(S_t M_t)
#           = d/dt ||J_t||_F^2
# ============================================================

def record_state(x, J, t, omega):
    M = J @ J.T

    eigvals, eigvecs = torch.linalg.eigh(M)

    # Descending order: m1 >= m2
    eigvals = torch.flip(
        eigvals,
        dims=[0],
    )

    eigvecs = torch.flip(
        eigvecs,
        dims=[1],
    )

    m1 = eigvals[0]
    m2 = eigvals[1]

    u1 = eigvecs[:, 0]
    u2 = eigvecs[:, 1]

    A = velocity_jacobian(
        x,
        t,
        omega,
    )

    S = 0.5 * (A + A.T)

    r1 = u1 @ S @ u1
    r2 = u2 @ S @ u2

    q1 = 2.0 * m1 * r1
    q2 = 2.0 * m2 * r2

    exact_rate = 2.0 * torch.trace(
        S @ M
    )

    complexity = torch.trace(M)

    return {
        "m1": m1.item(),
        "m2": m2.item(),
        "r1": r1.item(),
        "r2": r2.item(),
        "q1": q1.item(),
        "q2": q2.item(),
        "exact_rate": exact_rate.item(),
        "complexity": complexity.item(),
    }


# ============================================================
# Run one omega
# ============================================================

def run_omega(
    omega,
    z_samples,
    times,
):
    shape = (
        N_SAMPLES,
        N_STEPS + 1,
    )

    m1_all = np.zeros(shape)
    m2_all = np.zeros(shape)

    r1_all = np.zeros(shape)
    r2_all = np.zeros(shape)

    q1_all = np.zeros(shape)
    q2_all = np.zeros(shape)

    exact_rate_all = np.zeros(shape)
    complexity_all = np.zeros(shape)

    identity = torch.eye(
        2,
        dtype=torch.get_default_dtype(),
    )

    dt = 1.0 / N_STEPS

    print()
    print("=" * 72)
    print(f"Running omega = {omega:.1f}")
    print("=" * 72)

    for sample_idx in range(N_SAMPLES):
        if (
            sample_idx == 0
            or (sample_idx + 1) % 20 == 0
            or sample_idx == N_SAMPLES - 1
        ):
            print(
                f"Processing sample "
                f"{sample_idx + 1}/{N_SAMPLES}"
            )

        x = z_samples[
            sample_idx
        ].clone()

        J = identity.clone()

        # t = 0
        state = record_state(
            x,
            J,
            times[0],
            omega,
        )

        m1_all[sample_idx, 0] = state["m1"]
        m2_all[sample_idx, 0] = state["m2"]

        r1_all[sample_idx, 0] = state["r1"]
        r2_all[sample_idx, 0] = state["r2"]

        q1_all[sample_idx, 0] = state["q1"]
        q2_all[sample_idx, 0] = state["q2"]

        exact_rate_all[
            sample_idx, 0
        ] = state["exact_rate"]

        complexity_all[
            sample_idx, 0
        ] = state["complexity"]

        # Forward integration
        for step in range(N_STEPS):
            t = times[step]

            x, J = joint_rk4_step(
                x,
                J,
                t,
                dt,
                omega,
            )

            state = record_state(
                x,
                J,
                times[step + 1],
                omega,
            )

            m1_all[
                sample_idx,
                step + 1,
            ] = state["m1"]

            m2_all[
                sample_idx,
                step + 1,
            ] = state["m2"]

            r1_all[
                sample_idx,
                step + 1,
            ] = state["r1"]

            r2_all[
                sample_idx,
                step + 1,
            ] = state["r2"]

            q1_all[
                sample_idx,
                step + 1,
            ] = state["q1"]

            q2_all[
                sample_idx,
                step + 1,
            ] = state["q2"]

            exact_rate_all[
                sample_idx,
                step + 1,
            ] = state["exact_rate"]

            complexity_all[
                sample_idx,
                step + 1,
            ] = state["complexity"]

    # Population means
    q1 = np.mean(
        q1_all,
        axis=0,
    )

    q2 = np.mean(
        q2_all,
        axis=0,
    )

    exact_rate = np.mean(
        exact_rate_all,
        axis=0,
    )

    complexity = np.mean(
        complexity_all,
        axis=0,
    )

    mean_m1 = np.mean(
        m1_all,
        axis=0,
    )

    mean_m2 = np.mean(
        m2_all,
        axis=0,
    )

    mean_r1 = np.mean(
        r1_all,
        axis=0,
    )

    mean_r2 = np.mean(
        r2_all,
        axis=0,
    )

    q_sum = q1 + q2

    return {
        "omega": omega,
        "q1": q1,
        "q2": q2,
        "q_sum": q_sum,
        "exact_rate": exact_rate,
        "complexity": complexity,
        "mean_m1": mean_m1,
        "mean_m2": mean_m2,
        "mean_r1": mean_r1,
        "mean_r2": mean_r2,
    }


# ============================================================
# Trapezoidal integral helper
# ============================================================

def integrate_curve(y, x):
    return np.trapezoid(
        y,
        x,
    )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "Exp 06 — CFG Mapping-Complexity Mechanism"
    )
    print("=" * 72)

    print(f"Samples: {N_SAMPLES}")
    print(f"RK4 steps: {N_STEPS}")
    print(
        "Comparing omega = "
        + ", ".join(
            str(omega)
            for omega in OMEGAS
        )
    )

    # IMPORTANT:
    # The same latent samples are reused for both omega values.
    z_samples = sample_base(
        N_SAMPLES
    )

    times = torch.linspace(
        0.0,
        1.0,
        N_STEPS + 1,
        dtype=torch.get_default_dtype(),
    )

    times_np = (
        times
        .detach()
        .cpu()
        .numpy()
    )

    results = {}

    for omega in OMEGAS:
        results[omega] = run_omega(
            omega,
            z_samples,
            times,
        )

    unguided = results[1.0]
    guided = results[2.0]

    # ========================================================
    # Per-omega sanity checks
    # ========================================================

    print()
    print("=" * 72)
    print("Per-omega sanity checks")
    print("=" * 72)

    for omega in OMEGAS:
        result = results[omega]

        decomposition_error = np.max(
            np.abs(
                result["q_sum"]
                - result["exact_rate"]
            )
        )

        integral_rate = integrate_curve(
            result["exact_rate"],
            times_np,
        )

        observed_change = (
            result["complexity"][-1]
            - result["complexity"][0]
        )

        integration_error = (
            integral_rate
            - observed_change
        )

        min_idx = int(
            np.argmin(
                result["complexity"]
            )
        )

        print()
        print(f"omega = {omega:.1f}")
        print(
            f"  C(0) = "
            f"{result['complexity'][0]:.12f}"
        )
        print(
            f"  C(1) = "
            f"{result['complexity'][-1]:.12f}"
        )
        print(
            f"  min C(t) = "
            f"{result['complexity'][min_idx]:.12f} "
            f"at t={times_np[min_idx]:.3f}"
        )
        print(
            "  max |q1+q2-exact_rate| = "
            f"{decomposition_error:.6e}"
        )
        print(
            "  integral exact_rate dt = "
            f"{integral_rate:.12f}"
        )
        print(
            "  observed C(1)-C(0) = "
            f"{observed_change:.12f}"
        )
        print(
            "  integration error = "
            f"{integration_error:.6e}"
        )

    # ========================================================
    # Exact CFG difference
    #
    # Delta C(t) = C_guided(t) - C_unguided(t)
    #
    # Delta R(t) = R_guided(t) - R_unguided(t)
    #
    # Therefore:
    #
    # Delta C(1) - Delta C(0)
    # =
    # integral Delta R(t) dt
    #
    # Since both begin with J_0 = I,
    # Delta C(0) = 0.
    # ========================================================

    delta_complexity = (
        guided["complexity"]
        - unguided["complexity"]
    )

    delta_rate = (
        guided["exact_rate"]
        - unguided["exact_rate"]
    )

    delta_q1 = (
        guided["q1"]
        - unguided["q1"]
    )

    delta_q2 = (
        guided["q2"]
        - unguided["q2"]
    )

    integral_delta_rate = integrate_curve(
        delta_rate,
        times_np,
    )

    final_delta_complexity = (
        delta_complexity[-1]
    )

    difference_error = (
        integral_delta_rate
        - final_delta_complexity
    )

    # ========================================================
    # Where is the extra CFG reduction accumulated?
    # ========================================================

    cumulative_delta_rate = np.zeros_like(
        delta_rate
    )

    for i in range(1, len(times_np)):
        dt_i = (
            times_np[i]
            - times_np[i - 1]
        )

        cumulative_delta_rate[i] = (
            cumulative_delta_rate[i - 1]
            + 0.5
            * dt_i
            * (
                delta_rate[i - 1]
                + delta_rate[i]
            )
        )

    min_delta_rate_idx = int(
        np.argmin(delta_rate)
    )

    max_delta_rate_idx = int(
        np.argmax(delta_rate)
    )

    print()
    print("=" * 72)
    print("CFG mechanism comparison")
    print("=" * 72)

    print(
        "Final complexity difference "
        "C_omega2(1) - C_omega1(1):"
    )
    print(
        f"  {final_delta_complexity:.12f}"
    )

    print()
    print(
        "Integral of rate difference:"
    )
    print(
        f"  {integral_delta_rate:.12f}"
    )

    print()
    print(
        "Difference identity error:"
    )
    print(
        f"  {difference_error:.6e}"
    )

    print()
    print(
        "Most negative Delta rate:"
    )
    print(
        f"  t = "
        f"{times_np[min_delta_rate_idx]:.3f}"
    )
    print(
        f"  Delta rate = "
        f"{delta_rate[min_delta_rate_idx]:.6f}"
    )
    print(
        f"  Delta q1 = "
        f"{delta_q1[min_delta_rate_idx]:.6f}"
    )
    print(
        f"  Delta q2 = "
        f"{delta_q2[min_delta_rate_idx]:.6f}"
    )

    print()
    print(
        "Most positive Delta rate:"
    )
    print(
        f"  t = "
        f"{times_np[max_delta_rate_idx]:.3f}"
    )
    print(
        f"  Delta rate = "
        f"{delta_rate[max_delta_rate_idx]:.6f}"
    )

    # ========================================================
    # Small time table
    # ========================================================

    print()
    print("=" * 72)
    print("Selected times")
    print("=" * 72)

    print(
        "    t"
        "       C(w=1)"
        "       C(w=2)"
        "      Delta C"
        "      R(w=1)"
        "      R(w=2)"
        "      Delta R"
    )

    selected_times = [
        0.00,
        0.25,
        0.50,
        0.65,
        0.75,
        1.00,
    ]

    for selected_t in selected_times:
        idx = int(
            round(
                selected_t
                * N_STEPS
            )
        )

        print(
            f"{times_np[idx]:7.3f}"
            f"  "
            f"{unguided['complexity'][idx]:11.6f}"
            f"  "
            f"{guided['complexity'][idx]:11.6f}"
            f"  "
            f"{delta_complexity[idx]:11.6f}"
            f"  "
            f"{unguided['exact_rate'][idx]:11.6f}"
            f"  "
            f"{guided['exact_rate'][idx]:11.6f}"
            f"  "
            f"{delta_rate[idx]:11.6f}"
        )

    # ========================================================
    # Save results
    # ========================================================

    output_path = os.path.join(
        RESULTS_DIR,
        "exp06_cfg_mechanism.npz",
    )

    np.savez(
        output_path,
        times=times_np,

        complexity_omega1=(
            unguided["complexity"]
        ),
        complexity_omega2=(
            guided["complexity"]
        ),

        rate_omega1=(
            unguided["exact_rate"]
        ),
        rate_omega2=(
            guided["exact_rate"]
        ),

        q1_omega1=unguided["q1"],
        q2_omega1=unguided["q2"],

        q1_omega2=guided["q1"],
        q2_omega2=guided["q2"],

        delta_complexity=(
            delta_complexity
        ),
        delta_rate=delta_rate,

        delta_q1=delta_q1,
        delta_q2=delta_q2,

        cumulative_delta_rate=(
            cumulative_delta_rate
        ),
    )

    # ========================================================
    # Figure 1 — Complexity dynamics
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        times_np,
        unguided["complexity"],
        label=r"$\omega=1$",
        linewidth=2,
    )

    plt.plot(
        times_np,
        guided["complexity"],
        label=r"$\omega=2$",
        linewidth=2,
    )

    plt.xlabel("t")
    plt.ylabel(
        r"$C_t=E[\mathrm{tr}(M_t)]$"
    )

    plt.title(
        "Mapping Complexity: "
        "Unguided vs CFG"
    )

    plt.legend()
    plt.tight_layout()

    figure_path_1 = os.path.join(
        FIGURE_DIR,
        "exp06_complexity_comparison.png",
    )

    plt.savefig(
        figure_path_1,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Figure 2 — Exact complexity rates
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        times_np,
        unguided["exact_rate"],
        label=r"$\omega=1$",
        linewidth=2,
    )

    plt.plot(
        times_np,
        guided["exact_rate"],
        label=r"$\omega=2$",
        linewidth=2,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel("t")
    plt.ylabel(
        r"$2E[\mathrm{tr}(S_tM_t)]$"
    )

    plt.title(
        "Exact Mapping-Complexity Rate"
    )

    plt.legend()
    plt.tight_layout()

    figure_path_2 = os.path.join(
        FIGURE_DIR,
        "exp06_rate_comparison.png",
    )

    plt.savefig(
        figure_path_2,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Figure 3 — CFG-induced rate difference
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        times_np,
        delta_rate,
        label=r"$\Delta R_t$",
        linewidth=2,
    )

    plt.plot(
        times_np,
        delta_q1,
        "--",
        label=r"$\Delta q_1$",
        linewidth=1.5,
    )

    plt.plot(
        times_np,
        delta_q2,
        "--",
        label=r"$\Delta q_2$",
        linewidth=1.5,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel("t")
    plt.ylabel(
        "CFG-induced change "
        "in complexity rate"
    )

    plt.title(
        "Where Does CFG Change "
        "Mapping Complexity?"
    )

    plt.legend()
    plt.tight_layout()

    figure_path_3 = os.path.join(
        FIGURE_DIR,
        "exp06_cfg_rate_difference.png",
    )

    plt.savefig(
        figure_path_3,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Figure 4 — Accumulated CFG effect
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        times_np,
        delta_complexity,
        label=r"$C_t^{(2)}-C_t^{(1)}$",
        linewidth=2,
    )

    plt.plot(
        times_np,
        cumulative_delta_rate,
        "--",
        label=(
            r"$\int_0^t "
            r"\Delta R_s\,ds$"
        ),
        linewidth=1.5,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.xlabel("t")
    plt.ylabel(
        "CFG-induced complexity change"
    )

    plt.title(
        "Accumulated CFG Effect"
    )

    plt.legend()
    plt.tight_layout()

    figure_path_4 = os.path.join(
        FIGURE_DIR,
        "exp06_accumulated_cfg_effect.png",
    )

    plt.savefig(
        figure_path_4,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Done
    # ========================================================

    print()
    print("=" * 72)
    print("Saved")
    print("=" * 72)

    print(output_path)
    print(figure_path_1)
    print(figure_path_2)
    print(figure_path_3)
    print(figure_path_4)


if __name__ == "__main__":
    main()