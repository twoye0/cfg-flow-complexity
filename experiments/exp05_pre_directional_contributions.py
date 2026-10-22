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
OMEGA = 2.0

torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)

FIGURE_DIR = "results/figures"
RESULTS_DIR = "results"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Velocity and its Jacobian
# ============================================================

def velocity(x, t):
    return exp02_cfg_velocity(
        x,
        t,
        omega=OMEGA,
        c=TARGET_CLASS,
    )


def velocity_jacobian(x, t):
    """
    A_t = ∇_x v_t(x)
    """
    return jacrev(lambda y: velocity(y, t))(x)


# ============================================================
# Joint dynamics
#
#   dx/dt = v_t(x)
#   dJ/dt = A_t J
#
# ============================================================

def joint_dynamics(x, J, t):
    v = velocity(x, t)
    A = velocity_jacobian(x, t)

    dJ = A @ J

    return v, dJ


def joint_rk4_step(x, J, t, dt):
    k1_x, k1_J = joint_dynamics(x, J, t)

    k2_x, k2_J = joint_dynamics(
        x + 0.5 * dt * k1_x,
        J + 0.5 * dt * k1_J,
        t + 0.5 * dt,
    )

    k3_x, k3_J = joint_dynamics(
        x + 0.5 * dt * k2_x,
        J + 0.5 * dt * k2_J,
        t + 0.5 * dt,
    )

    k4_x, k4_J = joint_dynamics(
        x + dt * k3_x,
        J + dt * k3_J,
        t + dt,
    )

    x_next = x + (dt / 6.0) * (
        k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x
    )

    J_next = J + (dt / 6.0) * (
        k1_J + 2.0 * k2_J + 2.0 * k3_J + k4_J
    )

    return x_next, J_next


# ============================================================
# Directional contribution diagnostic
# ============================================================

def record_state(x, J, t):
    """
    M_t = J_t J_t^T

    Eigendecomposition:
        M_t = m1 u1 u1^T + m2 u2 u2^T
        with m1 >= m2

    Directional stretch rates:
        r_i = u_i^T S_t u_i

    Contributions:
        q_i(sample) = 2 m_i r_i

    Exact identity:
        2 tr(S_t M_t)
        = q_1(sample) + q_2(sample)
    """

    M = J @ J.T

    # eigh returns eigenvalues in ascending order.
    eigvals, eigvecs = torch.linalg.eigh(M)

    # Reverse so that m1 >= m2.
    eigvals = torch.flip(eigvals, dims=[0])
    eigvecs = torch.flip(eigvecs, dims=[1])

    m1 = eigvals[0]
    m2 = eigvals[1]

    u1 = eigvecs[:, 0]
    u2 = eigvecs[:, 1]

    A = velocity_jacobian(x, t)

    S = 0.5 * (A + A.T)

    r1 = u1 @ S @ u1
    r2 = u2 @ S @ u2

    contribution_1 = 2.0 * m1 * r1
    contribution_2 = 2.0 * m2 * r2

    exact_rhs = 2.0 * torch.trace(S @ M)

    trace_M = torch.trace(M)

    return {
        "m1": m1.item(),
        "m2": m2.item(),
        "r1": r1.item(),
        "r2": r2.item(),
        "q1_sample": contribution_1.item(),
        "q2_sample": contribution_2.item(),
        "exact_rhs": exact_rhs.item(),
        "trace_M": trace_M.item(),
    }


# ============================================================
# Main experiment
# ============================================================

def main():
    print("Pre-Exp 5 — Directional Complexity Contributions")
    print("=" * 60)
    print(f"Samples: {N_SAMPLES}")
    print(f"RK4 steps: {N_STEPS}")
    print(f"CFG omega: {OMEGA}")
    print()

    z_samples = sample_base(N_SAMPLES)

    times = torch.linspace(
        0.0,
        1.0,
        N_STEPS + 1,
        dtype=torch.get_default_dtype(),
    )

    dt = 1.0 / N_STEPS

    shape = (N_SAMPLES, N_STEPS + 1)

    m1_all = np.zeros(shape)
    m2_all = np.zeros(shape)

    r1_all = np.zeros(shape)
    r2_all = np.zeros(shape)

    q1_sample_all = np.zeros(shape)
    q2_sample_all = np.zeros(shape)

    exact_rhs_all = np.zeros(shape)
    trace_M_all = np.zeros(shape)

    identity = torch.eye(
        2,
        dtype=torch.get_default_dtype(),
    )

    # --------------------------------------------------------
    # Integrate each trajectory
    # --------------------------------------------------------

    for sample_idx in range(N_SAMPLES):

        if (
            sample_idx == 0
            or (sample_idx + 1) % 10 == 1
            or sample_idx == N_SAMPLES - 1
        ):
            print(
                f"Processing sample "
                f"{sample_idx + 1}/{N_SAMPLES}"
            )

        x = z_samples[sample_idx].clone()
        J = identity.clone()

        # t = 0
        state = record_state(
            x,
            J,
            times[0],
        )

        m1_all[sample_idx, 0] = state["m1"]
        m2_all[sample_idx, 0] = state["m2"]

        r1_all[sample_idx, 0] = state["r1"]
        r2_all[sample_idx, 0] = state["r2"]

        q1_sample_all[sample_idx, 0] = state["q1_sample"]
        q2_sample_all[sample_idx, 0] = state["q2_sample"]

        exact_rhs_all[sample_idx, 0] = state["exact_rhs"]
        trace_M_all[sample_idx, 0] = state["trace_M"]

        # Forward integration
        for step in range(N_STEPS):

            t = times[step]

            x, J = joint_rk4_step(
                x,
                J,
                t,
                dt,
            )

            state = record_state(
                x,
                J,
                times[step + 1],
            )

            m1_all[sample_idx, step + 1] = state["m1"]
            m2_all[sample_idx, step + 1] = state["m2"]

            r1_all[sample_idx, step + 1] = state["r1"]
            r2_all[sample_idx, step + 1] = state["r2"]

            q1_sample_all[sample_idx, step + 1] = (
                state["q1_sample"]
            )
            q2_sample_all[sample_idx, step + 1] = (
                state["q2_sample"]
            )

            exact_rhs_all[sample_idx, step + 1] = (
                state["exact_rhs"]
            )

            trace_M_all[sample_idx, step + 1] = (
                state["trace_M"]
            )

    # ========================================================
    # Population-level quantities
    # ========================================================

    times_np = times.detach().cpu().numpy()

    # IMPORTANT:
    # mapping complexity is an expectation,
    # so we use sample MEANS here.
    q1 = np.mean(q1_sample_all, axis=0)
    q2 = np.mean(q2_sample_all, axis=0)

    q_sum = q1 + q2

    exact_rate = np.mean(
        exact_rhs_all,
        axis=0,
    )

    complexity = np.mean(
        trace_M_all,
        axis=0,
    )

    mean_m1 = np.mean(m1_all, axis=0)
    mean_m2 = np.mean(m2_all, axis=0)

    mean_r1 = np.mean(r1_all, axis=0)
    mean_r2 = np.mean(r2_all, axis=0)

    # ========================================================
    # Sanity check 1:
    #
    # q1 + q2 must equal
    # 2 E[tr(S M)]
    #
    # ========================================================

    decomposition_error = np.abs(
        q_sum - exact_rate
    )

    print()
    print("Sanity check: directional decomposition")
    print(
        "max |(q1 + q2) - exact rate| "
        f"= {np.max(decomposition_error):.6e}"
    )
    print(
        "mean |(q1 + q2) - exact rate| "
        f"= {np.mean(decomposition_error):.6e}"
    )

    # ========================================================
    # Locate minimum complexity
    # ========================================================

    min_idx = int(np.argmin(complexity))
    min_time = times_np[min_idx]

    print()
    print("Mapping complexity:")
    print(
        f"C(0) = {complexity[0]:.6f}"
    )
    print(
        f"minimum C(t) = "
        f"{complexity[min_idx]:.6f} "
        f"at t={min_time:.3f}"
    )
    print(
        f"C(1) = {complexity[-1]:.6f}"
    )

    # ========================================================
    # Contributions near complexity minimum
    # ========================================================

    print()
    print("Directional contributions at C(t) minimum:")
    print(
        f"t = {min_time:.3f}"
    )
    print(
        f"q1 = 2 E[m1 r1] = "
        f"{q1[min_idx]:.6f}"
    )
    print(
        f"q2 = 2 E[m2 r2] = "
        f"{q2[min_idx]:.6f}"
    )
    print(
        f"q1 + q2 = "
        f"{q_sum[min_idx]:.6f}"
    )
    print(
        f"exact dC/dt = "
        f"{exact_rate[min_idx]:.6f}"
    )

    # Print a small neighborhood around the minimum.
    print()
    print("Neighborhood around complexity minimum:")
    print(
        "    t          q1          q2       "
        "q1+q2       C(t)"
    )

    for offset in [-4, -2, 0, 2, 4]:
        idx = min(
            max(min_idx + offset, 0),
            N_STEPS,
        )

        print(
            f"{times_np[idx]:7.3f}  "
            f"{q1[idx]:10.6f}  "
            f"{q2[idx]:10.6f}  "
            f"{q_sum[idx]:10.6f}  "
            f"{complexity[idx]:10.6f}"
        )

    # ========================================================
    # Find approximate zero crossing of total rate
    # ========================================================

    sign_changes = np.where(
        np.sign(q_sum[:-1])
        != np.sign(q_sum[1:])
    )[0]

    print()
    print("Zero crossings of q1 + q2:")

    if len(sign_changes) == 0:
        print("No zero crossing found.")
    else:
        for idx in sign_changes:
            print(
                f"between t={times_np[idx]:.3f} "
                f"and t={times_np[idx + 1]:.3f}"
            )

    # ========================================================
    # Save raw data
    # ========================================================

    output_path = os.path.join(
        RESULTS_DIR,
        "exp05_pre_directional_contributions.npz",
    )

    np.savez(
        output_path,
        times=times_np,
        m1=m1_all,
        m2=m2_all,
        r1=r1_all,
        r2=r2_all,
        q1_sample=q1_sample_all,
        q2_sample=q2_sample_all,
        q1=q1,
        q2=q2,
        q_sum=q_sum,
        exact_rate=exact_rate,
        trace_M=trace_M_all,
        complexity=complexity,
        mean_m1=mean_m1,
        mean_m2=mean_m2,
        mean_r1=mean_r1,
        mean_r2=mean_r2,
    )

    # ========================================================
    # Figure 1:
    # Exact directional contributions
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        times_np,
        q1,
        label=r"$q_1(t)=2\,E[m_1 r_1]$",
        linewidth=2,
    )

    plt.plot(
        times_np,
        q2,
        label=r"$q_2(t)=2\,E[m_2 r_2]$",
        linewidth=2,
    )

    plt.plot(
        times_np,
        q_sum,
        label=r"$q_1(t)+q_2(t)$",
        linewidth=2,
    )

    plt.plot(
        times_np,
        exact_rate,
        "--",
        label=r"$2\,E[\mathrm{tr}(S_t M_t)]$",
        linewidth=1.5,
    )

    plt.axhline(
        0.0,
        linewidth=1,
    )

    plt.axvline(
        min_time,
        linestyle=":",
        linewidth=1,
        label=rf"$C_t$ minimum ($t={min_time:.3f}$)",
    )

    plt.xlabel("t")
    plt.ylabel("Contribution to complexity rate")

    plt.title(
        rf"Directional Contributions to "
        rf"$\dot{{C}}_t$ "
        rf"(CFG $\omega={OMEGA}$)"
    )

    plt.legend()
    plt.tight_layout()

    figure_path_1 = os.path.join(
        FIGURE_DIR,
        "exp05_pre_directional_contributions.png",
    )

    plt.savefig(
        figure_path_1,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Figure 2:
    # Complexity + total rate
    #
    # Two separate figures are intentionally used rather than
    # mixing quantities with different units on twin axes.
    # ========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        times_np,
        complexity,
        linewidth=2,
    )

    plt.axvline(
        min_time,
        linestyle=":",
        linewidth=1,
        label=rf"minimum at $t={min_time:.3f}$",
    )

    plt.xlabel("t")
    plt.ylabel(
        r"$C_t=E[\mathrm{tr}(M_t)]$"
    )

    plt.title(
        rf"Mapping Complexity Dynamics "
        rf"(CFG $\omega={OMEGA}$)"
    )

    plt.legend()
    plt.tight_layout()

    figure_path_2 = os.path.join(
        FIGURE_DIR,
        "exp05_pre_directional_complexity.png",
    )

    plt.savefig(
        figure_path_2,
        dpi=200,
    )

    plt.close()

    # ========================================================
    # Done
    # ========================================================

    print()
    print("Saved raw results to:")
    print(output_path)

    print()
    print("Saved figures to:")
    print(figure_path_1)
    print(figure_path_2)


if __name__ == "__main__":
    main()