import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.distributions import sample_base
from src.velocity import exp02_cfg_velocity


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 0
N_SAMPLES = 200
N_STEPS = 200

TARGET_CLASS = 1
OMEGA = 2.0

EIGENGAP_THRESHOLD = 1e-3


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

torch.manual_seed(SEED)

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

z_samples = sample_base(N_SAMPLES)

DT = 1.0 / N_STEPS

times = np.linspace(
    0.0,
    1.0,
    N_STEPS + 1,
)


# ---------------------------------------------------------
# Velocity and velocity Jacobian
# ---------------------------------------------------------

def velocity(x, t):
    """
    Exp 2 CFG velocity field.
    """
    return exp02_cfg_velocity(
        x,
        t,
        omega=OMEGA,
        c=TARGET_CLASS,
    )


def velocity_jacobian(x, t):
    """
    A_t = nabla_x v_t(x)
    """
    return torch.func.jacrev(
        lambda y: velocity(y, t)
    )(x)


# ---------------------------------------------------------
# Joint dynamics
#
# dx/dt = v_t(x)
# dJ/dt = A_t J
#
# where
#
# A_t = nabla_x v_t(x_t)
# ---------------------------------------------------------

def joint_dynamics(x, J, t):
    """
    Return derivatives of x_t and J_t.
    """

    v = velocity(
        x,
        t,
    )

    A = velocity_jacobian(
        x,
        t,
    )

    dJ = A @ J

    return v, dJ


def joint_rk4_step(x, J, t, dt):
    """
    One RK4 step for the coupled system

        dx/dt = v_t(x)
        dJ/dt = A_t J.
    """

    # k1
    k1_x, k1_J = joint_dynamics(
        x,
        J,
        t,
    )

    # k2
    k2_x, k2_J = joint_dynamics(
        x + 0.5 * dt * k1_x,
        J + 0.5 * dt * k1_J,
        t + 0.5 * dt,
    )

    # k3
    k3_x, k3_J = joint_dynamics(
        x + 0.5 * dt * k2_x,
        J + 0.5 * dt * k2_J,
        t + 0.5 * dt,
    )

    # k4
    k4_x, k4_J = joint_dynamics(
        x + dt * k3_x,
        J + dt * k3_J,
        t + dt,
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


# ---------------------------------------------------------
# Storage
# ---------------------------------------------------------

energy_fraction = np.zeros(
    (N_SAMPLES, N_STEPS + 1),
    dtype=np.float64,
)

normalized_eigengap = np.zeros(
    (N_SAMPLES, N_STEPS + 1),
    dtype=np.float64,
)

top_direction = np.zeros(
    (N_SAMPLES, N_STEPS + 1, 2),
    dtype=np.float64,
)

top_stretch_rate = np.zeros(
    (N_SAMPLES, N_STEPS + 1),
    dtype=np.float64,
)

trace_M = np.zeros(
    (N_SAMPLES, N_STEPS + 1),
    dtype=np.float64,
)

rhs_trace_dynamics = np.zeros(
    (N_SAMPLES, N_STEPS + 1),
    dtype=np.float64,
)


# ---------------------------------------------------------
# Record diagnostics at current state
# ---------------------------------------------------------

def record_state(
    sample_idx,
    time_idx,
    x_t,
    J_t,
    t,
):
    """
    Given x_t and J_t, construct

        M_t = J_t J_t^T

    and record the directional-energy diagnostics.
    """

    # -----------------------------------------------------
    # M_t = J_t J_t^T
    # -----------------------------------------------------

    M_t = J_t @ J_t.T

    # Remove tiny numerical asymmetry.
    M_t = 0.5 * (
        M_t + M_t.T
    )

    # -----------------------------------------------------
    # Eigen-decomposition
    #
    # torch.linalg.eigh returns eigenvalues
    # in ascending order.
    # -----------------------------------------------------

    eigenvalues, eigenvectors = torch.linalg.eigh(
        M_t
    )

    m2 = eigenvalues[-2]
    m1 = eigenvalues[-1]

    u1 = eigenvectors[:, -1]

    total_energy = eigenvalues.sum()

    # -----------------------------------------------------
    # Top-direction energy fraction
    #
    # rho = m1 / (m1 + m2)
    # -----------------------------------------------------

    rho = m1 / total_energy

    # -----------------------------------------------------
    # Normalized eigengap
    #
    # gamma = (m1 - m2) / (m1 + m2)
    #
    # Used only to decide whether the top direction
    # is identifiable.
    # -----------------------------------------------------

    eigengap = (
        (m1 - m2)
        / total_energy
    )

    # -----------------------------------------------------
    # Velocity Jacobian
    #
    # A_t = nabla_x v_t(x_t)
    # -----------------------------------------------------

    A_t = velocity_jacobian(
        x_t,
        t,
    )

    # -----------------------------------------------------
    # Symmetric stretching component
    #
    # S_t = (A_t + A_t^T) / 2
    # -----------------------------------------------------

    S_t = 0.5 * (
        A_t + A_t.T
    )

    # -----------------------------------------------------
    # Stretch/contraction rate along the current
    # high-energy direction
    #
    # r1 = u1^T S_t u1
    # -----------------------------------------------------

    r1 = u1 @ S_t @ u1

    # -----------------------------------------------------
    # Exact instantaneous complexity derivative
    #
    # d/dt tr(M_t)
    # = 2 tr(S_t M_t)
    # -----------------------------------------------------

    rhs = 2.0 * torch.trace(
        S_t @ M_t
    )

    # -----------------------------------------------------
    # Store
    # -----------------------------------------------------

    energy_fraction[
        sample_idx,
        time_idx,
    ] = rho.item()

    normalized_eigengap[
        sample_idx,
        time_idx,
    ] = eigengap.item()

    top_direction[
        sample_idx,
        time_idx,
    ] = u1.detach().cpu().numpy()

    top_stretch_rate[
        sample_idx,
        time_idx,
    ] = r1.item()

    trace_M[
        sample_idx,
        time_idx,
    ] = torch.trace(M_t).item()

    rhs_trace_dynamics[
        sample_idx,
        time_idx,
    ] = rhs.item()


# ---------------------------------------------------------
# Main integration
# ---------------------------------------------------------

print("Pre-Exp 5 — M_t Dynamics Diagnostic")
print("=" * 60)
print(f"Samples: {N_SAMPLES}")
print(f"RK4 steps: {N_STEPS}")
print(f"CFG omega: {OMEGA}")
print()


for sample_idx, z in enumerate(z_samples):

    if (
        sample_idx % 10 == 0
        or sample_idx == N_SAMPLES - 1
    ):
        print(
            f"Processing sample "
            f"{sample_idx + 1}/{N_SAMPLES}"
        )

    # Initial conditions:
    #
    # x_0 = z
    # J_0 = I

    x_t = z.clone()

    J_t = torch.eye(
        2,
        dtype=z.dtype,
        device=z.device,
    )

    # Record t = 0.
    record_state(
        sample_idx,
        0,
        x_t,
        J_t,
        0.0,
    )

    # Forward integration.
    for step_idx in range(N_STEPS):

        t = step_idx * DT

        x_t, J_t = joint_rk4_step(
            x_t,
            J_t,
            t,
            DT,
        )

        t_next = (
            step_idx + 1
        ) * DT

        record_state(
            sample_idx,
            step_idx + 1,
            x_t,
            J_t,
            t_next,
        )


# ---------------------------------------------------------
# Direction-rotation diagnostic
# ---------------------------------------------------------

# u and -u represent the same eigen-direction.
#
# Also, if M_t is nearly isotropic, the top eigenvector
# is not uniquely defined.
#
# Therefore rotation is measured only when the normalized
# eigengap is sufficiently large at both adjacent times.

direction_rotation = np.full(
    (N_SAMPLES, N_STEPS),
    np.nan,
    dtype=np.float64,
)

for sample_idx in range(N_SAMPLES):

    for time_idx in range(N_STEPS):

        gap_now = normalized_eigengap[
            sample_idx,
            time_idx,
        ]

        gap_next = normalized_eigengap[
            sample_idx,
            time_idx + 1,
        ]

        if (
            gap_now < EIGENGAP_THRESHOLD
            or gap_next < EIGENGAP_THRESHOLD
        ):
            continue

        u_now = top_direction[
            sample_idx,
            time_idx,
        ]

        u_next = top_direction[
            sample_idx,
            time_idx + 1,
        ]

        # Sign ambiguity:
        #
        # u and -u represent the same direction.
        cosine = np.abs(
            np.dot(
                u_now,
                u_next,
            )
        )

        cosine = np.clip(
            cosine,
            0.0,
            1.0,
        )

        direction_rotation[
            sample_idx,
            time_idx,
        ] = np.arccos(cosine)


# ---------------------------------------------------------
# Sanity check 1
#
# M_0 = I
# -> rho(0) = 1/2
# ---------------------------------------------------------

rho_initial = energy_fraction[:, 0]

print()
print("Sanity check 1: M_0 = I")

print(
    "mean rho(0) = "
    f"{rho_initial.mean():.12f}"
)

print(
    "max |rho(0) - 0.5| = "
    f"{np.max(np.abs(rho_initial - 0.5)):.3e}"
)


# ---------------------------------------------------------
# Sanity check 2
#
# d/dt tr(M_t)
# = 2 tr(S_t M_t)
#
# The left-hand side is approximated using a
# forward finite difference.
# ---------------------------------------------------------

finite_difference = (
    trace_M[:, 1:]
    - trace_M[:, :-1]
) / DT

rhs_left = rhs_trace_dynamics[:, :-1]

dynamics_abs_error = np.abs(
    finite_difference
    - rhs_left
)

print()
print(
    "Sanity check 2: "
    "d/dt tr(M_t) = 2 tr(S_t M_t)"
)

print(
    "mean absolute error = "
    f"{dynamics_abs_error.mean():.6e}"
)

print(
    "median absolute error = "
    f"{np.median(dynamics_abs_error):.6e}"
)


# ---------------------------------------------------------
# Summary statistics:
# energy concentration
# ---------------------------------------------------------

median_rho = np.median(
    energy_fraction,
    axis=0,
)

rho_q25 = np.quantile(
    energy_fraction,
    0.25,
    axis=0,
)

rho_q75 = np.quantile(
    energy_fraction,
    0.75,
    axis=0,
)


# ---------------------------------------------------------
# Summary statistics:
# direction rotation
# ---------------------------------------------------------

# Some early time points can contain only NaNs because
# M_t is nearly isotropic there.
#
# The warning is expected, so suppress it explicitly.

with warnings.catch_warnings():

    warnings.simplefilter(
        "ignore",
        category=RuntimeWarning,
    )

    median_rotation = np.nanmedian(
        direction_rotation,
        axis=0,
    )

    rotation_q25 = np.nanquantile(
        direction_rotation,
        0.25,
        axis=0,
    )

    rotation_q75 = np.nanquantile(
        direction_rotation,
        0.75,
        axis=0,
    )


# ---------------------------------------------------------
# Summary statistics:
# stretching along top-energy direction
# ---------------------------------------------------------

median_r1 = np.median(
    top_stretch_rate,
    axis=0,
)

r1_q25 = np.quantile(
    top_stretch_rate,
    0.25,
    axis=0,
)

r1_q75 = np.quantile(
    top_stretch_rate,
    0.75,
    axis=0,
)


# ---------------------------------------------------------
# Summary statistics:
# mapping complexity
#
# C_t = E_z[tr(M_t)]
#     = E_z[||J_t||_F^2]
#
# Use the mean because the mapping-complexity definition
# itself is an expectation over z.
# ---------------------------------------------------------

mean_complexity = np.mean(
    trace_M,
    axis=0,
)

complexity_q25 = np.quantile(
    trace_M,
    0.25,
    axis=0,
)

complexity_q75 = np.quantile(
    trace_M,
    0.75,
    axis=0,
)


# ---------------------------------------------------------
# Summary statistics:
# instantaneous complexity-change rate
#
# dC_t/dt
# = 2 E_z[tr(S_t M_t)]
# ---------------------------------------------------------

mean_complexity_rate = np.mean(
    rhs_trace_dynamics,
    axis=0,
)

rate_q25 = np.quantile(
    rhs_trace_dynamics,
    0.25,
    axis=0,
)

rate_q75 = np.quantile(
    rhs_trace_dynamics,
    0.75,
    axis=0,
)


# ---------------------------------------------------------
# Save raw results
# ---------------------------------------------------------

np.savez(
    "results/exp05_pre_mt_dynamics.npz",
    times=times,
    omega=OMEGA,
    energy_fraction=energy_fraction,
    normalized_eigengap=normalized_eigengap,
    top_direction=top_direction,
    direction_rotation=direction_rotation,
    top_stretch_rate=top_stretch_rate,
    trace_M=trace_M,
    rhs_trace_dynamics=rhs_trace_dynamics,
    mean_complexity=mean_complexity,
    mean_complexity_rate=mean_complexity_rate,
)


# ---------------------------------------------------------
# Figure 1:
# Energy concentration
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    times,
    median_rho,
    label="Median",
)

plt.fill_between(
    times,
    rho_q25,
    rho_q75,
    alpha=0.25,
    label="25-75% quantile",
)

plt.axhline(
    0.5,
    linestyle="--",
    label="Isotropic",
)

plt.xlabel("Time t")

plt.ylabel(
    r"$m_1/(m_1+m_2)$"
)

plt.title(
    f"Pre-Exp 5: Energy Concentration "
    f"(omega={OMEGA})"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/"
    "exp05_pre_energy_concentration.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Figure 2:
# Direction rotation
# ---------------------------------------------------------

rotation_times = times[1:]

plt.figure(figsize=(6, 4))

plt.plot(
    rotation_times,
    median_rotation,
    label="Median",
)

plt.fill_between(
    rotation_times,
    rotation_q25,
    rotation_q75,
    alpha=0.25,
    label="25-75% quantile",
)

plt.xlabel("Time t")

plt.ylabel(
    r"$\Delta \theta$ (radians)"
)

plt.title(
    f"Pre-Exp 5: Top-Direction Rotation "
    f"(omega={OMEGA})"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/"
    "exp05_pre_direction_rotation.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Figure 3:
# Stretching along high-energy direction
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    times,
    median_r1,
    label="Median",
)

plt.fill_between(
    times,
    r1_q25,
    r1_q75,
    alpha=0.25,
    label="25-75% quantile",
)

plt.axhline(
    0.0,
    linestyle="--",
)

plt.xlabel("Time t")

plt.ylabel(
    r"$u_1^\top S_t u_1$"
)

plt.title(
    f"Pre-Exp 5: Top-Direction Stretching "
    f"(omega={OMEGA})"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/"
    "exp05_pre_top_direction_stretching.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Figure 4:
# Mapping complexity over time
#
# C_t = E_z[tr(M_t)]
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    times,
    mean_complexity,
    label="Mean complexity",
)

plt.fill_between(
    times,
    complexity_q25,
    complexity_q75,
    alpha=0.25,
    label="25-75% sample quantile",
)

plt.axhline(
    2.0,
    linestyle="--",
    label=r"$C_0=d=2$",
)

plt.xlabel("Time t")

plt.ylabel(
    r"$\mathbb{E}_z[\mathrm{tr}(M_t)]$"
)

plt.title(
    f"Pre-Exp 5: Mapping Complexity Dynamics "
    f"(omega={OMEGA})"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/"
    "exp05_pre_complexity_dynamics.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Figure 5:
# Instantaneous complexity-change rate
#
# dC_t/dt
# = 2 E_z[tr(S_t M_t)]
# ---------------------------------------------------------

plt.figure(figsize=(6, 4))

plt.plot(
    times,
    mean_complexity_rate,
    label="Mean rate",
)

plt.fill_between(
    times,
    rate_q25,
    rate_q75,
    alpha=0.25,
    label="25-75% sample quantile",
)

plt.axhline(
    0.0,
    linestyle="--",
)

plt.xlabel("Time t")

plt.ylabel(
    r"$2\,\mathbb{E}_z[\mathrm{tr}(S_t M_t)]$"
)

plt.title(
    f"Pre-Exp 5: Complexity Change Rate "
    f"(omega={OMEGA})"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "results/figures/"
    "exp05_pre_complexity_rate.png",
    dpi=200,
)

plt.close()


# ---------------------------------------------------------
# Final summary
# ---------------------------------------------------------

print()
print("Mapping complexity dynamics:")

print(
    "C(0) = "
    f"{mean_complexity[0]:.6f}"
)

min_idx = np.argmin(
    mean_complexity
)

print(
    "minimum C(t) = "
    f"{mean_complexity[min_idx]:.6f} "
    f"at t={times[min_idx]:.3f}"
)

print(
    "C(1) = "
    f"{mean_complexity[-1]:.6f}"
)

print()
print("Complexity-rate range:")

print(
    "min dC/dt = "
    f"{mean_complexity_rate.min():.6f}"
)

print(
    "max dC/dt = "
    f"{mean_complexity_rate.max():.6f}"
)


# ---------------------------------------------------------
# Final energy-concentration summary
# ---------------------------------------------------------

print()
print("Final-time energy concentration:")

print(
    "median rho(1) = "
    f"{median_rho[-1]:.6f}"
)

print(
    "25-75% = "
    f"[{rho_q25[-1]:.6f}, "
    f"{rho_q75[-1]:.6f}]"
)


# ---------------------------------------------------------
# Final rotation summary
# ---------------------------------------------------------

valid_rotation = direction_rotation[
    np.isfinite(direction_rotation)
]

print()
print("Valid direction-rotation measurements:")

print(
    f"{valid_rotation.size} / "
    f"{direction_rotation.size}"
)

if valid_rotation.size > 0:

    print(
        "median = "
        f"{np.median(valid_rotation):.6e} rad"
    )

    print(
        "95th percentile = "
        f"{np.quantile(valid_rotation, 0.95):.6e} rad"
    )


# ---------------------------------------------------------
# Saved files
# ---------------------------------------------------------

print()
print("Saved raw results to:")

print(
    "results/exp05_pre_mt_dynamics.npz"
)

print()
print("Saved figures to:")

print(
    "results/figures/"
    "exp05_pre_energy_concentration.png"
)

print(
    "results/figures/"
    "exp05_pre_direction_rotation.png"
)

print(
    "results/figures/"
    "exp05_pre_top_direction_stretching.png"
)

print(
    "results/figures/"
    "exp05_pre_complexity_dynamics.png"
)

print(
    "results/figures/"
    "exp05_pre_complexity_rate.png"
)