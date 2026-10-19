import torch

from src.distributions import class_mean, exp02_class_covariance


def interpolation_variance(t, sigma=0.5):
    """
    Variance of X_t under

        X_t = (1 - t) Z + t X,

    where

        Z ~ N(0, I)
        X | c ~ N(mu_c, sigma^2 I).

    Therefore

        Var(X_t | c)
        = ((1 - t)^2 + t^2 sigma^2) I.
    """
    return (1.0 - t) ** 2 + (t ** 2) * (sigma ** 2)


def conditional_velocity(x, t, c, sigma=0.5):
    """
    Exact marginal velocity

        v_t(x | c)
        = E[X - Z | X_t = x, c]

    for the linear interpolation

        X_t = (1 - t) Z + t X.
    """

    mu = class_mean(
        c,
        device=x.device,
        dtype=x.dtype,
    )

    var_t = interpolation_variance(t, sigma)

    coefficient = (
        t * sigma ** 2 - (1.0 - t)
    ) / var_t

    return mu + coefficient * (x - t * mu)


def component_log_prob(x, t, c, sigma=0.5):
    """
    Log-density of the intermediate conditional distribution

        p_t(x | c)

    up to an additive constant independent of c.

    Since all components have the same covariance, constants
    cancel when computing posterior responsibilities.
    """

    mu = class_mean(
        c,
        device=x.device,
        dtype=x.dtype,
    )

    mean_t = t * mu
    var_t = interpolation_variance(t, sigma)

    squared_distance = ((x - mean_t) ** 2).sum(dim=-1)

    return -0.5 * squared_distance / var_t


def responsibilities(x, t, sigma=0.5):
    """
    Compute

        r_c(x, t) = P(c | X_t = x)

    for the equally weighted two-component mixture.

    Returns a tensor of shape (..., 2).
    """

    log_prob_0 = component_log_prob(
        x, t, c=0, sigma=sigma
    )

    log_prob_1 = component_log_prob(
        x, t, c=1, sigma=sigma
    )

    logits = torch.stack(
        [log_prob_0, log_prob_1],
        dim=-1,
    )

    return torch.softmax(logits, dim=-1)


def unconditional_velocity(x, t, sigma=0.5):
    """
    Exact unconditional marginal velocity

        v_t(x)
        = sum_c P(c | X_t=x) v_t(x | c).
    """

    weights = responsibilities(x, t, sigma=sigma)

    v0 = conditional_velocity(
        x, t, c=0, sigma=sigma
    )

    v1 = conditional_velocity(
        x, t, c=1, sigma=sigma
    )

    return (
        weights[..., 0, None] * v0
        + weights[..., 1, None] * v1
    )


def cfg_velocity(x, t, omega, c=1, sigma=0.5):
    """
    Classifier-free guided marginal velocity:

        v_cfg
        = v_uncond
          + omega * (v_cond - v_uncond).

    omega = 0:
        unconditional velocity

    omega = 1:
        conditional/reference velocity
    """

    v_uncond = unconditional_velocity(
        x, t, sigma=sigma
    )

    v_cond = conditional_velocity(
        x, t, c=c, sigma=sigma
    )

    return v_uncond + omega * (v_cond - v_uncond)
# ---------------------------------------------------------
# Exp 2: General-covariance marginal velocities
# ---------------------------------------------------------

def exp02_interpolation_covariance(x, t, c):
    """
    Covariance of X_t | c for Exp 2:

        X_t = (1 - t) Z + t X

    with

        Z ~ N(0, I)
        X | c ~ N(mu_c, Sigma_c).

    Therefore

        C_{t,c}
        = (1 - t)^2 I + t^2 Sigma_c.
    """
    covariance = exp02_class_covariance(
        c,
        device=x.device,
        dtype=x.dtype,
    )

    identity = torch.eye(
        x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )

    return (
        (1.0 - t) ** 2 * identity
        + t ** 2 * covariance
    )


def exp02_conditional_velocity(x, t, c):
    """
    Exact marginal velocity for Exp 2:

        v_t(x | c)
        = mu_c
          + [t Sigma_c - (1-t) I]
            C_{t,c}^{-1}
            (x - t mu_c).
    """
    mu = class_mean(
        c,
        device=x.device,
        dtype=x.dtype,
    )

    covariance = exp02_class_covariance(
        c,
        device=x.device,
        dtype=x.dtype,
    )

    identity = torch.eye(
        x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )

    covariance_t = exp02_interpolation_covariance(
        x, t, c
    )

    matrix = (
        t * covariance
        - (1.0 - t) * identity
    )

    centered = x - t * mu

    # Solve C_{t,c} y = centered rather than explicitly
    # computing C_{t,c}^{-1}.
    solved = torch.linalg.solve(
        covariance_t,
        centered.unsqueeze(-1),
    ).squeeze(-1)

    return mu + torch.matmul(
        matrix,
        solved.unsqueeze(-1),
    ).squeeze(-1)


def exp02_component_log_prob(x, t, c):
    """
    Log-density of p_t(x | c), up to an additive constant
    shared by both classes.

    Unlike Exp 1, the class covariances are different, so the
    log-determinant term must be retained.
    """
    mu = class_mean(
        c,
        device=x.device,
        dtype=x.dtype,
    )

    covariance_t = exp02_interpolation_covariance(
        x, t, c
    )

    centered = x - t * mu

    solved = torch.linalg.solve(
        covariance_t,
        centered.unsqueeze(-1),
    ).squeeze(-1)

    quadratic = (
        centered * solved
    ).sum(dim=-1)

    logdet = torch.linalg.slogdet(
        covariance_t
    ).logabsdet

    return -0.5 * (quadratic + logdet)


def exp02_responsibilities(x, t):
    """
    Posterior class probabilities

        r_c(x,t) = P(c | X_t=x)

    for the equally weighted Exp 2 mixture.
    """
    log_prob_0 = exp02_component_log_prob(
        x, t, c=0
    )

    log_prob_1 = exp02_component_log_prob(
        x, t, c=1
    )

    logits = torch.stack(
        [log_prob_0, log_prob_1],
        dim=-1,
    )

    return torch.softmax(logits, dim=-1)


def exp02_unconditional_velocity(x, t):
    """
    Exact unconditional marginal velocity for Exp 2:

        v_t(x)
        = sum_c P(c | X_t=x) v_t(x | c).
    """
    weights = exp02_responsibilities(x, t)

    v0 = exp02_conditional_velocity(
        x, t, c=0
    )

    v1 = exp02_conditional_velocity(
        x, t, c=1
    )

    return (
        weights[..., 0, None] * v0
        + weights[..., 1, None] * v1
    )


def exp02_cfg_velocity(x, t, omega, c=1):
    """
    CFG marginal velocity for Exp 2:

        v_cfg
        = v_uncond
          + omega * (v_cond - v_uncond).

    omega = 0:
        unconditional velocity

    omega = 1:
        conditional/reference velocity
    """
    v_uncond = exp02_unconditional_velocity(
        x, t
    )

    v_cond = exp02_conditional_velocity(
        x, t, c=c
    )

    return (
        v_uncond
        + omega * (v_cond - v_uncond)
    )