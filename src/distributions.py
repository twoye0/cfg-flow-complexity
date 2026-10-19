import torch


# ---------------------------------------------------------
# Base distribution
# ---------------------------------------------------------

def sample_base(n, device="cpu", dtype=torch.float64):
    """
    Sample from the 2D standard Gaussian:

        p_Z = N(0, I).
    """
    return torch.randn(n, 2, device=device, dtype=dtype)


# ---------------------------------------------------------
# Conditional data distribution
# ---------------------------------------------------------

def class_mean(c, device="cpu", dtype=torch.float64):
    """
    Means of the two conditional Gaussian distributions.

        c = 0: (-2, 0)
        c = 1: ( 2, 0)
    """
    means = torch.tensor(
        [[-2.0, 0.0],
         [ 2.0, 0.0]],
        device=device,
        dtype=dtype,
    )

    return means[c]


def sample_conditional(
    n,
    c,
    sigma=0.5,
    device="cpu",
    dtype=torch.float64,
):
    """
    Sample from

        p_X(x | c) = N(mu_c, sigma^2 I).
    """
    mu = class_mean(c, device=device, dtype=dtype)

    noise = torch.randn(
        n, 2,
        device=device,
        dtype=dtype,
    )

    return mu + sigma * noise


# ---------------------------------------------------------
# Unconditional data distribution
# ---------------------------------------------------------

def sample_unconditional(
    n,
    sigma=0.5,
    device="cpu",
    dtype=torch.float64,
):
    """
    Sample from the equally weighted mixture

        p_X(x)
        = 0.5 p_X(x | c=0)
        + 0.5 p_X(x | c=1).
    """

    labels = torch.randint(
        low=0,
        high=2,
        size=(n,),
        device=device,
    )

    means = torch.stack(
        [
            class_mean(0, device=device, dtype=dtype),
            class_mean(1, device=device, dtype=dtype),
        ]
    )

    selected_means = means[labels]

    noise = torch.randn(
        n, 2,
        device=device,
        dtype=dtype,
    )

    return selected_means + sigma * noise
# ---------------------------------------------------------
# Exp 2: Conditional distributions with class-dependent
# covariance geometry
# ---------------------------------------------------------

def exp02_class_covariance(c, device="cpu", dtype=torch.float64):
    """
    Class-dependent covariance matrices for Exp 2.

        c = 0:
            [[0.25, 0.00],
             [0.00, 1.00]]

        c = 1:
            [[1.00, 0.00],
             [0.00, 0.25]]

    Unlike Exp 1, the two classes now differ not only in their
    means but also in their geometry along both coordinates.
    """
    covariances = torch.tensor(
        [
            [[0.25, 0.0],
             [0.0, 1.00]],

            [[1.00, 0.0],
             [0.0, 0.25]],
        ],
        device=device,
        dtype=dtype,
    )

    return covariances[c]


def sample_conditional_exp02(
    n,
    c,
    device="cpu",
    dtype=torch.float64,
):
    """
    Sample from the Exp 2 conditional distribution

        p_X(x | c) = N(mu_c, Sigma_c).

    The means are unchanged from Exp 1, while the covariance
    matrices are class-dependent.
    """
    mu = class_mean(c, device=device, dtype=dtype)
    covariance = exp02_class_covariance(
        c,
        device=device,
        dtype=dtype,
    )

    standard_noise = torch.randn(
        n,
        2,
        device=device,
        dtype=dtype,
    )

    chol = torch.linalg.cholesky(covariance)

    return mu + standard_noise @ chol.T


def sample_unconditional_exp02(
    n,
    device="cpu",
    dtype=torch.float64,
):
    """
    Sample from the equally weighted Exp 2 mixture

        p_X(x)
        = 0.5 p_X(x | c=0)
        + 0.5 p_X(x | c=1).
    """
    labels = torch.randint(
        low=0,
        high=2,
        size=(n,),
        device=device,
    )

    samples = torch.empty(
        n,
        2,
        device=device,
        dtype=dtype,
    )

    for c in (0, 1):
        mask = labels == c
        count = int(mask.sum().item())

        if count > 0:
            samples[mask] = sample_conditional_exp02(
                count,
                c=c,
                device=device,
                dtype=dtype,
            )

    return samples