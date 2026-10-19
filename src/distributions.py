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