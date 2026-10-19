from src.flow import cfg_flow


def remap(
    x,
    omega,
    c=1,
    sigma=0.5,
    n_steps=200,
):
    """
    CFG-induced data-to-data re-mapping:

        g_omega = f_{1,0}^{(omega)} o f_{0,1}

    In the code convention:

        data at t=1
            -> latent at t=0 using the reference flow (omega=1)
            -> data at t=1 using the guided flow (omega=omega)
    """

    # Reference data -> latent
    z = cfg_flow(
        x,
        t_start=1.0,
        t_end=0.0,
        omega=1.0,
        c=c,
        sigma=sigma,
        n_steps=n_steps,
    )

    # Latent -> guided data
    x_guided = cfg_flow(
        z,
        t_start=0.0,
        t_end=1.0,
        omega=omega,
        c=c,
        sigma=sigma,
        n_steps=n_steps,
    )

    return x_guided
import torch


def remap_jacobian(
    x,
    omega,
    c=1,
    sigma=0.5,
    n_steps=200,
):
    """
    Jacobian of g_omega at a single point x.

    Input:
        x: tensor of shape (2,)

    Output:
        J: tensor of shape (2, 2)
    """

    if x.ndim != 1:
        raise ValueError(
            "remap_jacobian expects a single point with shape (2,)."
        )

    def single_remap(x_single):
        return remap(
            x_single,
            omega=omega,
            c=c,
            sigma=sigma,
            n_steps=n_steps,
        )

    return torch.func.jacrev(single_remap)(x)


def remap_operator_norm(
    x,
    omega,
    c=1,
    sigma=0.5,
    n_steps=200,
):
    """
    Largest singular value of the Jacobian:

        ||J_g(x)||_op = sigma_max(J_g(x)).
    """

    J = remap_jacobian(
        x,
        omega=omega,
        c=c,
        sigma=sigma,
        n_steps=n_steps,
    )

    singular_values = torch.linalg.svdvals(J)

    return singular_values.max()