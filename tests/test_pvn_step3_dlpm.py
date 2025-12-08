import torch

from bem.datasets.Distributions import gen_sas, gen_sas_pvn
from dlpm.methods.dlpm import DLPM


def test_dlpm_gen_eps_symmetric_beta_zero_matches_original():
    """
    For beta = 0, DLPM must behave exactly as before: gen_eps uses 'sas'
    and its output should match a direct gen_sas call under the same seed.
    """
    alpha = 1.7
    beta = 0.0
    device = torch.device("cpu")
    diffusion_steps = 10

    dlpm = DLPM(
        alpha=alpha,
        device=device,
        diffusion_steps=diffusion_steps,
        time_spacing="linear",
        isotropic=True,
        clamp_a=None,
        clamp_eps=None,
        scale="scale_preserving",
        beta=beta,
    )

    size = (2048,)

    torch.manual_seed(777)
    eps_from_dlpm = dlpm.gen_eps.generate(size=size)

    torch.manual_seed(777)
    eps_direct = gen_sas(alpha, size=size, a=None, device=device, isotropic=True, clamp_eps=None)

    assert torch.allclose(eps_from_dlpm, eps_direct)


def test_dlpm_gen_eps_uses_pvn_when_beta_nonzero():
    """
    For beta != 0, DLPM.gen_eps should be backed by 'sas_pvn', i.e.,
    sampling from gen_eps must match a direct call to gen_sas_pvn under
    the same seed.
    """
    alpha = 1.7
    beta = 0.6
    device = torch.device("cpu")
    diffusion_steps = 10

    dlpm = DLPM(
        alpha=alpha,
        device=device,
        diffusion_steps=diffusion_steps,
        time_spacing="linear",
        isotropic=True,
        clamp_a=None,
        clamp_eps=None,
        scale="scale_preserving",
        beta=beta,
    )

    size = (4096,)

    torch.manual_seed(1234)
    eps_dlpm = dlpm.gen_eps.generate(size=size)

    torch.manual_seed(1234)
    eps_pvn = gen_sas_pvn(
        alpha=alpha,
        beta=beta,
        size=size,
        device=device,
        isotropic=True,
        clamp_eps=None,
    )

    assert torch.allclose(eps_dlpm, eps_pvn)


def test_dlpm_sample_x_t_from_xstart_skewed_runs_and_shapes():
    """
    Smoke test: ensure that sampling x_t from x_start works with skewed PVN noise.
    """
    alpha = 1.5
    beta = -0.4
    device = torch.device("cpu")
    diffusion_steps = 20

    dlpm = DLPM(
        alpha=alpha,
        device=device,
        diffusion_steps=diffusion_steps,
        time_spacing="linear",
        isotropic=True,
        clamp_a=None,
        clamp_eps=None,
        scale="scale_preserving",
        beta=beta,
    )

    batch_size = 32
    dim = 4
    x_start = torch.zeros((batch_size, dim), device=device)
    t = 5

    x_t, eps_t = dlpm.sample_x_t_from_xstart(x_start, t)
    assert x_t.shape == x_start.shape
    assert eps_t.shape == x_start.shape
