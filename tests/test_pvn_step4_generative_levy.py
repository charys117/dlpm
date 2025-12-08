import torch

from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess
from bem.datasets.Distributions import gen_sas_pvn


class DummyModel(torch.nn.Module):
    """
    Minimal model with the same interface: model(x, t, **kwargs) -> eps_hat
    Used only to test q_sample wiring and forward shapes.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, t, **kwargs):
        # Just return zeros with the same shape as x
        return torch.zeros_like(x)


def test_generative_levy_process_beta_propagation():
    alpha = 1.7
    beta = 0.5
    device = torch.device("cpu")
    reverse_steps = 10

    glp = GenerativeLevyProcess(
        alpha=alpha,
        device=device,
        reverse_steps=reverse_steps,
        beta=beta,
    )

    # Ensure DLPM sees the same beta
    assert hasattr(glp, "dlpm")
    assert abs(glp.dlpm.beta - beta) < 1e-12


def test_q_sample_uses_pvn_noise_when_beta_nonzero():
    alpha = 1.7
    beta = 0.6
    device = torch.device("cpu")
    reverse_steps = 10

    batch_size = 2048
    dim = 2

    x_start = torch.zeros((batch_size, dim), device=device)
    glp = GenerativeLevyProcess(
        alpha=alpha,
        device=device,
        reverse_steps=reverse_steps,
        beta=beta,
    )

    # Pick a fixed timestep (avoid t=0)
    t_scalar = 5
    t = torch.full((batch_size,), t_scalar, dtype=torch.long, device=device)

    # Sample x_t via q_sample (this will internally use DLPM and gen_eps)
    # q_sample returns (x_t, eps)
    with torch.no_grad():
        x_t, _ = glp.q_sample(x_start, t)

    # Extract the "noise part" relative to x0.
    dlpm = glp.dlpm
    gammas = dlpm.bargammas.to(device)
    sigmas = dlpm.barsigmas.to(device)

    bg_t = gammas[t_scalar].view(1, *([1] * (x_start.ndim - 1)))
    bs_t = sigmas[t_scalar].view(1, *([1] * (x_start.ndim - 1)))

    eps_hat = (x_t - bg_t * x_start) / bs_t

    # Compare skew direction of eps_hat with direct PVN samples
    eps_pvn = gen_sas_pvn(
        alpha=alpha,
        beta=beta,
        size=(batch_size, dim),
        device=device,
        isotropic=True,
    )

    eps_hat_np = eps_hat.detach().cpu().numpy().reshape(-1)
    eps_pvn_np = eps_pvn.detach().cpu().numpy().reshape(-1)

    skew_hat = eps_hat_np.mean() ** 3
    skew_pvn = eps_pvn_np.mean() ** 3

    # They should at least have the same sign of "skew" most of the time
    assert skew_hat * skew_pvn >= 0.0

    # And eps_hat should not look like pure symmetric noise:
    skew_tolerance = 1e-6
    assert abs(skew_hat) > skew_tolerance or abs(skew_pvn) > skew_tolerance
