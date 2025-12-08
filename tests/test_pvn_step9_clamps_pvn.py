import torch

import bem.datasets.Distributions as dist_mod
from bem.datasets.Distributions import gen_sas_pvn
from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess


def test_gen_sas_pvn_forwards_clamp_a_to_gen_skewed_levy():
    """
    Check that gen_sas_pvn passes clamp_a down to gen_skewed_levy (P sampling).
    """
    alpha = 1.7
    beta = 0.5
    device = torch.device("cpu")
    size = (16,)
    clamp_a_val = 123.0

    # Patch gen_skewed_levy to record clamp_a
    orig_gen_skewed_levy = dist_mod.gen_skewed_levy
    recorded = {}

    def fake_gen_skewed_levy(alpha_in, size_in, device=None, isotropic=True, clamp_a=None, random_state=None):
        recorded["clamp_a"] = clamp_a
        return torch.ones(size_in, device=device)

    try:
        dist_mod.gen_skewed_levy = fake_gen_skewed_levy
        _ = gen_sas_pvn(
            alpha=alpha,
            beta=beta,
            size=size,
            device=device,
            isotropic=True,
            clamp_eps=None,
            sigma=1.0,
            clamp_v=None,
            clamp_a=clamp_a_val,
        )
    finally:
        dist_mod.gen_skewed_levy = orig_gen_skewed_levy

    assert "clamp_a" in recorded
    assert recorded["clamp_a"] == clamp_a_val


def test_gen_sas_pvn_clamp_v_limits_V_contribution():
    """
    Force V to be huge and P to be zero-ish, and ensure clamp_v limits the V contribution.
    """
    alpha = 1.7
    beta = 0.5
    device = torch.device("cpu")
    n = 1024
    size = (n,)
    clamp_v = 5.0

    orig_gen_skewed_levy = dist_mod.gen_skewed_levy
    orig_sample = dist_mod._levy_stable_pvn.sample

    def fake_gen_skewed_levy(alpha_in, size_in, device=None, isotropic=True, clamp_a=None, random_state=None):
        return torch.zeros(size_in, device=device)

    def fake_sample(alpha, beta, size, loc, scale, type, is_isotropic):
        return torch.full((size,), 1e6, dtype=type, device=device)

    try:
        dist_mod.gen_skewed_levy = fake_gen_skewed_levy
        dist_mod._levy_stable_pvn.sample = fake_sample

        eps = gen_sas_pvn(
            alpha=alpha,
            beta=beta,
            size=size,
            device=device,
            isotropic=True,
            clamp_eps=None,
            sigma=1.0,
            clamp_v=clamp_v,
            clamp_a=None,
        )

    finally:
        dist_mod.gen_skewed_levy = orig_gen_skewed_levy
        dist_mod._levy_stable_pvn.sample = orig_sample

    eps_np = eps.detach().cpu().numpy()
    theta = (torch.sign(torch.tensor(beta)) * (abs(beta) ** (1.0 / alpha))).item()
    max_allowed = abs(theta) * clamp_v + 1e-3
    assert abs(eps_np).max() <= max_allowed


def test_training_losses_pvn_passes_clamp_v_to_gen_sas_pvn():
    """
    Ensure that GenerativeLevyProcess.training_losses (PVN branch) passes clamp_v
    down to gen_sas_pvn through DLPM.gen_eps.
    """
    alpha = 1.7
    beta = 0.5
    device = torch.device("cpu")
    reverse_steps = 4
    dim = 2
    batch_size = 8
    clamp_v_val = 7.0

    glp = GenerativeLevyProcess(
        alpha=alpha,
        device=device,
        reverse_steps=reverse_steps,
        beta=beta,
    )

    class TinyModel(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = torch.nn.Linear(dim, dim)

        def forward(self, x, t, **kwargs):
            b = x.shape[0]
            flat = x.view(b, -1)
            out = self.fc(flat)
            return out.view_as(x)

    model = TinyModel(dim=dim).to(device)
    models = {"default": model}
    x_start = torch.randn((batch_size, dim), device=device)

    # Patch gen_sas_pvn to record clamp_v
    orig_gen_sas_pvn = dist_mod.gen_sas_pvn
    recorded = {}

    def fake_gen_sas_pvn(
        alpha,
        beta,
        size,
        device=None,
        isotropic=True,
        clamp_eps=None,
        sigma=1.0,
        clamp_v=None,
        clamp_a=None,
    ):
        recorded["clamp_v"] = clamp_v
        return orig_gen_sas_pvn(
            alpha=alpha,
            beta=beta,
            size=size,
            device=device,
            isotropic=isotropic,
            clamp_eps=clamp_eps,
            sigma=sigma,
            clamp_v=clamp_v,
            clamp_a=clamp_a,
        )

    try:
        dist_mod.gen_sas_pvn = fake_gen_sas_pvn
        # ensure generator uses patched function
        glp.dlpm.gen_eps.generator = fake_gen_sas_pvn

        out = glp.training_losses(
            models=models,
            x_start=x_start,
            clamp_a=20.0,
            clamp_eps=200.0,
            clamp_v=clamp_v_val,
        )
        loss = out["loss"]
        assert torch.isfinite(loss)

    finally:
        dist_mod.gen_sas_pvn = orig_gen_sas_pvn

    assert "clamp_v" in recorded
    assert recorded["clamp_v"] == clamp_v_val
