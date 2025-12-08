import torch

import dlpm.methods.GenerativeLevyProcess as glp_mod
from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess
from dlpm.methods.dlpm import LossType


class TinyEpsModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 16)
        self.act = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(16, dim)

    def forward(self, x, t, **kwargs):
        b = x.shape[0]
        flat = x.view(b, -1)
        out = self.fc2(self.act(self.fc1(flat)))
        return out.view_as(x)


def _run_with_patched_compute_loss_terms(alpha, beta):
    device = torch.device("cpu")
    reverse_steps = 4
    dim = 2
    batch_size = 8

    glp = GenerativeLevyProcess(
        alpha=alpha,
        device=device,
        reverse_steps=reverse_steps,
        beta=beta,
    )

    model = TinyEpsModel(dim=dim).to(device)
    models = {"default": model}
    x_start = torch.randn((batch_size, dim), device=device)

    # Patch compute_loss_terms so both branches rely on it
    orig_compute_loss_terms = glp_mod.compute_loss_terms

    def fake_compute_loss_terms(x, y, lploss):
        b = x.shape[0]
        return 2.0 * torch.ones(b, device=x.device)

    try:
        glp_mod.compute_loss_terms = fake_compute_loss_terms

        out = glp.training_losses(
            models=models,
            x_start=x_start,
            loss_type=LossType.EPS_LOSS,
            lploss=2.0,
            loss_monte_carlo="mean",
        )
        loss = out["loss"]

        assert torch.is_tensor(loss)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert abs(loss.item() - 2.0) < 1e-6

    finally:
        glp_mod.compute_loss_terms = orig_compute_loss_terms


def test_symmetric_branch_uses_compute_loss_terms():
    _run_with_patched_compute_loss_terms(alpha=1.7, beta=0.0)


def test_pvn_branch_uses_compute_loss_terms():
    _run_with_patched_compute_loss_terms(alpha=1.7, beta=0.5)
