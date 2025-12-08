import torch

from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess
from dlpm.methods.dlpm import LossType


class TinyEpsModel(torch.nn.Module):
    """
    Minimal network that takes (x, t) and predicts eps_hat with the same shape as x.
    Used to test training_losses_dlpm with and without PVN.
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 64)
        self.act = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(64, dim)

    def forward(self, x, t, **kwargs):
        # Ignore t for simplicity; we just need a valid shape and gradient flow.
        b = x.shape[0]
        flat = x.view(b, -1)
        out = self.fc2(self.act(self.fc1(flat)))
        return out.view_as(x)


def _run_one_loss_step(alpha, beta):
    device = torch.device("cpu")
    reverse_steps = 10
    dim = 2
    batch_size = 16

    glp = GenerativeLevyProcess(
        alpha=alpha,
        device=device,
        reverse_steps=reverse_steps,
        beta=beta,
    )

    model = TinyEpsModel(dim=dim).to(device)

    x_start = torch.randn((batch_size, dim), device=device)
    models = {"default": model}

    out = glp.training_losses(
        models=models,
        x_start=x_start,
        loss_type=LossType.EPS_LOSS,
    )
    loss = out["loss"]

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()

    # Ensure some gradients flowed
    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
    assert grad_norm > 0.0


def test_training_losses_symmetric_branch_runs():
    """
    For beta = 0, we should still be able to run the original DLPM loss.
    """
    _run_one_loss_step(alpha=1.7, beta=0.0)


def test_training_losses_pvn_branch_runs():
    """
    For beta != 0, the PVN ε-loss branch should run and backpropagate.
    """
    _run_one_loss_step(alpha=1.7, beta=0.5)
