import torch

from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess
from dlpm.methods.dlpm import LossType


class TinyEpsModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, 64)
        self.act = torch.nn.SiLU()
        self.fc2 = torch.nn.Linear(64, dim)

    def forward(self, x, t, **kwargs):
        b = x.shape[0]
        flat = x.view(b, -1)
        out = self.fc2(self.act(self.fc1(flat)))
        return out.view_as(x)


def test_pvn_mini_training_loop():
    alpha = 1.7
    beta = 0.5
    device = torch.device("cpu")
    reverse_steps = 10
    dim = 2
    batch_size = 32

    glp = GenerativeLevyProcess(
        alpha=alpha,
        device=device,
        reverse_steps=reverse_steps,
        beta=beta,
    )

    model = TinyEpsModel(dim=dim).to(device)
    models = {"default": model}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()

        x_start = torch.randn((batch_size, dim), device=device)
        out = glp.training_losses(models=models, x_start=x_start, loss_type=LossType.EPS_LOSS)
        loss = out["loss"]

        assert torch.isfinite(loss)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Check that the loss has not exploded
    final_loss = losses[-1]
    assert final_loss < 1e3
