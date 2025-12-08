import os

import torch
import yaml

import bem.datasets.Distributions as dist_mod
import dlpm.dlpm_experiment as dlpm_exp
from dlpm.methods.dlpm import LossType


def _load_yaml_config(path):
    assert os.path.exists(path), f"Config file not found: {path}"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_mnist_pvn_config_has_training_clamp_v():
    """
    Ensure dlpm/configs/mnist_pvn.yml declares training.dlpm.clamp_v.
    """
    cfg = _load_yaml_config("dlpm/configs/mnist_pvn.yml")

    assert "training" in cfg, "Missing 'training' section in mnist_pvn config"
    assert "dlpm" in cfg["training"], "Missing 'training: dlpm' section in mnist_pvn config"

    train_dlpm = cfg["training"]["dlpm"]
    assert "clamp_v" in train_dlpm, "Missing 'training.dlpm.clamp_v' in mnist_pvn config"

    val = train_dlpm["clamp_v"]
    assert isinstance(val, (int, float)), f"training.dlpm.clamp_v should be numeric, got {type(val)}"


def test_mnist_pvn_config_has_eval_clamp_v():
    """
    Ensure dlpm/configs/mnist_pvn.yml declares eval.dlpm.clamp_v.
    """
    cfg = _load_yaml_config("dlpm/configs/mnist_pvn.yml")

    assert "eval" in cfg, "Missing 'eval' section in mnist_pvn config"
    assert "dlpm" in cfg["eval"], "Missing 'eval: dlpm' section in mnist_pvn config"

    eval_dlpm = cfg["eval"]["dlpm"]
    assert "clamp_v" in eval_dlpm, "Missing 'eval.dlpm.clamp_v' in mnist_pvn config"

    val = eval_dlpm["clamp_v"]
    assert isinstance(val, (int, float)), f"eval.dlpm.clamp_v should be numeric, got {type(val)}"


def test_mnist_pvn_config_applies_training_clamp_v_to_pvn_gen():
    """
    Ensure clamp_v from config is forwarded through GLP training to gen_sas_pvn.
    """
    cfg = _load_yaml_config("dlpm/configs/mnist_pvn.yml")
    cfg["device"] = "cpu"

    train_dlpm = cfg["training"]["dlpm"]
    clamp_v_cfg = train_dlpm["clamp_v"]

    method = dlpm_exp.init_method_by_parameter(cfg)

    class TinyModel(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = torch.nn.Linear(dim, dim)

        def forward(self, x, t, **kwargs):
            b = x.shape[0]
            flat = x.view(b, -1)
            out = self.fc(flat)
            return out.view_as(x)

    model = TinyModel(dim=2)
    models = {"default": model}
    x_start = torch.randn((8, 2))

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
        method.dlpm.gen_eps.generator = fake_gen_sas_pvn

        out = method.training_losses(
            models=models,
            x_start=x_start,
            loss_type=LossType.EPS_LOSS,
            clamp_a=train_dlpm.get("clamp_a"),
            clamp_eps=train_dlpm.get("clamp_eps"),
            clamp_v=clamp_v_cfg,
        )
        loss = out["loss"]
        assert torch.isfinite(loss)
    finally:
        dist_mod.gen_sas_pvn = orig_gen_sas_pvn
        method.dlpm.gen_eps.generator = orig_gen_sas_pvn

    assert recorded.get("clamp_v") == clamp_v_cfg
