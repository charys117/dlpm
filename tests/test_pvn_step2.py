import torch

from bem.datasets.Distributions import gen_sas_pvn
from bem.datasets import Data


def test_generator_sas_pvn_calls_gen_sas_pvn():
    """
    Ensure that Data.Generator('sas_pvn', ...) produces the same samples as
    a direct call to gen_sas_pvn when the random seed is fixed.
    """
    alpha = 1.7
    beta = 0.4
    size = (2048,)

    # Construct generator
    gen = Data.Generator(
        'sas_pvn',
        alpha=alpha,
        beta=beta,
        device="cpu",
        isotropic=True,
        clamp_eps=None,
    )

    torch.manual_seed(321)
    x_gen = gen.generate(size=size)

    torch.manual_seed(321)
    x_direct = gen_sas_pvn(
        alpha=alpha,
        beta=beta,
        size=size,
        device="cpu",
        isotropic=True,
        clamp_eps=None,
    )

    assert torch.allclose(x_gen, x_direct)


def test_generator_sas_pvn_shape_and_dtype():
    alpha = 1.7
    beta = -0.2
    size = (32, 4, 4)

    gen = Data.Generator(
        'sas_pvn',
        alpha=alpha,
        beta=beta,
        device="cpu",
        isotropic=True,
        clamp_eps=None,
    )

    x = gen.generate(size=size)
    assert x.shape == torch.Size(size)
    assert x.dtype == torch.float32
