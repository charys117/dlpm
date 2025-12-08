import numpy as np
import torch
import scipy.stats as st

from bem.datasets.Distributions import gen_sas, gen_sas_pvn


def test_gen_sas_pvn_beta_zero_matches_gen_sas():
    """
    For beta = 0, gen_sas_pvn should reduce exactly to gen_sas.
    """
    alpha = 1.7
    size = (4096,)

    torch.manual_seed(123)
    x_base = gen_sas(alpha, size, a=None, device="cpu", isotropic=True, clamp_eps=None)

    torch.manual_seed(123)
    x_pvn = gen_sas_pvn(alpha, beta=0.0, size=size, device="cpu", isotropic=True, clamp_eps=None)

    assert torch.allclose(x_base, x_pvn)


def _sample_pvn(alpha, beta, n=20000, seed=0):
    size = (n,)
    torch.manual_seed(seed)
    x = gen_sas_pvn(alpha, beta, size=size, device="cpu", isotropic=True, clamp_eps=None)
    return x.detach().cpu().numpy()


def _sample_scipy(alpha, beta, n=20000, seed=0):
    """
    Reference sampler from SciPy in S0 parametrization:
      Y ~ S0(alpha, beta, sigma=1, mu0=lambda),
    where lambda = beta * tan(pi alpha / 2), matching the PVN construction
    with zero constant shift in the mixture.
    """
    rng = np.random.default_rng(seed=seed)
    y = st.levy_stable.rvs(alpha, beta, loc=0, scale=1, size=n, random_state=rng)
    return y


def _assert_quantiles_close(x, y, rtol=0.2, atol=0.2):
    qs = np.linspace(0.1, 0.9, 9)
    qx = np.quantile(x, qs)
    qy = np.quantile(y, qs)
    assert np.allclose(qx, qy, rtol=rtol, atol=atol)


def test_gen_sas_pvn_matches_scipy_positive_beta():
    alpha = 1.7
    beta = 0.5
    x = _sample_pvn(alpha, beta)
    y = _sample_scipy(alpha, beta)
    _assert_quantiles_close(x, y)


def test_gen_sas_pvn_matches_scipy_negative_beta():
    alpha = 1.7
    beta = -0.5
    x = _sample_pvn(alpha, beta)
    y = _sample_scipy(alpha, beta)
    _assert_quantiles_close(x, y)


def test_gen_sas_pvn_tensor_shape_and_dtype():
    alpha = 1.7
    beta = 0.3
    size = (16, 3, 3)
    x = gen_sas_pvn(alpha, beta, size=size, device="cpu", isotropic=True, clamp_eps=None)
    assert x.shape == torch.Size(size)
    assert x.dtype == torch.float32
