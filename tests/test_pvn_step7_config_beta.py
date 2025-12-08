import sys

import torch

import dlpm.dlpm_experiment as dlpm_exp
import script_utils
from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess


def test_init_method_by_parameter_reads_beta_from_config_dict():
    """
    Check that init_method_by_parameter reads p['dlpm']['beta'] and passes
    it to GenerativeLevyProcess (and thus down to DLPM).
    """
    p = {
        "device": "cpu",
        "method": "dlpm",
        "dlpm": {
            "alpha": 1.7,
            "beta": 0.3,
            "reverse_steps": 10,
            "rescale_timesteps": True,
            "isotropic": True,
            "mean_predict": "EPSILON",
            "var_predict": "FIXED",
            "scale": "scale_preserving",
            "input_scaling": False,
        },
        "lim": {
            "alpha": 1.7,
            "reverse_steps": 10,
            "rescale_timesteps": True,
            "isotropic": True,
        },
    }

    method = dlpm_exp.init_method_by_parameter(p)
    assert isinstance(method, GenerativeLevyProcess)

    assert hasattr(method, "beta")
    assert abs(method.beta - 0.3) < 1e-12

    assert hasattr(method, "dlpm")
    assert abs(method.dlpm.beta - 0.3) < 1e-12


def test_update_parameters_before_loading_overrides_beta_from_cli():
    """
    Check that the new --beta CLI argument can override the config's beta.
    """
    p = {
        "method": "dlpm",
        "dlpm": {
            "alpha": 1.7,
            "reverse_steps": 10,
            "rescale_timesteps": True,
            "isotropic": True,
            "mean_predict": "EPSILON",
            "var_predict": "FIXED",
            "scale": "scale_preserving",
            "input_scaling": False,
        },
    }

    old_argv = sys.argv
    try:
        sys.argv = [
            "prog",
            "--config",
            "dummy",
            "--name",
            "dummy",
            "--beta",
            "0.9",
        ]
        args = script_utils.parse_args()
    finally:
        sys.argv = old_argv

    p_updated = script_utils.update_parameters_before_loading(p, args)

    assert "beta" in p_updated["dlpm"]
    assert abs(p_updated["dlpm"]["beta"] - 0.9) < 1e-12

    p_updated["device"] = "cpu"
    p_updated["lim"] = {
        "alpha": 1.7,
        "reverse_steps": 10,
        "rescale_timesteps": True,
        "isotropic": True,
    }

    method = dlpm_exp.init_method_by_parameter(p_updated)
    assert isinstance(method, GenerativeLevyProcess)
    assert abs(method.beta - 0.9) < 1e-12
    assert abs(method.dlpm.beta - 0.9) < 1e-12


def test_init_method_defaults_beta_to_zero_when_missing():
    """
    If 'beta' is missing in the config, init_method_by_parameter should
    default to beta=0.0.
    """
    p = {
        "device": "cpu",
        "method": "dlpm",
        "dlpm": {
            "alpha": 1.7,
            "reverse_steps": 10,
            "rescale_timesteps": True,
            "isotropic": True,
            "mean_predict": "EPSILON",
            "var_predict": "FIXED",
            "scale": "scale_preserving",
            "input_scaling": False,
        },
        "lim": {
            "alpha": 1.7,
            "reverse_steps": 10,
            "rescale_timesteps": True,
            "isotropic": True,
        },
    }

    method = dlpm_exp.init_method_by_parameter(p)
    assert isinstance(method, GenerativeLevyProcess)
    assert hasattr(method, "beta")
    assert abs(method.beta - 0.0) < 1e-12
    assert abs(method.dlpm.beta - 0.0) < 1e-12
