import sys

import pytest
import torch

import eval as eval_module
import bem.Experiments as Exp
import dlpm.dlpm_experiment as dlpm_exp
from dlpm.methods.GenerativeLevyProcess import GenerativeLevyProcess


def test_eval_exp_initializes_pvn_method(monkeypatch):
    """
    Check that eval.py + script_utils can:
      - load the mnist_pvn config,
      - propagate beta and clamps into the parameter dict p,
      - and build a GenerativeLevyProcess with the same beta

    We stub out Exp.Experiment, so no real training / evaluation / data loading happens.
    """

    called = {}

    # Save the original init_method_by_parameter so we can still use it
    orig_init_method_by_parameter = dlpm_exp.init_method_by_parameter

    def wrapped_init_method_by_parameter(p):
        # Record the parameter dict used to initialize the method
        called["p_from_init"] = p
        method = orig_init_method_by_parameter(p)
        called["method_from_init"] = method
        return method

    monkeypatch.setattr(dlpm_exp, "init_method_by_parameter", wrapped_init_method_by_parameter)

    class DummyEvalManager:
        def __init__(self):
            self.logger = None

        def reset(self, keep_losses=True, keep_evals=False):
            pass

    class DummyManager:
        def __init__(self):
            self.eval = DummyEvalManager()
            self.epochs = 0
            self.total_steps = 0
            self.logger = None

        def evaluate(self, evaluate_emas=True):
            pass

    class DummyUtils:
        def __init__(self, p):
            self._p = p

        def exp_hash(self, p):
            return "dummy_hash"

        def eval_hash(self, p):
            return "dummy_eval_hash"

    class DummyExperiment:
        """
        Minimal stand-in for bem.Experiments.Experiment.
        """

        def __init__(
            self,
            checkpoint_dir,
            p,
            logger,
            exp_hash,
            eval_hash,
            init_method_by_parameter,
            init_models_by_parameter,
            reset_models,
        ):
            called["checkpoint_dir"] = checkpoint_dir
            called["raw_p"] = p
            # ensure device exists for init_method_by_parameter
            p.setdefault("device", "cpu")

            self.p = p
            self.method = init_method_by_parameter(p)
            self.manager = DummyManager()
            self.utils = DummyUtils(p)

        def load(self, epoch=None):
            called["load_called"] = True

        def save(self, files, save_new_eval, curr_epoch):
            called["save_called"] = True
            called["save_files"] = files
            return "dummy_save_path"

        def prepare(self):
            called["prepare_called"] = True

        def print_parameters(self):
            called["print_parameters_called"] = True

        def terminate(self):
            called["terminate_called"] = True

    monkeypatch.setattr(Exp, "Experiment", DummyExperiment)

    old_argv = sys.argv
    try:
        sys.argv = [
            "prog",
            "--config",
            "mnist_pvn",
            "--name",
            "pvn_eval_test",
            "--method",
            "dlpm",
            "--epochs",
            "0",
            "--eval",
            "1",
        ]

        eval_module.eval_exp("dlpm/configs/")
    finally:
        sys.argv = old_argv

    assert called.get("prepare_called", False), "DummyExperiment.prepare was not called"
    assert called.get("terminate_called", False), "DummyExperiment.terminate was not called"

    assert "p_from_init" in called, "wrapped_init_method_by_parameter was not called"
    p_from_init = called["p_from_init"]

    assert "dlpm" in p_from_init, "Expected 'dlpm' section in parameters"
    assert "beta" in p_from_init["dlpm"], "Expected 'dlpm.beta' in parameters"

    beta_val = p_from_init["dlpm"]["beta"]
    assert isinstance(beta_val, float), f"dlpm.beta should be float, got {type(beta_val)}"
    assert abs(beta_val) > 0.0, "Expected non-zero beta for PVN config (mnist_pvn)"

    method = called["method_from_init"]
    assert isinstance(method, GenerativeLevyProcess), "Method is not a GenerativeLevyProcess"

    assert hasattr(method, "beta")
    assert abs(method.beta - beta_val) < 1e-12

    assert hasattr(method, "dlpm")
    assert abs(method.dlpm.beta - beta_val) < 1e-12
