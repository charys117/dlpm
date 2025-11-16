import os
import json
import wandb
import torch
import numpy as np
from bem.Logger import Logger



''' this will manage all the information that could be contained in dictionnary like this:
p = {
    'settings' : {'lr' : ..., 'optimizer': ...},
    'model' : {},
    'data' : {},
    'eval' : {}
}
'''

class NeptuneLogger(Logger):
    """
    Thin wrapper around Weights & Biases so we don't touch the rest of the
    experiment code. WANDB_* environment variables control the run setup.
    """

    def __init__(self, p=None):
        super().__init__()
        self.run = wandb.init(**self._wandb_settings())
        if p is not None:
            self.initialize(p)
    
    def initialize(self, p):
        sanitized = self._sanitize(p)
        wandb.config.update(sanitized, allow_val_change=True)

    def set_values(self, value_dict):
        print('setting values in wandb logger')
        payload = {}

        def aux(current_str, dic):
            for k, v in dic.items():
                new_str = '/'.join([current_str, k]) if current_str else k
                if isinstance(v, dict):
                    aux(new_str, v)
                else:
                    payload[new_str] = self._prepare_value(v)

        aux('', value_dict)
        if payload:
            wandb.log(payload)

    def log(self, data_type, data):
        wandb.log({f"eval/{data_type}": self._prepare_value(data)})

    def stop(self):
        wandb.finish()

    def _wandb_settings(self):
        settings = {
            'project': os.getenv('WANDB_PROJECT', 'dlpm'),
            'entity': os.getenv('WANDB_ENTITY'),
            'dir': os.getenv('WANDB_DIR'),
            'name': os.getenv('WANDB_RUN_NAME'),
            'group': os.getenv('WANDB_RUN_GROUP')
        }
        run_id = os.getenv('WANDB_RUN_ID')
        if run_id:
            settings['id'] = run_id
            settings['resume'] = os.getenv('WANDB_RESUME', 'allow')
        return {k: v for k, v in settings.items() if v}

    def _sanitize(self, value):
        if isinstance(value, dict):
            return {k: self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(v) for v in value]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.reshape(-1)[0].item()
            return value.tolist()
        if isinstance(value, (np.generic,)):
            return value.item()
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def _prepare_value(self, value):
        if value is None:
            return 0.0
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            if value.size == 1:
                return value.reshape(-1)[0].item()
            return wandb.Histogram(value.flatten())
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return 0.0
            try:
                arr = np.asarray(value, dtype=np.float32)
            except Exception:
                return str(value)
            if arr.size == 1:
                return arr.reshape(-1)[0].item()
            return wandb.Histogram(arr.flatten())
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, (int, float, bool, str)):
            return value
        return str(value)
