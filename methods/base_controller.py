"""
Base controller class for all Lunar Lander controllers.
"""

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path

METHODS_DIR = Path(__file__).parent
PARAMS_DIR = METHODS_DIR / "params"
WEIGHTS_DIR = METHODS_DIR / "weights"

PARAMS_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)


class BaseController:
    
    def __init__(self, params=None, weights=None, gravity_magnitude=10.0):
        self.params = params
        self.weights = weights
        self.gravity_magnitude = gravity_magnitude
    
    def compute_action(self, observation):
        raise NotImplementedError("Subclasses must implement compute_action")
    
    @classmethod
    def load_params(cls, params_file):
        params_path = PARAMS_DIR / params_file
        param_df = pd.read_json(params_path)
        return list(np.array(param_df.iloc[0]["Params"]))
    
    @classmethod
    def save_params(cls, params, params_file):
        params_path = PARAMS_DIR / params_file
        params_df = pd.DataFrame({'Params': [list(params)]})
        params_df.to_json(params_path)
    
    @classmethod
    def load_weights(cls, weights_file):
        weights_path = WEIGHTS_DIR / weights_file
        return torch.load(weights_path, map_location='cpu')
    
    @classmethod
    def save_weights(cls, weights, weights_file):
        weights_path = WEIGHTS_DIR / weights_file
        torch.save(weights, weights_path)


def create_controller_factory(method_number, gravity_magnitude=10.0, params=None, weights=None):
    """
    Create a factory function that instantiates a controller.
    
    Args:
        method_number: Integer method number (e.g., 1 for method_1.py)
        gravity_magnitude: Gravity constant
        params: Optional parameters (overrides loading from file)
        weights: Optional weights (overrides loading from file)
    
    Returns:
        A callable that creates controller instances
    """
    import importlib.util
    
    method_name = f"method_{method_number}"
    method_file = METHODS_DIR / f"{method_name}.py"
    
    spec = importlib.util.spec_from_file_location(method_name, method_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    config = module.MODULE_CONFIG
    controller_class = getattr(module, config['class_name'])
    
    loaded_params = params
    if loaded_params is None and config.get('params_file'):
        params_path = PARAMS_DIR / config['params_file']
        if params_path.exists():
            loaded_params = BaseController.load_params(config['params_file'])
    
    loaded_weights = weights
    if loaded_weights is None and config.get('weights_file'):
        weights_path = WEIGHTS_DIR / config['weights_file']
        if weights_path.exists():
            loaded_weights = BaseController.load_weights(config['weights_file'])
    
    def factory():
        return controller_class(
            params=loaded_params,
            weights=loaded_weights,
            gravity_magnitude=gravity_magnitude
        )
    
    return factory


def get_method_config(method_number):
    """Get the MODULE_CONFIG for a method."""
    import importlib.util
    
    method_name = f"method_{method_number}"
    method_file = METHODS_DIR / f"{method_name}.py"
    
    spec = importlib.util.spec_from_file_location(method_name, method_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module.MODULE_CONFIG
