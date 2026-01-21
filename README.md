# Lunar Lander Assignment

Modular framework for developing and evaluating Lunar Lander controllers.

## Directory Structure

```
lunar-lander-assignment/
├── main.py              # Evaluation script
├── train.py             # CMA-ES training script
├── evaluate.py          # Evaluation utilities
├── methods/             # Controller methods
│   ├── __init__.py      # Method discovery and loading
│   ├── method_1.py      # Simple controller
│   ├── method_2.py      # PID with CMA-ES params
│   ├── method_3.py      # Simple controller v2
│   ├── method_4.py      # PID simplified
│   ├── method_5.py      # PID with landing params
│   ├── method_6.py      # Simple with CMA-ES
│   ├── method_7.py      # DQN agent
│   ├── method_8.py      # Simple controller v3
│   ├── method_template.py  # Template for new methods
│   ├── params/          # CMA-ES parameters (JSON)
│   │   ├── method_2.json
│   │   ├── method_4.json
│   │   ├── method_5.json
│   │   └── method_6.json
│   └── weights/         # Neural network weights
│       └── method_7_best.pth
└── ...
```

## Quick Start

### List available methods
```bash
python3 main.py --list
```

### Evaluate a method
```bash
# Basic evaluation
python3 main.py --method method_1 --num-seeds 100

# With rendering
python3 main.py --method method_1 --num-seeds 10 --render

# Discrete action space (for DQN)
python3 main.py --method method_7 --discrete --num-seeds 100
```

### Train a method with CMA-ES
```bash
# List trainable methods
python3 train.py --list

# Train a method
python3 train.py --method method_5 --num-gen 400 --pop-size 100
```

## Adding a New Method

1. **Copy the template:**
   ```bash
   cp methods/method_template.py methods/method_9.py
   ```

2. **Edit the configuration:**
   ```python
   MODULE_CONFIG = {
       'class_name': 'MyController',      # Your controller class name
       'params_file': 'method_9.json',    # For CMA-ES parameters (optional)
       'weights_file': None,              # For NN weights (optional)
       'needs_gravity': True,             # If needs gravity_magnitude
       'needs_print': True,               # If needs print_ flag
       'cma_num_params': 6,               # Number of CMA-ES params (optional)
   }
   ```

3. **Implement your controller:**
   ```python
   class MyController:
       def __init__(self, flattened_params=None, gravity_magnitude=None, print_=False):
           # Initialize your controller
           pass
       
       def compute_action(self, observation):
           # Return [thrust, torque] for continuous
           # Return action_index (0-3) for discrete
           return [thrust, torque]
   ```

4. **Run:**
   ```bash
   # Evaluate
   python3 main.py --method method_9 --num-seeds 100

   # Train (if cma_num_params is set)
   python3 train.py --method method_9 --num-gen 400 --pop-size 100
   ```

## MODULE_CONFIG Options

| Key | Type | Description |
|-----|------|-------------|
| `class_name` | str | **Required.** Name of the controller class |
| `params_file` | str | JSON file in `methods/params/` with CMA-ES parameters |
| `weights_file` | str | Weights file in `methods/weights/` for NN methods |
| `needs_gravity` | bool | Pass `gravity_magnitude` to `__init__` |
| `needs_print` | bool | Pass `print_` flag to `__init__` |
| `params_arg` | str | Name of params argument (default: `flattened_params`) |
| `cma_num_params` | int | Number of parameters for CMA-ES training |
| `discrete_action` | bool | Use discrete action space |
| `use_load_for_eval` | bool | Use `load_for_eval()` classmethod (for NN methods) |

## Observation Space

```python
observation = [
    x,        # Horizontal position
    y,        # Vertical position
    vx,       # Horizontal velocity
    vy,       # Vertical velocity
    theta,    # Angle
    vtheta,   # Angular velocity
    leg1,     # Left leg contact (bool)
    leg2,     # Right leg contact (bool)
]
```

## Action Space

**Continuous:**
```python
action = [thrust, torque]  # thrust ∈ [0,1], torque ∈ [-1,1]
```

**Discrete:**
```python
action = 0  # Do nothing
action = 1  # Fire left engine
action = 2  # Fire main engine
action = 3  # Fire right engine
```
