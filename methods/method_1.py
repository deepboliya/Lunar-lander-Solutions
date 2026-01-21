"""
Method 1: Simple hand-tuned controller
No parameters or weights needed.
"""

from methods.base_controller import BaseController

MODULE_CONFIG = {
    'class_name': 'SimpleSolution',
    'params_file': None,
    'weights_file': None,
}


class SimpleSolution(BaseController):
    def __init__(self, params=None, weights=None, gravity_magnitude=10.0):
        super().__init__(params=params, weights=weights, gravity_magnitude=gravity_magnitude)

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        if vy < -0.2:
            thrust = 1
        else:
            thrust = 0
        
        target_theta = 0.8 * x

        if target_theta < -0.4:
            target_theta = -0.4
        if target_theta > 0.4:
            target_theta = 0.4
        
        torque = -5 * (target_theta - theta)

        return [thrust, torque]
