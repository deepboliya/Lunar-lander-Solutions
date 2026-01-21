"""
Method 3: Simple hand-tuned controller v2
No parameters or weights needed.
"""

from methods.base_controller import BaseController

MODULE_CONFIG = {
    'class_name': 'SimpleSolution2',
    'params_file': None,
    'weights_file': None,
}


class SimpleSolution2(BaseController):
    def __init__(self, params=None, weights=None, gravity_magnitude=10.0):
        super().__init__(params=params, weights=weights, gravity_magnitude=gravity_magnitude)

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        if y < 0.4:
            vy_target = -0.05
        else:
            vy_target = -0.5

        if vy < vy_target:
            thrust = 1
        else:
            thrust = 0
            
        if abs(x) > 1.0:
            gain_x = 2.0
        else:
            gain_x = 1.0
            
        target_theta = gain_x * x
        if target_theta < -0.4:
            target_theta = -0.4
        if target_theta > 0.4:
            target_theta = 0.4
            
        torque = -3 * (target_theta - theta)

        return [thrust, torque]
