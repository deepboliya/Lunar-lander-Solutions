"""
Method 8: Simple hand-tuned controller v3
No parameters or weights needed.
"""

from methods.base_controller import BaseController

MODULE_CONFIG = {
    'class_name': 'Solution4',
    'params_file': None,
    'weights_file': None,
}


class Solution4(BaseController):
    def __init__(self, params=None, weights=None, gravity_magnitude=10.0):
        super().__init__(params=params, weights=weights, gravity_magnitude=gravity_magnitude)

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        low_speed_height = 0.5
        low_speed = 0.1
        high_speed = 0.3
        x_gain = 1.0
        max_tilt = 0.5
        torque_gain = 5.0
        x_diff = 0.2

        if y < low_speed_height:
            vy_target = -low_speed
        else:
            vy_target = -high_speed

        if vy < vy_target:
            thrust = 1
        else:
            thrust = 0

        target_theta = x_gain * x + x_diff * vx

        if target_theta < -max_tilt:
            target_theta = -max_tilt
        if target_theta > max_tilt:
            target_theta = max_tilt
            
        torque = -torque_gain * (target_theta - theta)

        return [thrust, torque]
