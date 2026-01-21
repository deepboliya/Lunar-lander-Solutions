"""
Method 6: Simple controller with CMA-ES tuned parameters
Uses 6 parameters for gains and thresholds.
"""

from methods.base_controller import BaseController

MODULE_CONFIG = {
    'class_name': 'SimpleSolution3',
    'params_file': 'method_6.json',
    'weights_file': None,
    'cma_num_params': 6,
}


class SimpleSolution3(BaseController):
    def __init__(self, params=None, weights=None, gravity_magnitude=10.0):
        super().__init__(params=params, weights=weights, gravity_magnitude=gravity_magnitude)
        
        if self.params is None:
            self.params = [0.5, 0.1, 0.3, 1.0, 0.5, 5.0]

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        low_speed_height = self.params[0]
        low_speed = self.params[1]
        high_speed = self.params[2]
        x_gain = self.params[3]
        max_tilt = self.params[4]
        torque_gain = self.params[5]

        if y < low_speed_height:
            vy_target = -low_speed
        else:
            vy_target = -high_speed

        if vy < vy_target:
            thrust = 1
        else:
            thrust = 0

        target_theta = x_gain * x
        if target_theta < -max_tilt:
            target_theta = -max_tilt
        if target_theta > max_tilt:
            target_theta = max_tilt
            
        torque = -torque_gain * (target_theta - theta)

        return [thrust, torque]
