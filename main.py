#!/usr/bin/env python3
"""
Lunar Lander Controller Evaluation

Usage:
    python main.py --method 1 --num-seeds 100
    python main.py --method 7 --discrete --render
"""

import argparse
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from methods.base_controller import create_controller_factory, get_method_config

GRAVITY_MAGNITUDE = 10.0


def run_episode(controller_factory, seed, render=False, continuous=True):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", continuous=continuous, gravity=-GRAVITY_MAGNITUDE, render_mode=render_mode,
                    enable_wind=False, wind_power=10.0, turbulence_power=0.0)
    observation, info = env.reset(seed=seed)

    controller = controller_factory()

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = controller.compute_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_over = terminated or truncated

    env.close()
    return total_reward


def evaluate(controller_factory, seeds, render_first=False, continuous=True):
    rewards = []

    for i, seed in enumerate(seeds):
        render = render_first and (i == 0)
        reward = run_episode(controller_factory, seed, render=render, continuous=continuous)
        rewards.append(reward)
        print(f"  Seed {seed:4d}: reward = {reward:8.2f}")

    rewards = np.array(rewards)
    print("--Evaluation Results--")
    print(f"Episodes:\t{len(rewards)}")
    print(f"Mean:\t{np.mean(rewards):8.2f}")
    print(f"Median:\t{np.median(rewards):8.2f}")
    print(f"Std deviation:{np.std(rewards):8.2f}")
    print(f"Min:{np.min(rewards):8.2f}")
    print(f"Max:{np.max(rewards):8.2f}")

    bad_seeds = [(seed, reward) for seed, reward in zip(seeds, rewards) if reward < 200]
    if bad_seeds:
        print(f"Seeds with reward < 200 ({len(bad_seeds)}/{len(seeds)}):")
        for seed, reward in bad_seeds:
            print(f"    Seed {seed:4d}: {reward:8.2f}")
    else:
        print("All seeds achieved reward with >= 200")

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Run LunarLander controller evaluation")
    
    parser.add_argument('--method', type=int, required=True,
                        help='Method number to run (e.g., 1, 2, 3, ...)')
    parser.add_argument('--num-seeds', type=int, default=100,
                        help='Number of seeds to evaluate')
    parser.add_argument('--start-seed', type=int, default=0,
                        help='Starting seed value')
    parser.add_argument('--render', action='store_true',
                        help='Render the first episode')
    parser.add_argument('--discrete', action='store_true',
                        help='Use discrete action space (default is continuous)')
    
    args = parser.parse_args()
    
    config = get_method_config(args.method)
    if config.get('discrete_action', False) and not args.discrete:
        print(f"Method {args.method} uses discrete action space, enabling --discrete automatically")
        args.discrete = True
    
    continuous = not args.discrete
    
    controller_factory = create_controller_factory(args.method, gravity_magnitude=GRAVITY_MAGNITUDE)
    
    seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    print(f"\nEvaluating method {args.method} with {len(seeds)} seeds: {seeds[0]} to {seeds[-1]}")
    print(f"Continuous: {continuous}")
    print("-" * 50)
    
    evaluate(controller_factory, seeds, render_first=args.render, continuous=continuous)


if __name__ == '__main__':
    main()
