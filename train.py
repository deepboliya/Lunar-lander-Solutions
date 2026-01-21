#!/usr/bin/env python3
"""
CMA-ES Training for Lunar Lander Controllers

Usage:
    python train.py --method 5 --num-gen 400 --pop-size 100
    python train.py --method 6 --num-gen 200 --pop-size 50
"""

import argparse
import sys
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gymnasium as gym
import cma

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from methods.base_controller import get_method_config, PARAMS_DIR, METHODS_DIR
import importlib.util

warnings.filterwarnings('ignore')

GRAVITY_MAGNITUDE = 10.0


def load_method_module(method_num: int):
    method_file = METHODS_DIR / f"method_{method_num}.py"
    spec = importlib.util.spec_from_file_location(f"method_{method_num}", method_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_controller_class(method_num: int):
    module = load_method_module(method_num)
    config = get_method_config(method_num)
    class_name = config.get('class_name')
    return getattr(module, class_name)


def save_params(method_num: int, params, filename: str = None):
    if filename is None:
        filename = f"method_{method_num}.json"
    
    params_path = PARAMS_DIR / filename
    params_df = pd.DataFrame({'Params': [list(params)]})
    params_df.to_json(params_path)
    print(f"Saved parameters to {params_path}")
    return params_path


def train_cma_es(method_num: int, num_gen: int, pop_size: int, sigma0: float = 2.0):
    config = get_method_config(method_num)
    num_params = config.get('cma_num_params')
    
    print(f"\nTraining method {method_num} with CMA-ES")
    print(f"  Parameters: {num_params}")
    print(f"  Generations: {num_gen}")
    print(f"  Population size: {pop_size}")
    print("-" * 50)
    
    controller_class = get_controller_class(method_num)
    env = gym.make("LunarLander-v3", continuous=True, gravity=-GRAVITY_MAGNITUDE)
    
    def fitness(params):
        total_reward = 0
        
        controller = controller_class(
            params=list(params),
            weights=None,
            gravity_magnitude=GRAVITY_MAGNITUDE
        )
        
        obs, info = env.reset(seed=random.randint(0, 10000))
        done = False
        
        while not done:
            action = controller.compute_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
        
        return -total_reward
    
    x0 = np.random.normal(0, 1, (num_params, 1))
    opts = {'maxiter': num_gen, 'popsize': pop_size}
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    
    while not es.stop():
        X = es.ask()
        fitnesses = [fitness(x) for x in X]
        es.tell(X, fitnesses)
        es.logger.add()
        es.disp()
    
    es.result_pretty()
    env.close()
    
    return es.result


def save_training_results(method_num: int, cmaes_result):
    best_params = cmaes_result[0]
    params_file = f"method_{method_num}.json"
    save_params(method_num, best_params, params_file)
    
    try:
        dat_content = [i.strip().split() for i in open("outcmaes/fit.dat").readlines()]
        
        generations = []
        evaluations = []
        bestever = []
        best = []
        median = []
        worst = []
        
        for i in range(1, len(dat_content)):
            generations.append(int(dat_content[i][0]))
            evaluations.append(int(dat_content[i][1]))
            bestever.append(-float(dat_content[i][4]))
            best.append(-float(dat_content[i][5]))
            median.append(-float(dat_content[i][6]))
            worst.append(-float(dat_content[i][7]))
        
        logs_df = pd.DataFrame()
        logs_df['Generations'] = generations
        logs_df['Evaluations'] = evaluations
        logs_df['BestEver'] = bestever
        logs_df['Best'] = best
        logs_df['Median'] = median
        logs_df['Worst'] = worst
        
        logs_path = PARAMS_DIR / f"method_{method_num}_logs.csv"
        logs_df.to_csv(logs_path)
        print(f"Saved training logs to {logs_path}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best, color='green', label='Best')
        plt.plot(generations, median, color='blue', label='Median')
        plt.xlabel("Number of generations")
        plt.ylabel("Fitness (Reward)")
        plt.legend()
        plt.title(f'Learning Curve for method {method_num}')
        plt.grid(True, alpha=0.3)
        
        curve_path = PARAMS_DIR / f"method_{method_num}_learning_curve.jpg"
        plt.savefig(curve_path)
        plt.close()
        print(f"Saved learning curve to {curve_path}")
        
    except FileNotFoundError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train Lunar Lander controllers using CMA-ES")
    
    parser.add_argument('--method', type=int, required=True,
                        help='Method number to train (must have cma_num_params in config)')
    parser.add_argument('--num-gen', type=int, default=400,
                        help='Number of generations for CMA-ES')
    parser.add_argument('--pop-size', type=int, default=100,
                        help='Population size for CMA-ES')
    parser.add_argument('--sigma', type=float, default=2.0,
                        help='Initial sigma for CMA-ES')
    
    args = parser.parse_args()
    
    config = get_method_config(args.method)
    if config.get('cma_num_params') is None:
        print(f"Error: Method {args.method} is not trainable (no cma_num_params)")
        sys.exit(1)
    
    result = train_cma_es(
        args.method,
        num_gen=args.num_gen,
        pop_size=args.pop_size,
        sigma0=args.sigma
    )
    
    save_training_results(args.method, result)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Method: {args.method}")
    print(f"  Best fitness: {-result[1]:.2f}")
    print(f"  Parameters saved to: methods/params/method_{args.method}.json")
    print("=" * 50)


if __name__ == '__main__':
    main()
