# Lunar Lander v3

## Evaluate

```bash
python3 main.py --method 1 --num-seeds 100

python3 main.py --method 1 --num-seeds 10 --render
```
>Note: Use "--num-seeds 1 --start-seed X --render" if you just want to check how it performs on seed X 

## DQN (Method 3)

```bash
python3 main.py --method 3 --num-seeds 100
```

## Train with CMA-ES
- Right now, i have used CMA-ES just to get gains of method 2 - the cascaded pid controller
```bash
python3 train.py --method 2 --num-gen 400 --pop-size 100
```

## Results
- Method 1 and Method 2 are okayish. Gettings the weights for method 2 using cma-es is not a good method. need to provide a starting point. Can improve the fitness function by taking multiple episodes' average to avoid noisy reward.
- Method 3 (DQN) performs well and able to get > 200 score average on 100 concurrent episodes but fails on some random seeds
- Method 4 - **The holy grail**, somehow this works for all the seeds. Read the intuition written in the class. Also these the weights used in this weren't very hard to tune too. It was very intuitive when you render and check the problems.
>Note : Needs to test these with wind/turbulence
