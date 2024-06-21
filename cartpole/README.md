# Cartpole

Cartpole is like the hello world of reinforcement learning. The goal is to balance a pole on a cart. If it tips over too far or the cart moves too far from the center of the screen, then the simulation ends. However, for every frame that does not happen, the model gets an extra reward (1 per frame).

The model is allowed to know the following about the environment: the cart velocity and position and the pole angle and angular velocity. It can only act on the world by moving left or right.

## Training

Use the `--help` flag if you want to see what options are available, but sensible defaults are in place.
```
python3 train.py
```

## Simulation with trained model

```
python3 inference.py --model ./path/to/trained/model.pt
```
