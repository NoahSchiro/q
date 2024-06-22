# Lunar Lander

This is my second experiment and I was suprised to find that nearly the same exact code can be used in this script as the cartpole experiment.I find this scenario a little more interesting as the physics are more complex and the observation and action space is a little more intersting.

For this model, we are only allowed to see: the coordinates of the lander in x & y, it's linear velocities in x & y, it's angle, it's angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

It's action space is discrete and is just two values: the output of it's main booster and the output of it's lateral boosters (left or right).

The model is not aware of the positioning of the flags (where it is supposed to land)

## Training

Use the `--help` flag if you want to see what options are available, but sensible defaults are in place.
```
python3 train.py
```

## Simulation with trained model

```
python3 inference.py --model ./path/to/trained/model.pt
```
