# Histogram Filter for Robot Localization

This code implements a Bayes filter (also known as a histogram filter) to keep track of a robot's current position in a 2D grid world. The robot is equipped with a noisy odometer and a noisy color sensor. Given a stream of actions and corresponding observations, the code updates the robot's belief state using the Bayes rule.

## Requirements

- Python 3
- NumPy
- Matplotlib

## Usage

To run the code, simply execute the following command:

```bash
python histogram_filter.py
```
The code will load the data from the starter.npz file, which contains a binary color-map (the grid), a sequence of actions, a sequence of observations, and a sequence of the correct belief states3. The code will then perform the Bayes filter algorithm and plot the results.

## Output

The code will output the following plots:
 - State belief distribution at each time instance

The resulting gif is here below:

![HistogramFilter](./HistogramFilter.gif)

References
This code is based on the problem 1 of the [homework 1](./hw1.pdf) from ESE 650: Learning in Robotics course at University of Pennsylvania.
