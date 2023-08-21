# Policy Iteration

This project implements policy iteration to find the optimal trajectory for a robot in a grid world with obstacles and rewards. The robot can move in four directions (north, east, west, south) with some stochasticity. The goal is to maximize the expected discounted reward over an infinite horizon.

## Requirements

- Python 3.8 or higher
- Numpy
- Matplotlib

## Usage

- To run the policy iteration algorithm, execute `python policy_iteration.py`. This will generate four plots for the first four iterations, showing the value function and the feedback control for each state.
- To change the parameters of the grid world, such as the size, the obstacles, the rewards, and the transition probabilities, modify the `grid_world.py` file.

## Results

The policy iteration algorithm converges to a unique optimal policy after six iterations. The optimal policy guides the robot to avoid obstacles and reach the goal state with the highest reward. The value function reflects the expected discounted reward from each state under the optimal policy.

The following figure shows the optimal policy and value function after six iterations:

![Optimal policy and value function](.\PolicyIteration.gif)

## Solution Approach

### 1. Map and Transition Matrix Creation

The map given as shown below. I created an array of the map size 10x10. I assigned values for free cells as -1, obstacles as -10, and the goal cell as 10. The plot of my map is below.

I have created a function to create the transition matrix based on the current policy.

![Given Map](.\01_Report\Map.png)

![Created Map](.\01_Report\PolicyIteration_Map.png)

### 2. Policy Evaluation

The policy evaluation is done by

$$  
J^{(i+1)}(x) = q(x, u^{(i)}(x)) + \gamma E_\epsilon[J^{(i)}(f(x, u^{(i)}(x) + \epsilon))]
$$

Since this system is an MDP. This equation reduces to

$$
J^\pi = q_u + \gamma T J^\pi
$$

$$
J^\pi = (I - \gamma T)^{-1} q_u
$$

The plot after the first iteration is shown below.

![Caption](.\01_Report\Jpi_0.png)

### 3. Policy Iteration

In each iteration, the policy improvement was done after the policy evaluation step. The policy improvement is done as:

$$
u^{(k+1)}(x) = \arg\min_{u \in U} E_\epsilon[q(x, u) + \gamma J^\pi_{(k)}(f(x, u) + \epsilon)]
$$

I have attached the iterations for the cost and reward structure mentioned in the homework below.

![Policy Improvement with Goal Reward 10](.\01_Report\PolicyImprovement10reward.png)

I noticed that the policy doesn’t necessarily ever converge. I increased the reward to the goal and observed the policy converged with the goal.

![Policy Improvement with Goal Reward 100](.\01_Report\PolicyImprov1000.png)
