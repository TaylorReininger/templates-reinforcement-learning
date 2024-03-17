import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy

# Create log dir to store the run
log_dir = os.path.join(".", "training_runs")
os.makedirs(log_dir, exist_ok=True)


"""Create an Environment and an RL Algorithm to Train"""
# Create and wrap the environment with a Monitor
name_env = "LunarLanderContinuous-v2"
env = gym.make(name_env)
# Logs will be saved in log_dir/monitor.csv
# (this allows us to make reward curve plots later)
env = Monitor(env, log_dir)

# We use the Proximal Policy Optimization Reinforcement Learning algorithm, which is an on-policy agent.
# We can choose to either use a fully connected or CNN network architecture:
# MlpPolicy = Fully Connected
# CnnPolicy = CNN
model = PPO("MlpPolicy", env, verbose=1)

# Set the total number of steps that the RL agent will learn over
# (It starts to perform pretty well after 100k steps)
num_steps = 5000
# Train the model
# We also include an optional callback here to have training percentage display as a progress bar
model.learn(total_timesteps=num_steps, callback=ProgressBarCallback())
# Save off the trained agent
name_model = "ppo_lunar_lander"
model.save(name_model)


"""Visualize the Results of the Training Process"""
# Stable Baselines has a results plotter method that automatically leverages the data in the 
# logdir if you use the Monitor class to store results
results_plotter.plot_results(
    [log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO LunarLander Continuous"
)

# It can also be helpful to plot your own results
# Use time-series to xy function from Stable Baselines to load the data
step, reward = ts2xy(load_results(log_dir), "timesteps")

# Smooth the reward values to reduce some of the noise in the visualization
window_length = 50
weights = np.repeat(1.0, window_length) / window_length
r_smooth = np.convolve(reward, weights, "valid")

# Truncate x to be the same length as the smoothed y values
step_smooth = step[len(step) - len(r_smooth) :]

# Plot the reward curve
title = "Reward Curve"
fig = plt.figure(title)
plt.plot(step_smooth, r_smooth)
plt.xlabel("Number of Timesteps")
plt.ylabel("Reward")
plt.title(title + " Smoothed")
plt.show()


"""Perform an Evaluation of the Trained Model"""
# Remove to demonstrate saving and loading
del model
# Load the model back in
model = PPO.load(name_model)
# Create the environment again, but this time with render_mode set to human so we can see the actions
env = gym.make(name_env, render_mode="human")

# Initialize the first state/observation
observation, info = env.reset(seed=9)
# Continue to produce state-action pairs until the simulation has ended (or 1000 steps have been taken)
for _ in range(1000):

    # Use the trained agent to take actions given the observation (set deterministic to "True" to suppress exploration)
    action, _states = model.predict(observation, deterministic=True)
    # Step the environment forward with the action from the agent and get a new observation
    observation, reward, terminated, truncated, info = env.step(action)

    # When the simulation has been terminated or truncated, stop the evaluation process
    if terminated or truncated:
        break

# Close the environment object
env.close()

# Display the reward for this simulation
print("Reward: %d" % (reward))
