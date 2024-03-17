import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import matplotlib.pyplot as plt

# Create log dir
log_dir = "./tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = gym.make("LunarLanderContinuous-v2")
# Logs will be saved in log_dir/monitor.csv
env = Monitor(env, log_dir)

#env = gym.make("LunarLander-v2")
model = PPO("MlpPolicy", env, verbose=1)

num_steps = 500_000
model.learn(total_timesteps=num_steps, callback=ProgressBarCallback())
model.save("ppo_lunar_lander")
vec_env = model.get_env()

#del model # remove to demonstrate saving and loading
#model = PPO.load("ppo_lunar_lander")

from stable_baselines3.common import results_plotter

# Helper from the library
results_plotter.plot_results(
    [log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO LunarLander"
)

def moving_average(values, window):
   """
   Smooth values by doing a moving average
   :param values: (numpy array)
   :param window: (int)
   :return: (numpy array)
   """
   weights = np.repeat(1.0, window) / window
   return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
   """
   plot the results

   :param log_folder: (str) the save location of the results to plot
   :param title: (str) the title of the task to plot
   """
   x, y = ts2xy(load_results(log_folder), "timesteps")

   y = moving_average(y, window=50)
   # Truncate x
   x = x[len(x) - len(y) :]


   fig = plt.figure(title)
   plt.plot(x, y)
   plt.xlabel("Number of Timesteps")
   plt.ylabel("Rewards")
   plt.title(title + " Smoothed")
   plt.show()

plot_results(log_dir)

env = gym.make("LunarLanderContinuous-v2", render_mode="human")

print('Env Action Space')
print(env.action_space)


observation, info = env.reset(seed=42)
for _ in range(1000):

   action, _states = model.predict(observation, deterministic=True)
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      break
      #observation, info = env.reset()

env.close()