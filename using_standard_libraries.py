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
LOG_DIR = os.path.join(".", "training_runs")
os.makedirs(LOG_DIR, exist_ok=True)
# Specify the gym environment we wish to use
NAME_ENV = "LunarLanderContinuous-v2"

def train_agent(name_model, num_steps=1_000_000):
   
   """Create an Environment and an RL Algorithm to Train"""
   # Create and wrap the environment with a Monitor
   env = gym.make(NAME_ENV)
   # Logs will be saved in log_dir/monitor.csv
   # (this allows us to make reward curve plots later)
   env = Monitor(env, LOG_DIR)

   # We use the Proximal Policy Optimization Reinforcement Learning algorithm, which is an on-policy agent.
   # We can choose to either use a fully connected or CNN network architecture:
   # MlpPolicy = Fully Connected
   # CnnPolicy = CNN
   model = PPO("MlpPolicy", env, verbose=1)

   # Train the model
   # We also include an optional callback here to have training percentage display as a progress bar
   model.learn(total_timesteps=num_steps, callback=ProgressBarCallback())
   # Save off the trained agent
   
   model.save(name_model)


   """Visualize the Results of the Training Process"""
   # Stable Baselines has a results plotter method that automatically leverages the data in the 
   # logdir if you use the Monitor class to store results
   results_plotter.plot_results(
      [LOG_DIR], 1e5, results_plotter.X_TIMESTEPS, "PPO LunarLander Continuous"
   )

   # It can also be helpful to plot your own results
   # Use time-series to xy function from Stable Baselines to load the data
   step, reward = ts2xy(load_results(LOG_DIR), "timesteps")

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


def play_scenario(name_model):

   """Perform an Evaluation of the Trained Model"""
   # Load the model back in
   model = PPO.load(name_model)
   # Create the environment again, but this time with render_mode set to human so we can see the actions
   env = gym.make(NAME_ENV, render_mode="human")

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


def evaluate_agent(name_model):

   # Load the model back in
   model = PPO.load(name_model)
   # Make the environment, this time using rbg array rendering so we can save video off
   env = gym.make(NAME_ENV, render_mode="rgb_array")

   # Wrap the environment with a RecordVideo class to capture frames into videos
   def make_vid_logic(index):
      return True
   env = gym.wrappers.RecordVideo(env, video_folder=LOG_DIR, name_prefix="rl_vid", episode_trigger=make_vid_logic)

   # Run the simulation 5 times, saving off videos for each
   for index_eval in range(5):
      # Initialize a reward sum value
      episodic_reward = 0
      # Initialize the first state/observation
      observation, info = env.reset(seed=index_eval)
      # Continue to produce state-action pairs until the simulation has ended (or 1000 steps have been taken)
      while True: 
         env.render()
         # Use the trained agent to take actions given the observation (set deterministic to "True" to suppress exploration)
         action, _states = model.predict(observation, deterministic=True)
         # Step the environment forward with the action from the agent and get a new observation
         observation, reward, terminated, truncated, info = env.step(action)
         episodic_reward += reward
         # When the simulation has been terminated or truncated, stop the evaluation process
         if terminated or truncated:
            # Display the reward for this simulation
            print("Reward: %d" % (episodic_reward))
            break

   # Close the environment object
   env.close()


if __name__ == "__main__":

   # Choose a name for your trained model to be saved under
   name_model = "ppo_lunar_lander"
   # Set the number of training steps to train for
   num_steps=1_000_000
   # Train the agent
   train_agent(name_model, num_steps=num_steps)

   # Play the scenario so we can 
   play_scenario(name_model)
   
   # Run the agent several times and save off videos of the result
   evaluate_agent(name_model)






