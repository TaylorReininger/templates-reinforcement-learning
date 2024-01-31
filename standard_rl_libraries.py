# Citations: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Build the cartpole environment with multiple in parallel to accelerate training
env = make_vec_env("CartPole-v1", n_envs=2)

# Create the PPO model object from SB3
rl = PPO("MlpPolicy", env, verbose=1)

# Train the rl agent
num_steps = 50000
rl.learn(total_timesteps=num_steps)
print('============== MODEL TRAINED ==============')

# Now run the agent on the environment to see how it performs
observation = env.reset()
for time_step in range(1000):
    action, state = rl.predict(observation)
    observation, rewards, dones, info = env.step(action)
    env.render("human")