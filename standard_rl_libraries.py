# Citations: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

"""
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
    
"""


# In order to get the lunar lander environment to work, you will need to run "conda install -c conda-forge box2d-py"
env = gym.make("LunarLander-v2", continuous=True, gravity=-10.0, enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode='human')
model = PPO("MlpPolicy", env, verbose=1)


num_steps = 500
model.learn(total_timesteps=num_steps, log_interval=10)
model.save("ppo_lunar_lander")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_lunar_lander")

obs = vec_env.reset()
for time_step in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")