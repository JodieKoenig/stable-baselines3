import gym
import os
import base64
from pathlib import Path
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation_changes import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks_changes import EvalCallback, StopTrainingOnRewardThreshold

# Fake display to allow rendering
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


def evaluate_plot(model, env):
    # timestep_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=100, return_episode_rewards=True)
    mean_rewards = []
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=200)
    mean_rewards.append(mean_reward)
    return mean_rewards


# Model 1
no_total_timesteps = 1000
timesteps = 0
timestep_intervals = 100
timestep_rewards1 = []
env1 = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
model1 = PPO(policy="MultiInputPolicy", env=env1, verbose=0)
while timesteps <= no_total_timesteps:
    model1.learn(total_timesteps=timestep_intervals)
    timesteps = timesteps+timestep_intervals
    if timesteps == 100 or timesteps == 200 or timesteps == 500 or timesteps == 1000:
        print(timesteps)
    episode_rewards = evaluate_plot(model1, env1)
    timestep_rewards1.append(episode_rewards)


# Model 2
# no_total_timesteps = 10000
# timesteps = 0
# timestep_intervals = 1000
# timestep_rewards2 = np.zeros(no_total_timesteps)
# env2 = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
# model2 = PPO(policy="MultiInputPolicy", env=env1, verbose=0)
# while timesteps <= no_total_timesteps:
#     model2.learn(total_timesteps=timestep_intervals)
#     timesteps = timesteps+timestep_intervals
#     if timesteps == 1000 or timesteps == 2000 or timesteps == 5000 or timesteps == 10000:
#         print(timesteps)
#     episode_rewards = evaluate_plot(model1, env1)
#     timestep_rewards2[timesteps].append(episode_rewards)


# Plot models
x1 = np.linspace(0, len(timestep_rewards1) - 1, len(timestep_rewards1))
y1 = timestep_rewards1
# y2 = timestep_rewards2
fig, ax1 = plt.subplots()

ax1.set_xlabel('Trained Timesteps')
ax1.set_ylabel('Mean Reward over 200 evaluations', color='black')
plot_1 = ax1.plot(x1, y1, color='black')
ax1.tick_params(axis='y', labelcolor='black')

# Adding Twin Axes
# ax2 = ax1.twinx()
# ax2.set_ylabel('Success Rate', color='green')
# plot_2 = ax2.plot(x1, y2, color='green')
# ax2.tick_params(axis='y', labelcolor='green')

# Show plot
plt.show()





