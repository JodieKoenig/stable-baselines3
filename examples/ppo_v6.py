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

# Functions


def record_video(env_id, model, video_length=500, prefix='', video_folder='/Users/jodiekoenig/Documents/SkripsieVideos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env_vid = DummyVecEnv([lambda: gym.make(env_id)])
# Start the video at step=0 and record 500 steps
    eval_env_vid = VecVideoRecorder(eval_env_vid, video_folder=video_folder, record_video_trigger=lambda step: step == 0,
                                    video_length=video_length, name_prefix=prefix)

    obs = eval_env_vid.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env_vid.step(action)

# Close the video recorder
    eval_env_vid.close()


def init_models(no_models, hyperparameter_noise):
    # env1 = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
    # env2 = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
    # model1 = PPO(policy="MultiInputPolicy", env=env1, verbose=0)
    # model2 = PPO(policy="MultiInputPolicy", env=env2, verbose=0)
    for i in range(no_models):
        env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
        # envs.append(env)
        agent = PPO(policy="MultiInputPolicy", env=env, verbose=0)
        models.append(agent)

    return models, env

    # for agent in range(no_models):
    #     envs[agent] = make_vec_env("FetchReachDense-v1", n_envs=1)
    #     models[agent] = PPO("MultiInputPolicy", envs[agent], verbose=0)
    #     return models, envs


def pbt(models_array, no_models, env):
    print("pbt starting")
    train_models(models_array, no_models)
    timestep_rewards1, timestep_rewards2 = evaluate_plot(models_array, env)
    # evaluate_models(models)
    return timestep_rewards1, timestep_rewards2


def train_models(models_array, no_models_array):
    models[0].learn(total_timesteps=5000)
    models[1].learn(total_timesteps=10000)


def evaluate_plot(models_array, env):
    # timestep_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=100, return_episode_rewards=True)
    mean_reward1, std_reward1 = evaluate_policy(models_array[0], env, n_eval_episodes=200)
    timestep_rewards1.append(mean_reward1)

    mean_reward2, std_reward2 = evaluate_policy(models_array[1], env, n_eval_episodes=200)
    timestep_rewards2.append(mean_reward2)
    # print(f"mean_reward at {no_timesteps} timesteps:{mean_reward:.2f} +/- {std_reward:.2f}")
    return timestep_rewards1, timestep_rewards2
    # x1 = np.linspace(0, len(timestep_rewards)-1, len(timestep_rewards))
    # y1 = timestep_rewards
    # #
    # fig, ax = plt.subplots()
    # ax.plot(x1, y1)
    # plt.show()
    # for _ in range(1000):
    #     # renders the environment
    #     env.render()


# Fake display to allow rendering
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'

# Training env
# env_test = make_vec_env("FetchReachDense-v1", n_envs=1)
# model_test = PPO(policy="MultiInputPolicy", env=env_test, verbose=0)
# Separate untrained evaluation env
# eval_env = make_vec_env("FetchReachDense-v1", n_envs=1)

# Define variables
number_steps = 0
accum_rewards = 0
mean_ep_reward = 0
timestep_rewards1 = []
timestep_rewards2 = []
number_eval = 400
timesteps = 0
timestep_intervals = 100
no_total_timesteps = 5000
number_models = 2
models = []
envs = []

# Initialise models
# model_test.learn(total_timesteps=timestep_intervals)

models, env = init_models(number_models, False)
# print(models[0])
# models[0].learn(total_timesteps=timestep_intervals)
timestep_rewards1, timestep_rewards2 = pbt(models, number_models, env)


# Define model
# model = PPO("MultiInputPolicy", env, verbose=0)
# timestep_rewards = evaluate_plot(model, eval_env, timesteps)

# Evaluate untrained env
# mean_reward, std_reward = evaluate_policy(model_test, env_test, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# x = np.linspace(0, 40000, 1000)
# y = np.sin(x)
#
# fig, ax = plt.subplots()
# ax.plot(x, y)
# plt.show()
# video untrained model
# record_video('FetchReachDense-v1', model, video_length=500, prefix='ppo-fetchreachdense_untrained')

# Stop training when the model reaches the reward threshold
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-2, verbose=1)
# eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)

# Almost infinite number of timesteps, but the training will stop
# early as soon as the reward threshold is reached

# model.learn(int(1e10), callback=eval_callback)

# Train main env
# while timesteps <= no_total_timesteps:
#     model.learn(total_timesteps=timestep_intervals)
#     timesteps = timesteps+timestep_intervals
#     if timesteps == (100 or 1000 or 2000 or 3000 or 4000 or 5000):
#         print(timesteps)
#     timestep_rewards = evaluate_plot(model, eval_env, timesteps)
#
x1 = np.linspace(0, len(timestep_rewards1) - 1, len(timestep_rewards1))
y1 = timestep_rewards1
fig, ax = plt.subplots()
ax.plot(x1, y1)
ax.grid()
ax.set_xlabel("Trained Timesteps")
ax.set_ylabel("Mean Reward over 200 evaluations")
plt.show()

x2 = np.linspace(0, len(timestep_rewards2) - 1, len(timestep_rewards2))
y2 = timestep_rewards2
fig2, ax2 = plt.subplots()
ax.plot(x2, y2)
ax.grid()
ax.set_xlabel("Trained Timesteps")
ax.set_ylabel("Mean Reward over 200 evaluations")
plt.show()
print("plots showing")
# Evaluate the trained agent
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# video trained agent
# record_video('FetchReachDense-v1', model, video_length=500, prefix='ppo-fetchreachdense')

# show videos
# show_videos('/Users/jodiekoenig/Documents/SkripsieVideos', prefix='ppo-fetchreachdense')
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     number_steps = number_steps + 1
#     accum_rewards = accum_rewards + rewards
#     mean_ep_reward = accum_rewards / number_steps
#     print(number_steps)
#     print(accum_rewards)
#     print(mean_ep_reward)
#     env.render()

# example evaluate model
# def evaluate(model, num_episodes=100, deterministic=True):
#     """
#     Evaluate a RL agent
#     :param model: (BaseRLModel object) the RL Agent
#     :param num_episodes: (int) number of episodes to evaluate it
#     :return: (float) Mean reward for the last num_episodes
#     """
#     # This function will only work for a single Environment
#     env = model.get_env()
#     all_episode_rewards = []
#     for i in range(num_episodes):
#         timestep_rewards = []
#         done = False
#         obs = env.reset()
#         while not done:
#             # _states are only useful when using LSTM policies
#             action, _states = model.predict(obs, deterministic=deterministic)
#             # here, action, rewards and dones are arrays
#             # because we are using vectorized env
#             obs, reward, done, info = env.step(action)
#             timestep_rewards.append(reward)
#
#         all_episode_rewards.append(sum(timestep_rewards))
#
#     mean_episode_reward = np.mean(all_episode_rewards)
#     print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
#
#     return mean_episode_reward



