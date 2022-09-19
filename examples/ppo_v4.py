import gym
import os
import base64
from pathlib import Path
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks_changes import EvalCallback, StopTrainingOnRewardThreshold

# Functions

# pbt()
# ready
# explore


def show_videos(video_path='', prefix=''):
  """
  Taken from https://github.com/eleurent/highway-env

  :param video_path: (str) Path to the folder containing videos
  :param prefix: (str) Filter the video, showing only the one starting with this prefix
  """
  html = []
  for mp4 in Path(video_path).glob("{}*.mp4".format(prefix)):
      video_b64 = base64.b64encode(mp4.read_bytes())
      html.append('''<video alt="{}" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{}" type="video/mp4" />
                </video>'''.format(mp4, video_b64.decode('ascii')))
  ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


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


def evaluate_plot(model, env, no_timesteps):
    # timestep_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=100, return_episode_rewards=True)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    timestep_rewards.append(mean_reward)
    # print(f"mean_reward at {no_timesteps} timesteps:{mean_reward:.2f} +/- {std_reward:.2f}")
    return timestep_rewards
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
env = make_vec_env("FetchReachDense-v1", n_envs=1)
# Separate untrained evaluation env
eval_env = make_vec_env("FetchReachDense-v1", n_envs=1)

number_steps = 0
accum_rewards = 0
mean_ep_reward = 0
timestep_rewards = []
# number_eval = 400
timesteps = 0
timestep_intervals = 200
no_total_timesteps = 40000

# Define model
model = PPO("MultiInputPolicy", env, verbose=1)
timestep_rewards = evaluate_plot(model, eval_env, timesteps)

# Evaluate untrained env
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)
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
while timesteps < no_total_timesteps:
    model.learn(total_timesteps=timestep_intervals)
    timesteps = timesteps+timestep_intervals
    if timesteps == 1000 or timesteps == 5000 or timesteps == 10000 or timesteps == 20000 or timesteps == 30000 or timesteps == 40000:
        print(timesteps)
    timestep_rewards = evaluate_plot(model, eval_env, timesteps)

x1 = np.linspace(0, len(timestep_rewards) - 1, len(timestep_rewards))
y1 = timestep_rewards
fig, ax = plt.subplots()
ax.plot(2e2 * x1, y1)
ax.grid()
ax.set_xlabel("Trained Timesteps")
ax.set_ylabel("Mean Reward over 100 evaluations")
plt.show()
print("plot showing")
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



