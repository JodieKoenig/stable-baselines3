import gym
import os
import base64
from pathlib import Path
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import numpy as np
import random
import cloudpickle

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation_changes import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks_changes import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed


#   from stablebaselines3 examples
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    env_id = "FetchReachDense-v1"
    num_cpu = 2  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    current_performance = 0

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=25_000)

    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

# # Functions
#
# #   REFERENCE!!
# def record_video(env_id, model, video_length=500, prefix='',
#                  video_folder='/Users/jodiekoenig/Documents/SkripsieVideos/PBT/'):
#     """
#     :param env_id: (str)
#     :param model: (RL model)
#     :param video_length: (int)
#     :param prefix: (str)
#     :param video_folder: (str)
#     """
#     eval_env_vid = DummyVecEnv([lambda: gym.make(env_id)])
# # Start the video at step=0 and record 500 steps
#     eval_env_vid = VecVideoRecorder(eval_env_vid, video_folder=video_folder, record_video_trigger=lambda step: step == 0,
#                                     video_length=video_length, name_prefix=prefix)
#
#     obs = eval_env_vid.reset()
#     for _ in range(video_length):
#         action, _ = model.predict(obs)  # end episode inside here instead ?
#         obs, _, _, info = eval_env_vid.step(action)
#         # if info['is_success'] == 1      # end episode here if is success is 1 ?
#         #     break
#
# # Close the video recorder
#     eval_env_vid.close()
#
#
# # Fake display to allow rendering (REFERENCE!!)
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'
#
#
# # create class of agents
# class Agents:
#     def __init__(self):
#         self.model = PPO(policy="MultiInputPolicy", env=env, verbose=1, n_steps=500)
#         # put hyperparameters here instead?
#         # learning rate, gamma, gae lambda?, clip range? ent coef?, vf coef?
#         # self.model.learning_rate = np.random([])
#         self.performance = []
#         self.latest_performance = -np.inf
#         self.total_agent_timesteps = 0
#         self.env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
#     #
#     # def myfunc(self):
#     #     print("Hello my name is " + self.name)
#
#
# # List constants here
#
# num_agents = int(input("The number of agents: "))
# agents = []
# env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
# for agent in range(num_agents):
#     agent = Agents()
#     agents.append(agent)
#     # make hyperparameter noise here ??
#
# num_population_best_model = random.choice(range(num_agents))
# num_population_worst_model = random.choice(range(num_agents))
# population_best_model = agents[num_population_best_model]
# population_worst_model = agents[num_population_worst_model]
# best_mean_reward = -np.inf      # compared to worst
# worst_mean_reward = 0           # compared to best
#
# end_of_training = False
# step_size = 2000
# eval_frequency = 500
# eval_episodes = 50
# current_performance = 0         # performance storer for current return of learn
# session_performance = []
# best_mean_session_performance = -np.inf
# threshold_rate = 0.5
# current_threshold = -np.inf     # worst
# overall_timesteps = 0           # across all agents
# flag_success = 0    # is the agent successful yet
# pop_cumulative_mean_reward = []     # storing mean reward of ongoing main pop model
# pop_freq_performance = []
# timestep_array = []
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=current_threshold, verbose=1)
# eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1, eval_freq=eval_frequency,
#                              n_eval_episodes=eval_episodes, best_model_save_path="yes")
# total_steps = 0
# # enter pbt function:
# while not end_of_training:
#     if flag_success == 0:
#         for num in range(num_agents):
#             # overall_timesteps = 0
#             print(num)
#             session_performance = []
#             # step and eval (stops learning when ready)
#             # whichever finishes first
#             agents[num].model, current_performance = agents[num].model.learn(
#                 total_timesteps=step_size, callback=eval_callback, reset_num_timesteps=False)
#             agents[num].total_agent_timesteps = agents[num].model.num_timesteps
#             agents[num].performance.append(current_performance)
#             agents[num].latest_performance = current_performance
#
#             if np.mean(agents[num].latest_performance) < worst_mean_reward:
#                 num_population_worst_model = num
#                 population_worst_model = agents[num_population_worst_model]
#                 worst_mean_reward = np.mean(agents[num].latest_performance)
#
#             if eval_callback.best_mean_reward > best_mean_reward:
#                 best_mean_reward = eval_callback.best_mean_reward
#                 current_threshold = best_mean_reward + threshold_rate
#                 callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=current_threshold, verbose=1)
#                 eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1,
#                                              eval_freq=eval_frequency, n_eval_episodes=eval_episodes)
#
#             if np.mean(current_performance) > best_mean_session_performance:
#                 num_population_best_model = num
#                 best_mean_session_performance = np.mean(current_performance)
#
#             overall_timesteps += agents[num].total_agent_timesteps
#
#             # # EVAL METHOD 1: using the best mean from the callback every learn session
#             # # this method does not seem to pick up best model at all times (overlaps missed)
#             # if eval_callback.best_mean_reward > best_mean_reward:       # save new best model and update threshold
#             #     # this new best model is based on a single best mean
#             #     best_mean_reward = eval_callback.best_mean_reward
#             #     population_best_model = agents[agent]
#             #     num_population_best_model = agent
#             #     current_threshold = best_mean_reward + threshold_rate
#             #     callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=current_threshold, verbose=1)
#             #     eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1,
#             #                                  eval_freq=eval_frequency, n_eval_episodes=eval_episodes)
#
#             # EVAL METHOD 2: evaluating each agent after ready has been called and saving progress of all
#         # np.mean(agents[1].latest_performance)
#
#         # track best model performance every training session
#         pop_mean_reward, success_rate = evaluate_policy(
#             agents[num_population_best_model].model, env, n_eval_episodes=eval_episodes)
#         pop_cumulative_mean_reward.append(pop_mean_reward)
#         timestep_array.append(overall_timesteps)
#         overall_timesteps = 0
#         print(num_population_best_model)
#         print(success_rate)
#         print(pop_mean_reward)
#         # if success_rate == 1:
#         #     break
#
#         # explore and exploit function (once all agents are ready)
#         worst_mean_reward = 0  # refresh every iteration to only act on the latest performance
#         best_mean_session_performance = -np.inf
#         agents[num_population_worst_model].model.policy = agents[num_population_best_model].model.policy
#         # changing weights and hyperparemeters , anything else?
#         print("Replaced model policy of "f"{num_population_worst_model} "
#               f"with model policy of "f"{num_population_best_model}")
#
#     else:
#         pop_mean_reward, success_rate = evaluate_policy(
#             agents[num_population_best_model].model, env, n_eval_episodes=eval_episodes)
#         # pop_cumulative_mean_reward.append(pop_mean_reward)
#         print(num_population_best_model)
#         print(success_rate)
#         print(pop_mean_reward)
#
#     if success_rate == 1:
#         flag_success += 1
#         if flag_success == 5:    # track 5 successful evaluations in a row
#             end_of_training = True
#     else:
#         flag_success = 0
#
# # Plot population model cumulative mean reward
# print(overall_timesteps)
# # x1 = np.linspace(0, len(pop_cumulative_mean_reward) - 1, len(pop_cumulative_mean_reward)) # 2500, 6000, timestep_array
# x1 = timestep_array
# y1 = pop_cumulative_mean_reward
# fig, ax = plt.subplots()
# fig.suptitle('Mean Reward of Population Based Best Model Until 100% Success Rate')
# ax.plot(x1, y1)
# fig.savefig("/Users/jodiekoenig/Documents/SkripsiePlotting/Test4_PBT/v5.pdf", dpi=150)
# record_video('FetchReachDense-v1', agents[num_population_best_model].model, video_length=500, prefix='pbt-v5')
#
