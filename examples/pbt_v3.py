import gym
import os
import base64
from pathlib import Path
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import numpy as np
import random

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks_changes import EvalCallback, StopTrainingOnRewardThreshold


def record_video(env_id, model, video_length=500, prefix='',
                 video_folder='/Users/jodiekoenig/Documents/SkripsieVideos/PBT/'):
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


# Fake display to allow rendering
os.system("Xvfb :1 -screen 0 1024x768x24 &")
os.environ['DISPLAY'] = ':1'


num_agents = int(input("The number of agents: "))
agents = []
env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
for agent in range(num_agents):
    agents.append(PPO(policy="MultiInputPolicy", env=env, verbose=1, n_steps=500))
    # make parameter noise here

population_best_model = random.choice(agents)
end_of_training = False
step_size = 2000
eval_frequency = 500
eval_episodes = 50
threshold_rate = 0.5
current_threshold = -np.inf     # worst
best_mean_reward = -np.inf      # worst
worst_mean_reward = -np.inf     # worst
overall_timesteps_per_agent = 0           # across all agents
flag_success = 0    # is the agent successful yet
pop_cumulative_mean_reward = []     # storing mean reward of ongoing main pop model
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=current_threshold, verbose=1)
eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1, eval_freq=eval_frequency,
                             n_eval_episodes=eval_episodes)
# enter pbt function:
while not end_of_training:
    if flag_success == 0:
        for agent in range(num_agents):
            print(agent)
            # step and eval (stops learning when ready)
            # whichever finishes first
            agents[agent].learn(total_timesteps=step_size, callback=eval_callback, reset_num_timesteps=False)
            overall_timesteps_per_agent[agent] = agents[agent].total_timesteps      # check this!!!

            # EVAL METHOD 1: using the best mean from the callback every learn session
            # this method does not seem to pick up best model at all times (overlaps missed)
            if eval_callback.best_mean_reward > best_mean_reward:       # save new best model and update threshold
                # this new best model is based on a single best mean
                best_mean_reward = eval_callback.best_mean_reward
                population_best_model = agents[agent]
                num_population_best_model = agent
                current_threshold = best_mean_reward + threshold_rate
                callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=current_threshold, verbose=1)
                eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1,
                                             eval_freq=eval_frequency, n_eval_episodes=eval_episodes)
            # track best model performance every training session
            pop_mean_reward, success_rate = evaluate_policy(population_best_model, env, n_eval_episodes=eval_episodes)
            pop_cumulative_mean_reward.append(pop_mean_reward)
            print(num_population_best_model)
            print(success_rate)
            print(pop_mean_reward)
            if success_rate == 1:
                break

            # EVAL METHOD 2: evaluating each agent after ready has been called and saving progress of all

    else:
        pop_mean_reward, success_rate = evaluate_policy(population_best_model, env, n_eval_episodes=eval_episodes)
        pop_cumulative_mean_reward.append(pop_mean_reward)
        print(num_population_best_model)
        print(success_rate)
        print(pop_mean_reward)

    if success_rate == 1:
        flag_success += 1
        if flag_success == 5:    # track 5 successful evaluations in a row
            end_of_training = True
    else:
        flag_success = 0

# Plot population model cumulative mean reward
x1 = np.linspace(0, len(pop_cumulative_mean_reward) - 1, len(pop_cumulative_mean_reward))*population_best_model.n_steps
y1 = pop_cumulative_mean_reward
fig, ax = plt.subplots()
fig.suptitle('Mean Reward of Population Based Best Model Until 100% Success Rate')
ax.plot(x1, y1)
fig.savefig("/Users/jodiekoenig/Documents/SkripsiePlotting/Test4_PBT/v2.pdf", dpi=150)
record_video('FetchReachDense-v1', population_best_model, video_length=500, prefix='pbt-v0')





