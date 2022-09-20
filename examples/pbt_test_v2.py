import pickle

import gym
import os
import base64
from pathlib import Path
from IPython import display as ipythondisplay
import matplotlib.pyplot as plt
import numpy as np
import random
import torch as th
import multiprocessing
import cloudpickle
import copy

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed


# class Agents:
#     def __init__(self, env, number):
#         # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[32, 32], vf=[32, 32])])
#         # Create the agent
#         self.model = PPO(policy="MultiInputPolicy", env=env, verbose=1, n_steps=512)
#         # put hyperparameters here instead?
#         # learning rate, gamma, gae lambda?, clip range? ent coef?, vf coef?
#         # self.model.learning_rate = np.random([])
#         self.performance = []
#         self.latest_performance = -np.inf
#         self.total_timesteps = 0
#         self.number = number
#         self.success_rate = 0
#         self.env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)


class Agents:
    def __init__(self, env, number):
        self.env = env
        self.number = number
        self.latest_performance = -np.inf
        self.performance = []
        self.success_rate = 0
        self.recent_timesteps = 0
        self.trained_timesteps = 0
        self.total_timesteps = 0

        self.model = PPO(policy="MultiInputPolicy", env=env, verbose=1, n_steps=1024)
        # https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
        self.model.gamma = random.uniform(0.802, 0.988)
        self.model.gae_lambda = random.uniform(0.91, 0.99)
        self.model.learning_rate = random.uniform(0.000105, 0.0029)
        # self.model.vf_coef = random.choice([0.5, 1])
        # self.model.ent_coef = random.uniform(0, 0.01)
        # self.model.clip_range = random.choice([0.1, 0.2, 0.3])
        # self.model.target_kl = random.uniform(0.003, 0.01)


def noise(mean, std_dev):
    parameter = float(np.random.normal(mean, std_dev, 1))
    return parameter


def train(ppo_agent, processing_queue, reward_threshold_training):
    train_agent = cloudpickle.loads(ppo_agent)

    callback_on_best_training = StopTrainingOnRewardThreshold(reward_threshold=reward_threshold_training, verbose=1)
    eval_callback_training = EvalCallback(train_agent.env, callback_on_new_best=callback_on_best_training,
                                          verbose=1, eval_freq=128, n_eval_episodes=25, best_model_save_path="yes")

    train_agent.model = train_agent.model.learn(
        total_timesteps=1024, callback=eval_callback_training, reset_num_timesteps=False)
    train_agent.latest_performance, train_agent.success_rate = evaluate_policy(
        train_agent.model, train_agent.env, n_eval_episodes=25)
    train_agent.performance.append(train_agent.latest_performance)

    train_agent.total_timesteps = train_agent.model.num_timesteps
    train_agent.recent_timesteps = train_agent.total_timesteps - train_agent.trained_timesteps
    train_agent.trained_timesteps = train_agent.total_timesteps

    send = cloudpickle.dumps(train_agent)
    processing_queue.put(send)


if __name__ == '__main__':
    queue = multiprocessing.Queue()
    env_temp = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-np.inf, verbose=1)
    eval_callback = EvalCallback(env_temp, callback_on_new_best=callback_on_best, verbose=1, eval_freq=128,
                                 n_eval_episodes=50, best_model_save_path="yes")
    num_agents = 3
    agents = []
    pbt_performance = []
    pbt_timesteps = []
    current_timesteps = 0
    end_of_training = False
    flag_success = 0
    reward_threshold = -np.inf
    current_rewards = []

    for num in range(num_agents):
        agent = Agents(env=make_vec_env(env_id="FetchReachDense-v1", n_envs=1), number=num)
        agents.append(agent)
        current_mean_reward, success_rate = evaluate_policy(
            agents[num].model, agents[num].env, n_eval_episodes=50)
        agents[num].performance.append(current_mean_reward)
        print("Agent number "f"{num} "f" has current mean reward of "f"{current_mean_reward}")
        current_rewards.append(current_mean_reward)

    stored_agents = copy.deepcopy(agents)
    best_agent = copy.deepcopy(agents[current_rewards.index(max(current_rewards))])
    best_agent_env = copy.deepcopy(agents[current_rewards.index(max(current_rewards))].env)
    best_agent_number = current_rewards.index(max(current_rewards))
    processes = [multiprocessing.Process(target=train, args=(cloudpickle.dumps(agent), queue, reward_threshold))
                 for agent in agents]

    for process in processes:
        process.start()
    # for work in workers:
    #     work.join()

    # agents[0].model = agents[0].model.learn(total_timesteps=40000)
    # original_reward, original_success_rate = evaluate_policy(
    #     agents[0].model, agents[0].env, n_eval_episodes=50)
    # print(original_reward)
    # print(original_success_rate)
    # test_agent = cloudpickle.dumps(agents[0].model.policy)
    # test_agent = copy.deepcopy(test_agent)
    # agents[1].model.policy = cloudpickle.loads(test_agent)
    # copy_reward, copy_success_rate = evaluate_policy(
    #     agents[1].model, agents[1].env, n_eval_episodes=50)
    # print(copy_reward)
    # print(copy_success_rate)

    while not end_of_training:
        if flag_success == 0:
            result = queue.get()
            current_agent = cloudpickle.loads(result)
            current_agent_number = current_agent.number
            agents[current_agent_number] = current_agent
            stored_agents[current_agent_number] = copy.deepcopy(result)
            stored_agents[current_agent_number] = cloudpickle.loads(stored_agents[current_agent_number])

            if agents[current_agent_number].latest_performance >= reward_threshold:
                reward_threshold = agents[current_agent_number].latest_performance + 0.3
                print("New reward threshold of "f"{reward_threshold}"f"")
                # # best_agent = copy.deepcopy(agents[current_agent_number].model)
                # # best_agent = copy.deepcopy(current_agent)
                # best_agent = copy.deepcopy(result)
                # best_agent = cloudpickle.loads(best_agent)
                # best_agent_env = best_agent.env
                # best_agent_number = current_agent_number

            current_rewards[current_agent_number] = agents[current_agent_number].latest_performance
            best_agent = stored_agents[current_rewards.index(max(current_rewards))]
            best_agent_number = current_rewards.index(max(current_rewards))

            print("Max current reward is agent "f"{current_rewards.index(max(current_rewards))}"
                  f" on "f"{max(current_rewards)}"f" \n")
            print("Agent number "f"{current_agent_number}"f" "
                  f"has current mean reward of "f"{agents[current_agent_number].latest_performance}"f" "
                  f"after "f"{agents[current_agent_number].total_timesteps}"f" timesteps")

            if agents[current_agent_number].latest_performance <= min(current_rewards):
                # if agents[current_agent_number].latest_performance < (max(current_rewards) - 2):  # only replace if
                replace = cloudpickle.dumps(best_agent.model)
                replace = copy.deepcopy(replace)
                agents[current_agent_number].model = cloudpickle.loads(replace)
                agents[current_agent_number].model.gamma = noise(best_agent.model.gamma, 0.002)
                agents[current_agent_number].model.gae_lambda = noise(best_agent.model.gae_lambda, 0.01)
                agents[current_agent_number].model.learning_rate = noise(best_agent.model.learning_rate, 0.0001)
                print("Replaced model of agent "f"{current_agent_number}"f" "
                      f"with current best model of agent "f"{best_agent_number}")
                multiprocessing.Process(target=train, args=(cloudpickle.dumps(agents[current_agent_number]), queue,
                                                            reward_threshold)).start()
            else:
                # best_agent = copy.copy(agents[current_agent_number].model)
                # best_agent_number = current_agent_number
                # agents[current_agent_number].model.gamma = noise(best_agent.model.gamma, 0.002)
                # agents[current_agent_number].model.gae_lambda = noise(best_agent.model.gae_lambda, 0.01)
                # agents[current_agent_number].model.learning_rate = noise(best_agent.model.learning_rate, 0.0001)
                multiprocessing.Process(target=train, args=(cloudpickle.dumps(agents[current_agent_number]),
                                                            queue, reward_threshold)).start()

            best_agent_performance, best_agent_success_rate = evaluate_policy(
                best_agent.model, best_agent.env, n_eval_episodes=50)
            pbt_performance.append(best_agent_performance)

            current_timesteps = current_timesteps + agents[current_agent_number].recent_timesteps
            pbt_timesteps.append(current_timesteps)

            print("Best agent number "f"{best_agent_number} "f", success rate "f"{best_agent_success_rate} "f" "
                  f"and performance "f"{best_agent_performance}"f" after total timesteps "f"{current_timesteps}")

        else:
            best_agent_performance, best_agent_success_rate = evaluate_policy(
                best_agent.model, best_agent.env, n_eval_episodes=50)
            print("Best agent number "f"{best_agent.number} "f", success rate "f"{best_agent_success_rate} "f" "
                  f"and performance "f"{best_agent_performance}")

        if best_agent_success_rate >= 1:
            flag_success += 1
            if flag_success == 5:  # track 5 successful evaluations in a row
                end_of_training = True
        else:
            flag_success = 0

    print("done")
    print(pbt_performance)
    print(pbt_timesteps)

# >>> # In either the same shell or a new Python shell on the same machine
# >>> import numpy as np
# >>> from multiprocessing import shared_memory
# >>> # Attach to the existing shared memory block
# >>> existing_shm = shared_memory.SharedMemory(name='psm_21467_46075')
# >>> # Note that a.shape is (6,) and a.dtype is np.int64 in this example
# >>> c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
# >>> c
# array([1, 1, 2, 3, 5, 8])
# >>> c[-1] = 888
# >>> c
# array([  1,   1,   2,   3,   5, 888])
#
# >>> # Back in the first Python interactive shell, b reflects this change
# >>> b
# array([  1,   1,   2,   3,   5, 888])
#
# >>> # Clean up from within the second Python shell
# >>> del c  # Unnecessary; merely emphasizing the array is no longer used
# >>> existing_shm.close()
#
# >>> # Clean up from within the first Python shell
# >>> del b  # Unnecessary; merely emphasizing the array is no longer used
# >>> shm.close()
# >>> shm.unlink()  # Free and release the shared memory block at the very end



# def train(queue, ppo_agent, reward_threshold):
#     # print(ppo_agent.total_timesteps)
#     # print(ppo_agent.model.num_timesteps)
#     # ppo_agent.model, current_performance = ppo_agent.model.learn(
#     #     total_timesteps=500, callback=eval_callback, reset_num_timesteps=False)
#     ppo_agent.model, current_performance = ppo_agent.model.learn(
#         total_timesteps=40000, reset_num_timesteps=False)
#     ppo_agent.total_timesteps = ppo_agent.model.num_timesteps
#     # ppo_agent.model.policy.
#     # ppo_agent.model.learning_rate = 10
#     reward_threshold = 100
#     queue.put(['hello', ppo_agent.number, reward_threshold, ])
#     # return ppo_agent.model


# if __name__ == '__main__':
#
#     q = Queue()
#     continue_training = 1
#     # current_threshold = -np.inf     # worst
#     current_threshold = 0
#     env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
#     callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=current_threshold, verbose=1)
#     eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1, eval_freq=500,
#                                  n_eval_episodes=10, best_model_save_path="yes")
#
#     class Agents:
#         def __init__(self, env, number):
#             self.model = PPO(policy="MultiInputPolicy", env=env, verbose=1, n_steps=500)
#             # put hyperparameters here instead?
#             # learning rate, gamma, gae lambda?, clip range? ent coef?, vf coef?
#             # self.model.learning_rate = np.random([])
#             self.performance = []
#             self.latest_performance = -np.inf
#             self.total_timesteps = 0
#             self.number = number
#         #
#         # def myfunc(self):
#         #     print("Hello my name is " + self.name)
#
#     num_agents = int(input("The number of agents: "))
#     agents = []
#     agents_stored = []
#     processes = []
#     envs = []
#
#     for number in range(num_agents):
#         env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
#         envs.append(env)
#         agent = Agents(env=envs[number], number=number)
#         agents.append(agent)
#         agents_stored = agents
#         process = Process(target=train, args=(q, agents[number], current_threshold))
#         processes.append(process)
#         processes[number].start()
#
#     while continue_training:
#         info = q.get()
#         print(info)
#         print(processes[info[2]])
        # ppo_agent = info[2]
        # print(int(ppo_agent.total_timesteps))

        # num_agent = info[0]
        # performance = info[1]
        # print(num_agent, performance)
        # process[num_process].start()

    #
    # obs = env.reset()
    # for _ in range(1000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

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
