import multiprocessing
import time
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
# from multiprocessing import shared_memory

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation_changes import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks_changes import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed


def cube(x, agent):
    agent.learning_rate = 10
    return agent


if __name__ == "__main__":
    pool = multiprocessing.Pool(4)
    start_time = time.perf_counter()
    env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
    agent = PPO(policy="MultiInputPolicy", env=env, verbose=1, n_steps=1024)
    processes = [pool.apply_async(cube, args=(x, agent)) for x in range(1, 4)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time - start_time} seconds")
    print(result)
