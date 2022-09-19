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
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks_changes import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed
from multiprocessing import Process, Manager


class Tester:
    num = 0.0
    name = 'none'

    def __init__(self, tnum=num, tname=name):
        self.num = tnum
        self.name = tname
        # self.env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
        # policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[16, 16], vf=[16, 16])])
        # Create the agent
        self.model = PPO(policy="MultiInputPolicy", env=env, verbose=1, n_steps=1024)


def f(d, l, agent):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()
    agent.model.learning_rate = 10
    agent.num = 5
    print("done")


if __name__ == '__main__':
    with Manager() as manager:
        d = manager.dict()
        l = manager.list(range(10))
        print(d)
        print(l)
        env = make_vec_env(env_id="FetchReachDense-v1", n_envs=1)
        agent = manager.Value('f',(Tester(tnum=1*2.0)))
        print(agent.model.learning_rate)
        print(agent.num)

        p = Process(target=f, args=(d, l, agent))
        p.start()
        p.join()

        print(d)
        print(l)
        print(agent.num)
        print(agent.model.learning_rate)