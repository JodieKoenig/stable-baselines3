import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("FetchReachDense-v1", n_envs=1)
j = 0

model = PPO("MultiInputPolicy", env, verbose=1)
# what is the action space, how far is it moving?
model.learn(total_timesteps=10000)
model.save("ppo_fetchreachdense")
# final_policy = popbased()mod1,mod2,mod3)
del model # remove to demonstrate saving and loading
# octave plotting!!
model = PPO.load("ppo_fetchreachdense")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    # print(action)
    obs, rewards, dones, info = env.step(action)

    # if rewards == 0:
    #     j = j+1
    env.render()
    # print(rewards)


# ABOUT PPO:
# Policy: MultiInputPolicy = MultiInputActorCriticPolicy