import gym
import mujoco_py
# Setting FetchPickAndPlace-v1 as the environment
env = gym.make('FetchReach-v1')
# Sets an initial state
if env.reset() is not None:
    print('OKAY')
else:
    print('Problem')
env.reset()
# Rendering our instance 300 times
for _ in range(30000):
    # renders the environment
    env.render()
    # Takes a random action from its action space
    # aka the number of unique actions an agent can perform
    env.step(env.action_space.sample())
env.close()