import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN

env = gym.make("highway-v0")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("dqn_fullytrained")
env.close()
