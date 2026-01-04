import gymnasium as gym
import highway_env  # noqa: F401
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

env = gym.make("highway-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos/fullytrained", episode_trigger=lambda e: e == 0)

model = DQN.load("dqn_fullytrained")

obs, info = env.reset()
for _ in range(800):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    if terminated or truncated:
        break

env.close()
