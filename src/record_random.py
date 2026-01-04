import gymnasium as gym
import highway_env  # noqa: F401
from gymnasium.wrappers import RecordVideo

env = gym.make("highway-v0", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos/random", episode_trigger=lambda e: e == 0)

obs, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()

#Untrained tek video kaydı