#Training sırasında rewardları kaydedip grafik üretmek için 

import os
import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


def main() -> None:
    os.makedirs("plots", exist_ok=True)

    env = gym.make("highway-v0")
    env = Monitor(env)

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    model.save("dqn_final")

    # episode rewards come from Monitor
    rewards = env.get_episode_rewards()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward vs Episodes")
    plt.savefig("plots/reward_curve.png", dpi=200, bbox_inches="tight")
    env.close()


if __name__ == "__main__":
    main()
