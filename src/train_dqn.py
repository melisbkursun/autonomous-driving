from __future__ import annotations

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN


def main() -> None:
    env = gym.make("highway-v0")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
    )
    model.learn(total_timesteps=1000)
    model.save("dqn_highway")


if __name__ == "__main__":
    main()
