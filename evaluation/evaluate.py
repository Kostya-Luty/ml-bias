import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import SAC

gym.register_envs(gymnasium_robotics)


def evaluate_policy(model_path, n_episodes=100, seed=0):
    """Run a saved SAC policy deterministically and return mean success rate."""
    env = gym.make("FetchReach-v4")
    model = SAC.load(model_path, env=env)

    successes = []
    obs, _ = env.reset(seed=seed)

    for _ in range(n_episodes):
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes.append(float(info["is_success"]))
                obs, _ = env.reset()
                break

    env.close()
    return sum(successes) / len(successes)


if __name__ == "__main__":
    rate = evaluate_policy("results/test1")
    print(f"Success rate: {rate:.1%}")
