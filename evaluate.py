"""
evaluate.py — Load a trained SAC model and run evaluation rollouts.
Prints a clean summary with mean reward, std, min, max, and success rate.

Usage:
    python evaluate.py                       # Evaluates best_model (default)
    python evaluate.py --model ./final_model # Evaluates a specific checkpoint
    python evaluate.py --episodes 20         # Runs 20 eval episodes
"""

import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import config


# ─── Argument Parser ─────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained SAC model")
    parser.add_argument(
        "--model",
        type=str,
        default=config.BEST_MODEL_PATH,
        help="Path to saved model (default: best_model)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=config.EVAL_ENV_ID,
        help="Environment ID to evaluate on"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment visually during eval"
    )
    return parser.parse_args()


# ─── Evaluation Loop ─────────────────────────────────────────
def evaluate_model(model_path: str, env_id: str, n_episodes: int, render: bool = False):
    """
    Runs n_episodes rollouts and collects per-episode stats.
    Returns a dict with reward stats.
    """
    # Load model
    print(f"[INFO] Loading model from: {model_path}")
    model = SAC.load(model_path, device="cuda")

    # Create env
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    # ─── Rollouts ───
    episode_rewards = []
    episode_lengths = []

    # HalfCheetah reward threshold — anything above 4000 is considered "successful"
    # (random policy scores ~-300, trained SAC typically reaches 4000-8000)
    SUCCESS_THRESHOLD = 4000.0

    print(f"\n[INFO] Running {n_episodes} evaluation episodes on {env_id}...")
    print("-" * 55)

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        status = "✓ SUCCESS" if total_reward >= SUCCESS_THRESHOLD else "  ---"
        print(f"  Episode {ep+1:>3}/{n_episodes} | Reward: {total_reward:>10.2f} | Steps: {steps:>5} | {status}")

    env.close()

    # ─── Summary Stats ───
    rewards = np.array(episode_rewards)
    success_rate = (rewards >= SUCCESS_THRESHOLD).mean() * 100.0

    summary = {
        "mean_reward":   float(rewards.mean()),
        "std_reward":    float(rewards.std()),
        "min_reward":    float(rewards.min()),
        "max_reward":    float(rewards.max()),
        "median_reward": float(np.median(rewards)),
        "success_rate":  success_rate,
        "n_episodes":    n_episodes,
        "success_threshold": SUCCESS_THRESHOLD,
    }

    return summary


# ─── Pretty Print ────────────────────────────────────────────
def print_summary(summary: dict):
    print("\n" + "=" * 55)
    print("          EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Episodes Evaluated :  {summary['n_episodes']}")
    print(f"  Success Threshold  :  {summary['success_threshold']:.0f}")
    print("-" * 55)
    print(f"  Mean Reward        :  {summary['mean_reward']:>10.2f}")
    print(f"  Std  Reward        :  {summary['std_reward']:>10.2f}")
    print(f"  Median Reward      :  {summary['median_reward']:>10.2f}")
    print(f"  Min  Reward        :  {summary['min_reward']:>10.2f}")
    print(f"  Max  Reward        :  {summary['max_reward']:>10.2f}")
    print("-" * 55)
    print(f"  ★ Success Rate     :  {summary['success_rate']:>9.1f}%")
    print("=" * 55)


# ─── Entry Point ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    summary = evaluate_model(
        model_path=args.model,
        env_id=args.env,
        n_episodes=args.episodes,
        render=args.render,
    )

    print_summary(summary)