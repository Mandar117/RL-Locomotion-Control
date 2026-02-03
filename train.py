"""
train.py — SAC training on MuJoCo HalfCheetah-v5 with W&B logging
Usage: python train.py
"""

import os
import numpy as np
import gymnasium as gym
import wandb
import stable_baselines3 as sb3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
import config


# ─── W&B Callback ────────────────────────────────────────────
class WandBCallback(sb3.common.callbacks.BaseCallback):
    """Logs training metrics to W&B every LOG_FREQ steps."""

    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Pull latest metrics from the model's logger
            metrics = {}
            for key in ["train/actor_loss", "train/critic_loss",
                        "train/ent_coef", "train/ent_coef_loss",
                        "rollout/ep_len_mean", "rollout/ep_rew_mean"]:
                val = self.logger.name_to_value.get(key)
                if val is not None:
                    metrics[key] = val

            metrics["train/timesteps"] = self.num_timesteps

            if metrics:
                wandb.log(metrics, step=self.num_timesteps)

        return True


# ─── Eval Callback with W&B logging ──────────────────────────
class WandBEvalCallback(EvalCallback):
    """Extends EvalCallback to also push eval metrics into W&B."""

    def _on_step(self) -> bool:
        result = super()._on_step()

        # After each eval, log best and last eval reward
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                wandb.log({
                    "eval/mean_reward": self.last_mean_reward,
                    "eval/std_reward": self.last_mean_ep_length if hasattr(self, "last_mean_ep_length") else 0,
                }, step=self.num_timesteps)

            if self.best_mean_reward is not None:
                wandb.log({
                    "eval/best_mean_reward": self.best_mean_reward,
                }, step=self.num_timesteps)

        return result


# ─── Environment Factory ─────────────────────────────────────
def make_env(env_id: str, seed: int = 0):
    """Creates a single monitored environment."""
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_vec_env(env_id: str, n_envs: int, seed: int = 42):
    """Creates a vectorised environment with unique seeds."""
    envs = [make_env(env_id, seed=seed + i) for i in range(n_envs)]
    return DummyVecEnv(envs)


# ─── Main Training Loop ──────────────────────────────────────
def train():
    # --- Init W&B ---
    wandb.init(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=config.WANDB_RUN_NAME,
        config={
            "algorithm": config.ALGORITHM,
            "env_id": config.ENV_ID,
            "total_timesteps": config.TOTAL_TIMESTEPS,
            "num_envs": config.NUM_ENVS,
            "learning_rate": config.LEARNING_RATE,
            "batch_size": config.BATCH_SIZE,
            "tau": config.TAU,
            "gamma": config.GAMMA,
            "ent_coef": config.ENT_COEF,
            "seed": config.SEED,
            "learning_starts": config.LEARNING_STARTS,
        },
        resume="allow",
    )

    # --- Create Envs ---
    print(f"[INFO] Creating {config.NUM_ENVS} parallel {config.ENV_ID} environments...")
    train_env = make_vec_env(config.ENV_ID, config.NUM_ENVS, seed=config.SEED)

    # Separate single env for evaluation
    eval_env = make_vec_env(config.EVAL_ENV_ID, n_envs=1, seed=config.SEED + 100)

    # --- Create Directories ---
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # --- Callbacks ---
    # 1. Periodic checkpoint saves
    checkpoint_cb = CheckpointCallback(
        save_freq=config.SAVE_FREQ // config.NUM_ENVS,  # Adjust for vec env
        save_path=config.CHECKPOINT_DIR,
        name_prefix="sac_halfcheetah",
    )

    # 2. Evaluation + best model saving
    eval_cb = WandBEvalCallback(
        eval_env=eval_env,
        eval_freq=config.EVAL_FREQ // config.NUM_ENVS,
        n_eval_episodes=config.N_EVAL_EPISODES,
        best_model_save_path=config.BEST_MODEL_PATH,
        log_path="./eval_logs",
        verbose=1,
    )

    # 3. W&B training metric logger
    wandb_cb = WandBCallback(log_freq=config.WANDB_LOG_FREQ)

    # Combine all callbacks
    callbacks = CallbackList([checkpoint_cb, eval_cb, wandb_cb])

    # --- Initialise SAC Model ---
    print(f"[INFO] Initialising SAC on {config.ENV_ID}...")
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        tau=config.TAU,
        gamma=config.GAMMA,
        ent_coef=config.ENT_COEF,
        learning_starts=config.LEARNING_STARTS,
        policy_frequency=config.POLICY_FREQUENCY,
        seed=config.SEED,
        device="cuda",
        verbose=1,
    )

    # --- Train ---
    print(f"[INFO] Training for {config.TOTAL_TIMESTEPS:,} timesteps...")
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callbacks,
        log_interval=10,
    )

    # --- Save Final Model ---
    final_path = "./final_model"
    model.save(final_path)
    print(f"[INFO] Final model saved to {final_path}")

    wandb.finish()
    print("[INFO] Training complete. Check W&B for results.")


# ─── Entry Point ─────────────────────────────────────────────
if __name__ == "__main__":
    train()