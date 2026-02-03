"""
config.py — Central configuration for RL Locomotion Project
All hyperparameters, env settings, and W&B config live here.
"""

# ─── Environment ─────────────────────────────────────────────
ENV_ID = "HalfCheetah-v5"          # MuJoCo locomotion task
NUM_ENVS = 4                        # Parallel envs (good for 6GB VRAM)

# ─── Algorithm ───────────────────────────────────────────────
ALGORITHM = "SAC"                   # SAC is best for continuous control
TOTAL_TIMESTEPS = 1_000_000         # 1M steps — ~15-20 min on RTX 3060
LEARNING_STARTS = 10_000            # Steps before training begins (fills replay buffer)
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
TAU = 0.005                         # Soft update coefficient
GAMMA = 0.99                        # Discount factor
ENT_COEF = "auto"                   # Auto-tune entropy coefficient (SAC feature)
POLICY_FREQUENCY = 1                # How often to update policy vs critic

# ─── Evaluation ──────────────────────────────────────────────
EVAL_FREQ = 5_000                   # Evaluate every N steps
N_EVAL_EPISODES = 10                # Number of eval rollouts per checkpoint
EVAL_ENV_ID = "HalfCheetah-v5"

# ─── Checkpointing ──────────────────────────────────────────
SAVE_FREQ = 50_000                  # Save model checkpoint every N steps
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = "./best_model"

# ─── W&B ─────────────────────────────────────────────────────
WANDB_PROJECT = "rl-locomotion-halfcheetah"
WANDB_ENTITY = None                 # Set to your W&B username if needed
WANDB_RUN_NAME = None               # None = auto-generated name
WANDB_LOG_FREQ = 100                # Log metrics every N steps

# ─── Reproducibility ─────────────────────────────────────────
SEED = 42