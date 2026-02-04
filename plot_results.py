"""
plot_results.py — Generate clean plots from W&B run data
Usage: python3 plot_results.py
"""

import wandb
import matplotlib.pyplot as plt
import numpy as np
import os

# ─── Configuration ───────────────────────────────────────────
WANDB_PROJECT = "rl-locomotion-halfcheetah"
WANDB_ENTITY = "made2806-university-of-colorado-boulder"  # Your W&B username
RUN_ID = "wezdtj74"  # The run ID from your training (wild-microwave-3)

OUTPUT_DIR = "./plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Download Run Data ───────────────────────────────────────
print(f"[INFO] Fetching run data from W&B: {WANDB_PROJECT}/{RUN_ID}")
api = wandb.Api()
run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{RUN_ID}")

# Pull history
history = run.history()

# ─── Plot 1: Training Reward Curve ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Filter out NaN values
eval_steps = history["train/timesteps"].dropna()
eval_rewards = history["eval/mean_reward"].dropna()

# Align them (eval happens every 5k steps)
min_len = min(len(eval_steps), len(eval_rewards))
eval_steps = eval_steps.iloc[:min_len]
eval_rewards = eval_rewards.iloc[:min_len]

ax.plot(eval_steps, eval_rewards, linewidth=2, color="#2E86AB", label="Mean Eval Reward")
ax.axhline(y=4000, color="red", linestyle="--", linewidth=1.5, label="Success Threshold (4000)")
ax.fill_between(eval_steps, 4000, eval_rewards, where=(eval_rewards >= 4000), 
                alpha=0.2, color="green", label="Above Threshold")

ax.set_xlabel("Training Steps", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Reward", fontsize=12, fontweight="bold")
ax.set_title("SAC Training on HalfCheetah-v5: Reward Progression", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(left=0)

plt.tight_layout()
plot1_path = os.path.join(OUTPUT_DIR, "training_reward_curve.png")
plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
print(f"[SAVED] {plot1_path}")
plt.close()

# ─── Plot 2: Actor & Critic Loss ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Actor loss
actor_loss = history["train/actor_loss"].dropna()
actor_steps = history["train/timesteps"].iloc[:len(actor_loss)]
ax1.plot(actor_steps, actor_loss, linewidth=1.5, color="#A23B72", alpha=0.8)
ax1.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
ax1.set_ylabel("Actor Loss", fontsize=11, fontweight="bold")
ax1.set_title("Actor Loss Over Time", fontsize=12, fontweight="bold")
ax1.grid(alpha=0.3)

# Critic loss
critic_loss = history["train/critic_loss"].dropna()
critic_steps = history["train/timesteps"].iloc[:len(critic_loss)]
ax2.plot(critic_steps, critic_loss, linewidth=1.5, color="#F18F01", alpha=0.8)
ax2.set_xlabel("Training Steps", fontsize=11, fontweight="bold")
ax2.set_ylabel("Critic Loss", fontsize=11, fontweight="bold")
ax2.set_title("Critic Loss Over Time", fontsize=12, fontweight="bold")
ax2.grid(alpha=0.3)

plt.tight_layout()
plot2_path = os.path.join(OUTPUT_DIR, "training_losses.png")
plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
print(f"[SAVED] {plot2_path}")
plt.close()

# ─── Plot 3: Final Evaluation Bar Chart ──────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

metrics = ["Mean", "Median", "Max", "Min"]
values = [7651.50, 7648.86, 7695.85, 7608.45]
colors = ["#06A77D", "#0C7C59", "#005377", "#01497C"]

bars = ax.barh(metrics, values, color=colors, edgecolor="black", linewidth=1.2)
ax.axvline(x=4000, color="red", linestyle="--", linewidth=2, label="Success Threshold")

# Add value labels on bars
for bar, val in zip(bars, values):
    ax.text(val + 50, bar.get_y() + bar.get_height()/2, f"{val:.0f}", 
            va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Reward", fontsize=12, fontweight="bold")
ax.set_title("Final Evaluation Results (20 Episodes, 100% Success)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plot3_path = os.path.join(OUTPUT_DIR, "evaluation_results.png")
plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
print(f"[SAVED] {plot3_path}")
plt.close()

print(f"\n[INFO] All plots saved to {OUTPUT_DIR}/")
print("[INFO] Use these in your README or portfolio!")