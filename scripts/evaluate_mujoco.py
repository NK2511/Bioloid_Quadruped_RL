import os
import sys
import time
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
#  ✏️  EDIT THIS SECTION ONLY
# ─────────────────────────────────────────────────────────────────────────────

# Choose which agent to run:
#   "walker"      →  BioloidMujocoEnv          (walk forward)
#   "turn_left"   →  BioloidMujocoTurnLeftEnv  (turn left in place)
#   "turn_right"  →  BioloidMujocoTurnRightEnv (turn right in place)
TASK = "walker"

# Full path to the .pth checkpoint file you want to watch
CHECKPOINT = r"C:\Desktop\Python\Quadruped_Reinforcement_Learning\output\mujoco\turn_left\checkpoints\checkpoint_ep3750.pth"

# How many episodes to run
EPISODES = 3

# Max steps per episode (1000 ≈ ~17 seconds of simulation)
MAX_STEPS = 1000

# Playback speed  (1.0 = realtime,  0.5 = half speed,  2.0 = double speed)
SPEED = 1.0

# ─────────────────────────────────────────────────────────────────────────────
#  🚫  DO NOT EDIT BELOW THIS LINE
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from brains.sac.sac_agent import soft_actor_critic_agent
from envs.mujoco.quadruped_mujoco_env import BioloidMujocoEnv
from envs.mujoco.turn_left_mujoco_env import BioloidMujocoTurnLeftEnv
from envs.mujoco.turn_right_mujoco_env import BioloidMujocoTurnRightEnv

TASK_ENV_MAP = {
    "walker":     BioloidMujocoEnv,
    "turn_left":  BioloidMujocoTurnLeftEnv,
    "turn_right": BioloidMujocoTurnRightEnv,
}


def load_agent(checkpoint_path, env, device):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"\n❌  Checkpoint not found:\n   {checkpoint_path}\n"
            f"   Double-check the CHECKPOINT path at the top of this file."
        )

    print(f"   Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try to infer hidden_size from the saved weights
    hidden_size = ckpt.get("hidden_size", 256)
    if isinstance(ckpt, dict) and "actor" in ckpt:
        try:
            hidden_size = list(ckpt["actor"].values())[0].shape[0]
        except Exception:
            pass

    agent = soft_actor_critic_agent(
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space,
        device=device,
        hidden_size=hidden_size,
        seed=0, lr=0.0, gamma=0.0, tau=0.0, alpha=0.0,
    )

    if isinstance(ckpt, dict) and "actor" in ckpt:
        agent.policy.load_state_dict(ckpt["actor"], strict=False)
        print(f"   ✅ Loaded  (episode saved = {ckpt.get('episode', '?')},  hidden_size = {hidden_size})")
    else:
        agent.policy.load_state_dict(ckpt, strict=False)
        print(f"   ✅ Loaded actor weights  (hidden_size = {hidden_size})")

    agent.policy.eval()
    return agent


def main():
    if TASK not in TASK_ENV_MAP:
        raise ValueError(f"TASK must be one of {list(TASK_ENV_MAP.keys())}, got '{TASK}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'═'*55}")
    print(f"  MuJoCo Evaluator")
    print(f"{'═'*55}")
    print(f"  Task      : {TASK}")
    print(f"  Episodes  : {EPISODES}")
    print(f"  Max steps : {MAX_STEPS}")
    print(f"  Speed     : {SPEED}x")
    print(f"  Device    : {device}")
    print(f"{'═'*55}")

    EnvClass = TASK_ENV_MAP[TASK]
    env = EnvClass(render_mode="human", max_steps=MAX_STEPS)

    agent = load_agent(CHECKPOINT, env, device)

    step_sleep = (1.0 / 60.0) / max(SPEED, 0.01)
    all_rewards = []

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0
        done = False

        print(f"\n─── Episode {ep} / {EPISODES} ───────────────────────────────")

        while not done and step < MAX_STEPS:
            action = agent.select_action(obs, eval=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

            env.render()
            time.sleep(step_sleep)

        status = "✅ Survived" if not terminated else "💀 Fell"
        print(f"  {status}  |  Steps: {step:>5}  |  Reward: {ep_reward:>10.2f}")
        all_rewards.append(ep_reward)

    print(f"\n{'═'*55}")
    print(f"  Results over {EPISODES} episode(s):")
    print(f"  Mean : {np.mean(all_rewards):.2f}")
    print(f"  Best : {np.max(all_rewards):.2f}")
    print(f"  Worst: {np.min(all_rewards):.2f}")
    print(f"{'═'*55}\n")

    env.close()


if __name__ == "__main__":
    main()
