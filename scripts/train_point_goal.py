import os
import time
import argparse
from collections import deque

import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from envs.point_goal_env import BioloidEnvPointGoal
from sac.sac_agent_discrete import SACDiscreteAgent 
from sac.replay_memory import ReplayMemory

def save_checkpoint(agent, directory: str, episode: int, total_steps: int, updates: int):
    """Saves a checkpoint for the point-goal agent."""
    os.makedirs(directory, exist_ok=True)

    ckpt = {
        "policy": agent.policy.state_dict(),
        "critic": agent.critic.state_dict(),
        "policy_optim": agent.policy_optim.state_dict(),
        "critic_optim": agent.critic_optim.state_dict(),
        "episode": int(episode),
        "total_steps": int(total_steps),
        "updates": int(updates),
    }
    path = os.path.join(directory, f"point_goal_checkpoint_ep{episode}.pth")
    torch.save(ckpt, path)
    return path


def load_resume(agent, resume_path: str, device: torch.device) -> dict:
    """Loads a full training state for the point-goal agent and returns training counters."""
    info = {"episode": 0, "total_steps": 0, "updates": 0}
    if not (resume_path and os.path.isfile(resume_path)):
        return info
    
    data = torch.load(resume_path, map_location=device)

    if "policy" in data: agent.policy.load_state_dict(data["policy"])
    if "critic" in data: agent.critic.load_state_dict(data["critic"])
    if "policy_optim" in data: agent.policy_optim.load_state_dict(data["policy_optim"])
    if "critic_optim" in data: agent.critic_optim.load_state_dict(data["critic_optim"])

    info["episode"] = data.get("episode", 0)
    info["total_steps"] = data.get("total_steps", 0)
    info["updates"] = data.get("updates", 0)
    print(f"✅ Resuming training from: {os.path.basename(resume_path)} at episode {info['episode']}")
    return info


def parse_args():
    p = argparse.ArgumentParser(description="Train a high-level 'Point-Goal' agent for navigation.")
    p.add_argument("--gui", action="store_true", help="Run with PyBullet GUI to watch the training.")
    p.add_argument("--save-dir", type=str, default="../bioloid_checkpoints/point_goal_checkpoints", help="Directory to save the new Point-Goal model.")
    p.add_argument("--log-dir", type=str, default="../runs/bioloid_sac_point_goal", help="Directory for TensorBoard logs.")
    p.add_argument("--checkpoint-interval", type=int, default=100, help="Save a Point-Goal checkpoint every N episodes.")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--start-steps", type=int, default=1_000, help="Number of initial random navigation actions to collect.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate for the new Point-Goal agent.")
    p.add_argument("--resume-path", type=str, default="", help="Path to a checkpoint to RESUME training from.")
    p.add_argument("--num-episodes", type=int, default=10_000)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 1. Setup Environment and Replay Memory ---
    # The environment will internally load the expert agents.
    env = BioloidEnvPointGoal(render_mode="GUI" if args.gui else "DIRECT")
    memory = ReplayMemory(args.seed, 100_000) # Smaller memory is fine for HRL

    # --- 2. Create the Point-Goal Agent ---
    # This agent learns to choose between high-level commands.
    agent = SACDiscreteAgent(
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space,
        device=device,
        hidden_size=256,
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        alpha=0.2 # Fixed alpha for simplicity in discrete SAC
    )

    # --- Optional: Resume Training ---
    start_episode = 0
    total_numsteps = 0
    updates = 0
    if args.resume_path:
        resume_info = load_resume(agent, args.resume_path, device)
        start_episode = resume_info.get("episode", 0) + 1
        total_numsteps = resume_info.get("total_steps", 0)
        updates = resume_info.get("updates", 0)

    print("\n--- Point-Goal Agent Training Initialized ---")
    print(f"The agent will learn to navigate to a random goal.")
    print(f"Checkpoints will be saved in: {args.save_dir}")

    # --- 3. Start the Training Process ---
    writer = SummaryWriter(log_dir=args.log_dir)
    scores_deque = deque(maxlen=100)
    time_start = time.time()

    for i_episode in range(start_episode, args.num_episodes):
        state, _ = env.reset(seed=args.seed + i_episode)
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            # --- Action Selection ---
            if total_numsteps < args.start_steps:
                action = env.action_space.sample() # Random high-level command
            else:
                action = agent.select_action(state)

            # --- Environment Step ---
            # The env handles executing the command for a fixed duration
            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # --- Store Experience ---
            mask = 0.0 if terminated else 1.0
            memory.push(state, action, reward, next_state, mask)

            # --- Agent Update ---
            if len(memory) > args.batch_size:
                agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1
            total_numsteps += 1


        scores_deque.append(episode_reward)
        avg_score = np.mean(scores_deque)
        writer.add_scalar("train/episode_return", episode_reward, i_episode)
        writer.add_scalar("train/avg100_return", avg_score, i_episode)
        writer.add_scalar("train/episode_length", episode_steps, i_episode)
        if "goal_reached" in info:
             writer.add_scalar("train/goal_reached", float(info["goal_reached"]), i_episode)


        s = int(time.time() - time_start)
        print(
            "Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.2f}, Avg.Score: {:.2f}, Goal: {}, Time: {:02}:{:02}:{:02}".format(
                i_episode, total_numsteps, episode_steps, episode_reward, avg_score,
                "Yes" if info.get("goal_reached") else "No",
                s // 3600, (s % 3600) // 60, s % 60,
            )
        )

        # Save a checkpoint of the Point-Goal agent
        if (i_episode % args.checkpoint_interval == 0) and i_episode > 0:
            path = save_checkpoint(agent, args.save_dir, i_episode, total_numsteps, updates)
            print(f"[Checkpoint] Saved Point-Goal agent checkpoint: {path}")

    env.close()
    writer.close()


if __name__ == "__main__":

    main()
