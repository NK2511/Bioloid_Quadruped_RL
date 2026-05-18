import os
import time
import argparse
from collections import deque

import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from envs.mujoco.point_goal_env_mujoco import BioloidMujocoEnvPointGoal
from brains.sac.sac_agent_discrete import SACDiscreteAgent 
from brains.sac.replay_memory import ReplayMemory

def save_checkpoint(agent, directory: str, episode: int, total_steps: int, updates: int):
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
    path = os.path.join(directory, f"mujoco_point_goal_ep{episode}.pth")
    torch.save(ckpt, path)
    return path


def load_resume(agent, resume_path: str, device: torch.device) -> dict:
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
    print(f"[SUCCESS] Resuming training from episode {info['episode']}")
    return info


def parse_args():
    p = argparse.ArgumentParser(description="Train a high-level Navigation agent in MuJoCo.")
    p.add_argument("--save-dir", type=str, default="journal_paper/mujoco_navigation/checkpoints")
    p.add_argument("--log-dir", type=str, default="journal_paper/mujoco_navigation/runs")
    p.add_argument("--checkpoint-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--start-steps", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume-path", type=str, default="")
    p.add_argument("--num-episodes", type=int, default=10_000)
    
    # Defaults in Kaggle will be overridden by CLI args
    p.add_argument("--walker-path", type=str, default="models/mujoco/sac/Walker.pth")
    p.add_argument("--turn-left-path", type=str, default="models/mujoco/sac/Left_Turner.pth")
    p.add_argument("--turn-right-path", type=str, default="models/mujoco/sac/Right_Turner.pth")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = BioloidMujocoEnvPointGoal(
        walker_path=args.walker_path,
        turn_left_path=args.turn_left_path,
        turn_right_path=args.turn_right_path,
    )
    memory = ReplayMemory(args.seed, 100_000)

    agent = SACDiscreteAgent(
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space,
        device=device,
        hidden_size=256,
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        alpha=0.2 
    )

    resume_info = load_resume(agent, args.resume_path, device)
    start_episode = resume_info.get("episode", 0) + (1 if args.resume_path else 0)
    total_numsteps = resume_info.get("total_steps", 0)
    updates = resume_info.get("updates", 0)

    writer = SummaryWriter(log_dir=args.log_dir)
    scores_deque = deque(maxlen=100)
    time_start = time.time()

    print("\n--- MuJoCo Point-Goal Agent Training Started ---")

    try:
        for i_episode in range(start_episode, args.num_episodes):
            state, _ = env.reset(seed=args.seed + i_episode)
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                if total_numsteps < args.start_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state)

                next_state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                mask = 0.0 if terminated else 1.0
                memory.push(state, action, reward, next_state, mask)

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
            
            s = int(time.time() - time_start)
            goal_status = "Yes" if info.get("goal_reached") else "No"
            print(f"Ep.: {i_episode}, Steps: {total_numsteps}, Score: {episode_reward:.2f}, Avg: {avg_score:.2f}, Goal: {goal_status}, Time: {s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}")

            if (i_episode % args.checkpoint_interval == 0) and i_episode > 0:
                save_checkpoint(agent, args.save_dir, i_episode, total_numsteps, updates)

    except KeyboardInterrupt:
        print("\n--- Training Interrupted ---")

    finally:
        env.close()
        writer.close()

if __name__ == "__main__":
    main()
