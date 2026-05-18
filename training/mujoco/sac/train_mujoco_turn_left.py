import os
import time
import argparse
import sys
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from envs.mujoco.turn_left_mujoco_env import BioloidMujocoTurnLeftEnv
from brains.sac.sac_agent import soft_actor_critic_agent
from brains.sac.replay_memory import ReplayMemory

# (Training Logic functions same as walker...)
def save_full_checkpoint(agent, directory: str, episode_id, total_steps: int, updates: int, extra: Optional[Dict[str, Any]] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    ckpt: Dict[str, Any] = {
        "actor": agent.policy.state_dict(),
        "critic": agent.critic.state_dict(),
        "policy_optim": getattr(agent, "policy_optim", None).state_dict() if getattr(agent, "policy_optim", None) else None,
        "critic_optim": getattr(agent, "critic_optim", None).state_dict() if getattr(agent, "critic_optim", None) else None,
        "alpha": float(getattr(agent, "alpha", torch.tensor(0.0)).detach().cpu().item()),
        "log_alpha": getattr(agent, "log_alpha", None).detach().cpu() if getattr(agent, "log_alpha", None) is not None else None,
        "alpha_optim": getattr(agent, "alpha_optim", None).state_dict() if getattr(agent, "alpha_optim", None) else None,
        "episode": episode_id,
        "total_steps": int(total_steps),
        "updates": int(updates),
    }
    if extra:
        ckpt.update(extra)
    path = os.path.join(directory, f"checkpoint_ep{episode_id}.pth" if isinstance(episode_id, int) else f"{episode_id}_checkpoint.pth")
    torch.save(ckpt, path)
    return path

def load_full_resume(agent, resume_full_path: str, device: torch.device) -> Dict[str, int]:
    info = {"episode": 0, "total_steps": 0, "updates": 0}
    if not resume_full_path or not os.path.isfile(resume_full_path): return info
    data = torch.load(resume_full_path, map_location=device)
    if data.get("actor"): agent.policy.load_state_dict(data["actor"], strict=False)
    if data.get("critic"): 
        agent.critic.load_state_dict(data["critic"], strict=False)
        with torch.no_grad(): agent.critic_target.load_state_dict(agent.critic.state_dict())
    if data.get("policy_optim") and getattr(agent, "policy_optim", None): agent.policy_optim.load_state_dict(data["policy_optim"])
    if data.get("critic_optim") and getattr(agent, "critic_optim", None): agent.critic_optim.load_state_dict(data["critic_optim"])
    if data.get("log_alpha") is not None and getattr(agent, "log_alpha", None) is not None:
        agent.log_alpha.data = data["log_alpha"].to(device)
        agent.alpha = agent.log_alpha.exp()
    if data.get("alpha_optim") and getattr(agent, "alpha_optim", None): agent.alpha_optim.load_state_dict(data["alpha_optim"])
    info["episode"], info["total_steps"], info["updates"] = int(data.get("episode", 0)), int(data.get("total_steps", 0)), int(data.get("updates", 0))
    print(f"[Resume] Loaded from {resume_full_path}"); return info

class Callback:
    def on_training_start(self, ctx: Dict[str, Any]): pass
    def on_step(self, ctx: Dict[str, Any]): pass
    def on_episode_end(self, ctx: Dict[str, Any]): pass

class TensorBoardCallback(Callback):
    def __init__(self, log_dir: str): self.writer = SummaryWriter(log_dir=log_dir)
    def on_step(self, ctx: Dict[str, Any]): self.writer.add_scalar("train/step_reward", ctx["reward"], ctx["global_step"])
    def on_episode_end(self, ctx: Dict[str, Any]):
        self.writer.add_scalar("train/episode_return", ctx["episode_reward"], ctx["episode"])
        self.writer.add_scalar("train/avg100_return", ctx["avg_score"], ctx["episode"])

def sac_train(env, agent, memory, args, callbacks: List[Callback], resume_info: Dict[str, int]):
    total_numsteps, updates, resume_episode = int(resume_info["total_steps"]), int(resume_info["updates"]), int(resume_info["episode"])
    time_start, scores_deque = time.time(), deque(maxlen=100)
    for cb in callbacks: cb.on_training_start({"env": env, "agent": agent})
    try:
        for i_episode in range(resume_episode, args.num_episodes):
            state, _ = env.reset(); episode_reward = 0.0
            for _ in range(env.max_steps):
                action = env.action_space.sample() if total_numsteps < args.start_steps else agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                memory.push(state, action, reward, next_state, 0.0 if terminated else 1.0)
                state, episode_reward, total_numsteps = next_state, episode_reward + float(reward), total_numsteps + 1
                for cb in callbacks: cb.on_step({"episode": i_episode, "global_step": total_numsteps, "reward": float(reward)})
                if len(memory) > args.batch_size:
                    agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1
                if terminated or truncated: break
            scores_deque.append(episode_reward); avg_score = float(np.mean(scores_deque))
            for cb in callbacks: cb.on_episode_end({"episode": i_episode, "episode_reward": episode_reward, "avg_score": avg_score})
            if i_episode % 10 == 0:
                s = int(time.time() - time_start)
                print(f"Ep: {i_episode:4d} | Steps: {total_numsteps:6d} | Score: {episode_reward:7.2f} | Avg: {avg_score:7.2f} | Time: {s//3600:02}:{(s%3600)//60:02}:{s%60:02}")
            if args.checkpoint_interval > 0 and i_episode % args.checkpoint_interval == 0 and i_episode > 0:
                save_full_checkpoint(agent, args.save_dir, i_episode, total_numsteps, updates)
    except KeyboardInterrupt: print("\n--- Training Interrupted ---")
    finally: save_full_checkpoint(agent, args.save_dir, "final", total_numsteps, updates)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="output/mujoco/checkpoints_turn_left")
    parser.add_argument("--log-dir", type=str, default="output/mujoco/runs_turn_left")
    parser.add_argument("--model-path", type=str, default=os.path.join(PROJECT_ROOT, "assets", "mujoco", "Bioloid_Quadruped_Model", "Bioloid_Quadruped_Model.xml"))
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--resume-full-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-episodes", type=int, default=10000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = BioloidMujocoTurnLeftEnv(xml_path=args.model_path)
    agent = soft_actor_critic_agent(env.observation_space.shape[0], env.action_space, device=device, lr=args.lr)
    memory = ReplayMemory(args.seed, 1_000_000)
    resume_info = load_full_resume(agent, args.resume_full_path, device)
    sac_train(env, agent, memory, args, [TensorBoardCallback(args.log_dir)], resume_info)

if __name__ == "__main__": main()
