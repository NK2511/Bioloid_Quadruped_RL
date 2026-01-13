import os
import time
import argparse
from collections import deque
from typing import List, Dict, Any, Optional
import sys
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# sys.path.append(os.path.join(PROJECT_ROOT, 'envs'))
from envs.quadruped_env import BioloidAntLikeEnv
from sac.sac_agent import soft_actor_critic_agent
from sac.replay_memory import ReplayMemory


def save(agent, directory, filename, episode, reward):
    # Sanitize reward string for cleaner filenames (e.g., -12.34 -> "neg12p34")
    reward_str = str(reward).replace('.', 'p').replace('-', 'neg')

    os.makedirs(directory, exist_ok=True)
    torch.save(agent.policy.state_dict(), f"{directory}/{filename}_actor_{episode}_{reward_str}.pth")
    torch.save(agent.critic.state_dict(), f"{directory}/{filename}_critic_{episode}_{reward_str}.pth")


def save_full_checkpoint(agent, directory: str, episode: int, total_steps: int, updates: int, extra: Optional[Dict[str, Any]] = None) -> str:
    """Save a full, perfect-resume checkpoint: actor, critic, optimizers, alpha/log_alpha and counters."""
    os.makedirs(directory, exist_ok=True)
    ckpt: Dict[str, Any] = {
        "actor": agent.policy.state_dict(),
        "critic": agent.critic.state_dict(),
        "policy_optim": getattr(agent, "policy_optim", None).state_dict() if getattr(agent, "policy_optim", None) else None,
        "critic_optim": getattr(agent, "critic_optim", None).state_dict() if getattr(agent, "critic_optim", None) else None,
        "alpha": float(getattr(agent, "alpha", torch.tensor(0.0)).detach().cpu().item()),
        "log_alpha": getattr(agent, "log_alpha", None).detach().cpu() if getattr(agent, "log_alpha", None) is not None else None,
        "alpha_optim": getattr(agent, "alpha_optim", None).state_dict() if getattr(agent, "alpha_optim", None) else None,
        "episode": int(episode),
        "total_steps": int(total_steps),
        "updates": int(updates),
    }
    if extra:
        ckpt.update(extra)
    path = os.path.join(directory, f"checkpoint_ep{episode}.pth")
    torch.save(ckpt, path)
    return path


def load_full_resume(agent, resume_full_path: str, device: torch.device) -> Dict[str, int]:
    """Load actor, critic, optimizers, and alpha/log_alpha from a full checkpoint.
    Returns {'episode', 'total_steps', 'updates'}. Also syncs critic_target with critic.
    """
    info = {"episode": 0, "total_steps": 0, "updates": 0}
    if not (resume_full_path and os.path.isfile(resume_full_path)):
        return info
    data = torch.load(resume_full_path, map_location=device)
    if data.get("actor"):
        agent.policy.load_state_dict(data["actor"], strict=False)
    if data.get("critic"):
        agent.critic.load_state_dict(data["critic"], strict=False)
        with torch.no_grad():
            agent.critic_target.load_state_dict(agent.critic.state_dict())
    if data.get("policy_optim") and getattr(agent, "policy_optim", None):
        agent.policy_optim.load_state_dict(data["policy_optim"])
    if data.get("critic_optim") and getattr(agent, "critic_optim", None):
        agent.critic_optim.load_state_dict(data["critic_optim"])
    if data.get("log_alpha") is not None and getattr(agent, "log_alpha", None) is not None:
        agent.log_alpha.data = data["log_alpha"].to(device)
        agent.alpha = agent.log_alpha.exp()
    if data.get("alpha_optim") and getattr(agent, "alpha_optim", None):
        agent.alpha_optim.load_state_dict(data["alpha_optim"])
    info["episode"] = int(data.get("episode", 0))
    info["total_steps"] = int(data.get("total_steps", 0))
    info["updates"] = int(data.get("updates", 0))
    print(f"[Resume] Loaded full checkpoint from {resume_full_path}")
    return info


class Callback:
    def on_training_start(self, ctx: Dict[str, Any]):
        pass

    def on_episode_start(self, ctx: Dict[str, Any]):
        pass

    def on_step(self, ctx: Dict[str, Any]):
        pass

    def on_episode_end(self, ctx: Dict[str, Any]):
        pass

    def on_training_end(self, ctx: Dict[str, Any]):
        pass


class TensorBoardCallback(Callback):
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_step(self, ctx: Dict[str, Any]):
        t = ctx["global_step"]
        self.writer.add_scalar("train/step_reward", ctx["reward"], t)

    def on_episode_end(self, ctx: Dict[str, Any]):
        ep = ctx["episode"]
        self.writer.add_scalar("train/episode_return", ctx["episode_reward"], ep)
        self.writer.add_scalar("train/episode_length", ctx["episode_steps"], ep)
        self.writer.add_scalar("train/avg100_return", ctx["avg_score"], ep)

    def on_training_end(self, ctx: Dict[str, Any]):
        self.writer.close()


class CheckpointOnBestAvg(Callback):
    def __init__(self, save_dir: str, filename_prefix: str = "weights"):
        self.best = -np.inf
        self.save_dir = save_dir
        self.filename_prefix = filename_prefix

    def on_episode_end(self, ctx: Dict[str, Any]):
        avg = ctx["avg_score"]
        if avg > self.best:
            self.best = avg
            ep = ctx["episode"]
            reward_round = round(ctx["episode_reward"], 2)
            save(ctx["agent"], self.save_dir, self.filename_prefix, str(ep), str(reward_round))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gui", action="store_true", help="Run all episodes with PyBullet GUI")
    p.add_argument("--gui-every", type=int, default=0, help="Run every Nth episode with GUI (forces reset)")
    p.add_argument("--eval-interval", type=int, default=0, help="Evaluate every N episodes")
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--save-dir", type=str, default="../bioloid_checkpoints/walker_checkpoints")
    p.add_argument("--log-dir", type=str, default="../runs/walker")
    p.add_argument("--checkpoint-interval", type=int, default=50, help="Every N episodes, save a full checkpoint (0=disabled)")
    p.add_argument("--checkpoint-dir", type=str, default="../bioloid_checkpoints/walker_checkpoints", help="Directory to store full checkpoints (defaults to save-dir)")
    p.add_argument("--resume-full-path", type=str, default="", help="Path to a full checkpoint to resume from")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start-steps", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-episodes", type=int, default=10_000, help="Maximum number of training episodes")
    return p.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate env (start in DIRECT; we'll toggle per episode if requested)
    env = BioloidAntLikeEnv(render_mode="DIRECT")

    max_steps = env.max_steps

    agent = soft_actor_critic_agent(
        env.observation_space.shape[0],
        env.action_space,
        device=device,
        hidden_size=256,
        seed=args.seed,
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
    )

    memory = ReplayMemory(args.seed, 1_000_000)

    print("device:", device)
    print("state dim:", env.observation_space.shape[0])
    print("action space:", env.action_space)
    print("learning rate:", args.lr)

    # Optional resume from full checkpoint
    resume_info = {"episode": 0, "total_steps": 0, "updates": 0}
    if args.resume_full_path:
        resume_info = load_full_resume(agent, args.resume_full_path, device)
    start_episode = int(resume_info.get("episode", 0)) + (1 if resume_info.get("episode", 0) > 0 else 0)
    start_total_steps = int(resume_info.get("total_steps", 0))
    start_updates = int(resume_info.get("updates", 0))
    if args.resume_full_path:
        print(f"[Resume] episode={start_episode}, total_steps={start_total_steps}, updates={start_updates}")

    callbacks: List[Callback] = [
        TensorBoardCallback(args.log_dir),
    ]

    # Use save-dir if checkpoint-dir not explicitly set by user
    checkpoint_dir = args.checkpoint_dir or args.save_dir

    scores, avg_scores = sac_train(
        env,
        agent,
        memory,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        num_episodes=args.num_episodes,
        max_steps=max_steps,
        callbacks=callbacks,
        gui_all=args.gui,
        gui_every=args.gui_every,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_dir=checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        resume_episode=start_episode,
        resume_total_steps=start_total_steps,
        resume_updates=start_updates,
    )

    env.close()


def sac_train(env, agent, memory, batch_size, start_steps, num_episodes, max_steps, callbacks: List[Callback], gui_all=False, gui_every=0, eval_interval=0, eval_episodes=5, save_dir="dir_bioloid_ant_like", checkpoint_interval=0, resume_episode=0, resume_total_steps=0, resume_updates=0):
    total_numsteps = int(resume_total_steps)
    updates = int(resume_updates)

    time_start = time.time()
    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    for cb in callbacks:
        cb.on_training_start({"env": env, "agent": agent})

    for i_episode in range(resume_episode, num_episodes):
        use_gui = gui_all or (gui_every > 0 and (i_episode % gui_every == 0))
        if use_gui:
            state, _ = env.enable_gui(True)
        else:
            if env.render_mode != "DIRECT":
                state, _ = env.enable_gui(False)
            else:
                state, _ = env.reset()

        for cb in callbacks:
            cb.on_episode_start({"episode": i_episode, "env": env})

        episode_reward = 0.0
        episode_steps = 0

        for step in range(max_steps):
            if total_numsteps < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Treat time-limit truncation as non-terminal for bootstrapping
            mask = 0.0 if terminated else 1.0

            memory.push(state, action, reward, next_state, mask)

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1
            total_numsteps += 1

            for cb in callbacks:
                cb.on_step({
                    "episode": i_episode,
                    "global_step": total_numsteps,
                    "episode_step": episode_steps,
                    "reward": float(reward),
                    "info": info,
                })

            if len(memory) > batch_size:
                agent.update_parameters(memory, batch_size, updates)
                updates += 1

            if done:
                break

        scores_deque.append(episode_reward)
        scores_array.append(episode_reward)
        avg_score = float(np.mean(scores_deque))
        avg_scores_array.append(avg_score)

        for cb in callbacks:
            cb.on_episode_end({
                "episode": i_episode,
                "episode_steps": episode_steps,
                "episode_reward": episode_reward,
                "avg_score": avg_score,
                "agent": agent,
            })

        s = int(time.time() - time_start)
        print(
            "Ep.: {}, Total Steps: {}, Ep.Steps: {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}".format(
                i_episode,
                total_numsteps,
                episode_steps,
                episode_reward,
                avg_score,
                s // 3600,
                (s % 3600) // 60,
                s % 60,
            )
        )

        # Optional evaluation
        if eval_interval and (i_episode % eval_interval == 0) and i_episode > 0:
            eval_return = evaluate_policy(agent, episodes=eval_episodes)
            print(f"Eval@{i_episode}: avg_return={eval_return:.2f}")

        # Optional periodic full checkpoint (perfect resume)
        if checkpoint_interval and (i_episode % checkpoint_interval == 0) and i_episode > 0:
            path = save_full_checkpoint(agent, save_dir, i_episode, total_numsteps, updates)
            print(f"[Checkpoint] Saved full checkpoint: {path}")

        # Stop condition placeholder (tune later)
        if avg_score > 2500.0:
            print("Solved environment with Avg Score:", avg_score)
            break

    for cb in callbacks:
        cb.on_training_end({"episodes": num_episodes, "env": env, "agent": agent})

    return scores_array, avg_scores_array


def evaluate_policy(agent, episodes=5):
    env_eval = BioloidAntLikeEnv(render_mode="DIRECT")
    total = 0.0
    for _ in range(episodes):
        state, _ = env_eval.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < env_eval.max_steps:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env_eval.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)
            steps += 1
        total += ep_ret
    env_eval.close()
    return total / float(max(1, episodes))


if __name__ == "__main__":
    main()
