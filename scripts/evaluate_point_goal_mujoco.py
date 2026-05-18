import argparse
import time
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from envs.mujoco.point_goal_env_mujoco import BioloidMujocoEnvPointGoal, NavigationCommands
from brains.sac.sac_agent_discrete import SACDiscreteAgent


"""
Evaluate Point Goal Script (MuJoCo version)
===========================================

This script evaluates a trained hierarchical Point-Goal navigation agent under MuJoCo.
It loads a pre-trained 'point-goal' agent which selects high-level commands (Walk, Turn Left, Turn Right)
based on the robot's current state and the relative position of the goal.
"""


def load_point_goal_agent(model_path: str, env, device: torch.device) -> SACDiscreteAgent:
    """Helper to load a pre-trained discrete-action point-goal agent."""
    if not model_path:
        raise ValueError("Model path for the point-goal agent cannot be empty.")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise FileNotFoundError(f"Error loading point-goal model from {model_path}: {e}")

    # The agent is created with the same structure used during training.
    agent = SACDiscreteAgent(
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space,
        device=device,
        hidden_size=256,
        lr=0.0, gamma=0.0, tau=0.0, alpha=0.0,
    )

    # Load policy weights.
    if "policy" in checkpoint:
        agent.policy.load_state_dict(checkpoint['policy'])
    else:
        agent.policy.load_state_dict(checkpoint)

    print(f"✅ Successfully loaded point-goal agent policy from: {model_path}")
    agent.policy.eval()
    return agent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained MuJoCo point-goal navigation agent by following a path.")
    p.add_argument("--point-goal-path", type=str, default=r"models\mujoco\sac\Point_goal.pth", help="Path to point-goal policy.")
    p.add_argument("--walker-path", type=str, help="Optional path to the walker expert model.")
    p.add_argument("--turn-left-path", type=str, help="Optional path to the turn-left expert model.")
    p.add_argument("--turn-right-path", type=str, help="Optional path to the turn-right expert model.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    waypoints = [
        (1.5, 1.0),   
        (1.5, -1.0),  
        (0.0, 0.0),  
    ]

    env_kwargs = {'render_mode': 'human'}
    if args.walker_path: env_kwargs['walker_path'] = args.walker_path
    if args.turn_left_path: env_kwargs['turn_left_path'] = args.turn_left_path
    if args.turn_right_path: env_kwargs['turn_right_path'] = args.turn_right_path

    print("Initializing MuJoCo Bioloid Point-Goal Environment...")
    env = BioloidMujocoEnvPointGoal(**env_kwargs)

    # --- 2. Load the Trained Point-Goal Agent ---
    try:
        agent = load_point_goal_agent(args.point_goal_path, env, device)
    except (ValueError, FileNotFoundError) as e:
        print(f"Failed to load the point-goal agent. Exiting. Error: {e}")
        env.close()
        return

    # --- 3. Run the Evaluation Loop for the Path ---
    obs, _ = env.reset(set_new_goal=False) 

    total_path_steps = 0
    path_completed = True

    for i, (goal_x, goal_y) in enumerate(waypoints):
        print(f"\n--- Navigating to Waypoint {i+1}/{len(waypoints)}: ({goal_x:.2f}, {goal_y:.2f}) ---")

        # Set the current waypoint as the goal
        goal_pos = np.array([goal_x, goal_y])
        env.goal_position = goal_pos

        # Recalculate observation and metrics for the new goal from the robot's current position
        obs = env._get_point_goal_observation()
        env.last_dist_to_goal, env.last_angle_to_goal = env._get_raw_goal_metrics()

        waypoint_reached = False
        step_count_for_waypoint = 0
        while not waypoint_reached:
            action = agent.select_action(obs, eval=True)
            obs, _, terminated, truncated, info = env.step(action)

            command_name = NavigationCommands(action).name
            dist = info.get('dist_to_goal', 0.0)
            print(f"Step {step_count_for_waypoint+1:03d} | Command: {command_name:<11} | Distance to Goal: {dist:.2f}m", end='\r')

            step_count_for_waypoint += 1
            total_path_steps += 1
            time.sleep(1./60.) # Slow down for better viewing

            if info.get("goal_reached"):
                print(f"\n✅ Waypoint {i+1} reached in {step_count_for_waypoint} steps!")
                waypoint_reached = True
            elif terminated or truncated:
                print(f"\n❌ Episode ended before reaching waypoint {i+1}. Reason: Terminated={terminated}, Truncated={truncated}")
                path_completed = False
                break
        
        if not path_completed:
            break

    print("\n\n--- Path Evaluation Finished ---")
    if path_completed:
        print(f"✅ Successfully navigated the full path of {len(waypoints)} waypoints in {total_path_steps} total steps!")
    else:
        print(f"❌ Path navigation failed.")

    time.sleep(5)
    env.close()

if __name__ == "__main__":
    main()
