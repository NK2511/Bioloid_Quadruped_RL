import argparse
import time

import numpy as np
import torch
import pybullet as p

from envs.point_goal_env import BioloidEnvPointGoal, NavigationCommands
from sac.sac_agent_discrete import SACDiscreteAgent


"""
Evaluate Point Goal Script
==========================

This script evaluates a trained hierarchical Point-Goal navigation agent.
It loads a pre-trained 'point-goal' agent which selects high-level commands (Walk, Turn Left, Turn Right)
based on the robot's current state and the relative position of the goal.

The script:
1.  Initializing the `BioloidEnvPointGoal` environment.
2.  Loads the high-level `sac_agent_discrete` policy from `models/Point_goal.pth`.
3.  Defines a set of waypoints.
4.   Sequentially sets each waypoint as the goal and runs the agent to navigate towards it.
5.  Visualizes the path and goal in the PyBullet GUI (if enabled).

Usage:
    python evaluate_point_goal.py --walker-path models/Walker.pth ...
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
        hidden_size=256, # Assuming 256, as used in training
        lr=0.0, gamma=0.0, tau=0.0, alpha=0.0, # These are not used for eval
    )

    # Load just the policy's weights for evaluation.
    if "policy" in checkpoint:
        agent.policy.load_state_dict(checkpoint['policy'])
    else:
        # Handle checkpoints that might only contain the policy state_dict
        agent.policy.load_state_dict(checkpoint)

    print(f"✅ Successfully loaded point-goal agent policy from: {model_path}")
    agent.policy.eval()
    return agent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained point-goal navigation agent by following a hardcoded path.")
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

    env_kwargs = {}
    if args.walker_path: env_kwargs['walker_path'] = args.walker_path
    if args.turn_left_path: env_kwargs['turn_left_path'] = args.turn_left_path
    if args.turn_right_path: env_kwargs['turn_right_path'] = args.turn_right_path

    env = BioloidEnvPointGoal(render_mode="GUI", **env_kwargs)

    # --- 2. Load the Trained Point-Goal (Navigator) Agent ---
    try:
        agent = load_point_goal_agent(r"models\Point_goal.pth", env, device) # Hardcoded to use Point_goal.pth
    except (ValueError, FileNotFoundError) as e:
        print(f"Failed to load the point-goal agent. Exiting. Error: {e}")
        env.close()
        return

    # --- 3. Run the Evaluation Loop for the Path ---
    obs, _ = env.reset(set_new_goal=False) 

    total_path_steps = 0
    path_completed = True

    # --- Visualize the Waypoint Path ---
    if env.render_mode == "GUI":
        print("Visualizing waypoint path...")
        path_points = [p.getBasePositionAndOrientation(env.base_env.robot_id)[0][:2]] + waypoints
        for i in range(len(path_points) - 1):
            p.addUserDebugLine(
                [path_points[i][0], path_points[i][1], 0.02],
                [path_points[i+1][0], path_points[i+1][1], 0.02],
                [0, 0, 1], # Blue line
                lineWidth=3,
                lifeTime=0,
                physicsClientId=env.client_id
            )

    for i, (goal_x, goal_y) in enumerate(waypoints):
        print(f"\n--- Navigating to Waypoint {i+1}/{len(waypoints)}: ({goal_x:.2f}, {goal_y:.2f}) ---")

        # Set the current waypoint as the goal
        goal_pos = np.array([goal_x, goal_y])
        env.goal_position = goal_pos


        if env.goal_marker_body_id != -1:
            p.removeBody(env.goal_marker_body_id, physicsClientId=env.client_id)
        env.goal_marker_body_id = p.createMultiBody(
            baseVisualShapeIndex=env.goal_marker_shape_id,
            basePosition=[goal_pos[0], goal_pos[1], 0.05],
            physicsClientId=env.client_id
        )

        # Recalculate observation and metrics for the new goal from the robot's current position
        obs = env._get_point_goal_observation()
        env.last_dist_to_goal, env.last_angle_to_goal = env._get_raw_goal_metrics()

        # --- Inner navigation loop for the current waypoint ---
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
            # Only consider this a failure if termination was NOT due to reaching the goal.
            elif terminated or truncated:
                print(f"\n❌ Episode ended before reaching waypoint {i+1}. Reason: Terminated={terminated}, Truncated={truncated}")
                path_completed = False
                break # Exit inner loop
        
        if not path_completed:
            break # Exit outer loop

    print("\n\n--- Path Evaluation Finished ---")
    if path_completed:
        print(f"✅ Successfully navigated the full path of {len(waypoints)} waypoints in {total_path_steps} total steps!")
    else:
        print(f"❌ Path navigation failed.")

    time.sleep(5) # Pause for 5 seconds before closing
    env.close()

if __name__ == "__main__":
    main()