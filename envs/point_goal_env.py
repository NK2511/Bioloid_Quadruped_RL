import math
from typing import Optional, Tuple, Dict, Any, List
from enum import IntEnum
import sys
import os

import gymnasium as gym
import numpy as np
import torch
import pybullet as p
import pybullet_data
from gymnasium import spaces


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from .quadruped_env import BioloidAntLikeEnv
from sac.sac_agent import soft_actor_critic_agent


def load_expert_agent(model_path: str, env, device: torch.device) -> soft_actor_critic_agent:

    if not model_path:
        raise ValueError(f"Model path cannot be empty for expert agent.")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise FileNotFoundError(f"Error loading model from {model_path}: {e}")

    hidden_size = checkpoint.get("hidden_size", 256)
    # Experts were trained on 28-dim obs space
    agent = soft_actor_critic_agent(
        28, action_space=env.action_space, device=device, hidden_size=hidden_size,
        seed=0, lr=0.0, gamma=0.0, tau=0.0, alpha=0.0,
    )
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        agent.policy.load_state_dict(checkpoint['actor'])
    else:
        agent.policy.load_state_dict(checkpoint)
    agent.policy.eval()
    return agent


def transform_observation_for_expert(obs: np.ndarray, base_quat_world: list) -> np.ndarray:

    _, _, yaw = p.getEulerFromQuaternion(base_quat_world)
    inverse_yaw_quat = p.getQuaternionFromEuler([0, 0, -yaw])
    world_ang_vel = obs[19:22]
    rotated_ang_vel, _ = p.multiplyTransforms([0, 0, 0], inverse_yaw_quat, world_ang_vel, [0, 0, 0, 1])
    _, rotated_base_quat = p.multiplyTransforms([0, 0, 0], inverse_yaw_quat, [0, 0, 0], base_quat_world)
    obs[19:22] = rotated_ang_vel
    obs[22:26] = rotated_base_quat
    return obs


class NavigationCommands(IntEnum):
    WALK = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3


class BioloidEnvPointGoal(gym.Env):
    """
    A high-level 'point-goal' environment for navigation.
    - The agent chooses a high-level command (walk, turn, stop).
    - The environment executes that command for a fixed duration using a pre-trained 'expert' agent.
    - The goal is to navigate to a randomly chosen target point.
    """
    metadata = {"render_modes": ["GUI", "DIRECT"]}

    def __init__(
        self,
        render_mode: str = "DIRECT",
        max_steps: int = 1000,
        # Paths to pre-trained expert models
        walker_path: str = r"models\Walker.pth",
        turn_left_path: str = r"models\Left_Turner.pth",
        turn_right_path: str = r"models\Right_Turner.pth",
        # Control parameters
        command_duration_steps: int = 30, # How many physics steps to execute a command for
        goal_distance_threshold: float = 0.2, # meters
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # --- 1. Create the underlying physics environment ---
        # Use a very long max_steps here, as we control termination in this overlay env.
        self.base_env = BioloidAntLikeEnv(render_mode=render_mode, max_steps=max_steps * command_duration_steps, urdf_path=r"assets\Bioloid_Quadruped_Model.urdf")
        self.client_id = self.base_env.client_id

        # --- 2. Load the expert agents ---
        print("--- Loading Expert Agents for Overlay Environment ---")
        self.walker_agent = load_expert_agent(walker_path, self.base_env, self.device)
        self.turn_left_agent = load_expert_agent(turn_left_path, self.base_env, self.device)
        self.turn_right_agent = load_expert_agent(turn_right_path, self.base_env, self.device)
        print("--- Expert Agents Loaded ---")

        # --- 3. Store original joint limits for dynamic switching ---
        self.original_joint_limits = {}
        for j_index in self.base_env.joint_indices:
            info = p.getJointInfo(self.base_env.robot_id, j_index, physicsClientId=self.client_id)
            self.original_joint_limits[j_index] = {'jointLowerLimit': info[8], 'jointUpperLimit': info[9]}

        # --- 4. Define Point-Goal Env Parameters ---
        self.max_steps = max_steps
        self.command_duration_steps = command_duration_steps
        self.goal_dist_thresh = goal_distance_threshold

        # --- 5. Define Action and Observation Space for the Point-Goal Agent ---
        # Discrete actions: WALK, TURN_LEFT, TURN_RIGHT, STOP
        self.action_space = spaces.Discrete(len(NavigationCommands))

        # Observation: [dist_to_goal, angle_to_goal] + [proprioceptive_state(28)]
        base_obs_dim = self.base_env.observation_space.shape[0] # 28
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + base_obs_dim,), dtype=np.float32)

        # --- Episode State ---
        self.step_count = 0
        self.goal_position = np.array([0.0, 0.0])
        self.last_dist_to_goal = 0.0
        self.last_angle_to_goal = 0.0
        self.np_random = np.random.RandomState()
        
        # --- Visual marker for the goal ---
        self.goal_marker_shape_id = -1
        if self.render_mode == "GUI":
            self.goal_marker_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.8], physicsClientId=self.client_id
            )
        self.goal_marker_body_id = -1

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, set_new_goal: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random.seed(seed)

        # Reset the underlying environment
        self.base_env.reset(seed=seed, options=options) # This calls p.resetSimulation()
        self.step_count = 0

        if set_new_goal:
            # --- Set a new random goal ---
            # Spawn goal in a ring around the origin
            radius = self.np_random.uniform(1.5, 2.5)
            angle = self.np_random.uniform(-np.pi, np.pi)
            self.goal_position = np.array([radius * np.cos(angle), radius * np.sin(angle)])

            # --- Update the visual goal marker ---
            if self.render_mode == "GUI":
                if self.goal_marker_body_id != -1:
                    p.removeBody(self.goal_marker_body_id, physicsClientId=self.client_id)
                self.goal_marker_body_id = p.createMultiBody(
                    baseVisualShapeIndex=self.goal_marker_shape_id,
                    basePosition=[self.goal_position[0], self.goal_position[1], 0.05],
                    physicsClientId=self.client_id
                )

        # Get the raw, un-normalized distance to the goal for reward calculation.
        # Also get the initial angle for the first step's reward calculation.
        self.last_dist_to_goal, self.last_angle_to_goal = self._get_raw_goal_metrics()

        # Get the full observation for the agent.
        obs = self._get_point_goal_observation()
        # The observation includes the normalized angle, but we need the raw angle for reward.
        self.last_angle_to_goal = self._get_raw_goal_metrics()[1]

        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        command = NavigationCommands(action)
        
        # --- Execute the chosen command for a fixed duration ---
        sub_step_reward = 0.0
        is_turning = (command == NavigationCommands.TURN_LEFT or command == NavigationCommands.TURN_RIGHT)
        self._set_turn_mode_physics(is_turning)


        original_target_velocity = self.base_env.target_velocity
        if is_turning:
            self.base_env.target_velocity = 0.0

        for _ in range(self.command_duration_steps):
            # Get the current low-level state from the base environment
            base_obs = self.base_env._get_obs()
            
            # Select the correct expert action
            if command == NavigationCommands.WALK:
                base_pos, base_quat = p.getBasePositionAndOrientation(self.base_env.robot_id, physicsClientId=self.client_id)
                expert_obs = transform_observation_for_expert(base_obs.copy(), base_quat)
                expert_action = self.walker_agent.select_action(expert_obs, eval=True)
            elif command == NavigationCommands.TURN_LEFT:
                expert_action = self.turn_left_agent.select_action(base_obs, eval=True)
            elif command == NavigationCommands.TURN_RIGHT:
                expert_action = self.turn_right_agent.select_action(base_obs, eval=True)
            else: # STOP
                joint_states = p.getJointStates(self.base_env.robot_id, self.base_env.joint_indices, physicsClientId=self.client_id)
                current_positions = np.array([s[0] for s in joint_states])
                current_velocities = np.array([s[1] for s in joint_states])
                torque = -2.5 * current_positions - 0.15 * current_velocities
                expert_action = np.clip(torque, -0.7, 0.7)

            # Step the underlying physics simulation
            _, r, term, trunc, _ = self.base_env.step(expert_action)
            sub_step_reward += r

            if term: # If the robot falls, the episode ends immediately
                break

        # Restore the original target velocity after the command is done
        self.base_env.target_velocity = original_target_velocity

        # --- Calculate Point-Goal Reward and State ---
        # Get the new observation for the agent
        point_goal_obs = self._get_point_goal_observation()
        # Get the raw, un-normalized metrics for reward calculation
        dist_to_goal, angle_to_goal = self._get_raw_goal_metrics()

        # --- Reward Component 1: Distance Reduction ---
        distance_reduction = self.last_dist_to_goal - dist_to_goal
        distance_reward = 50.0 * distance_reduction

        # --- Reward Component 2: Orientation Improvement ---
        # Reward for reducing the absolute angle to the goal.
        angle_reduction = abs(self.last_angle_to_goal) - abs(angle_to_goal)
        # Scale this reward; it's important but shouldn't overpower distance.
        orientation_reward = 10.0 * angle_reduction

        # --- Reward Component 3: Goal Achievement Bonus ---
        goal_reached = dist_to_goal < self.goal_dist_thresh
        goal_bonus = 0.0
        if goal_reached:
            goal_bonus = 200.0

        # --- Reward Component 4: Efficiency Penalty ---
        alive_penalty = -0.1

        reward = distance_reward + orientation_reward + goal_bonus + alive_penalty

        # --- Termination and Truncation ---
        self.step_count += 1
        terminated = self.base_env._is_terminated() or goal_reached
        truncated = self.step_count >= self.max_steps

        # Update state for next step
        self.last_dist_to_goal = dist_to_goal
        self.last_angle_to_goal = angle_to_goal

        info = {"dist_to_goal": dist_to_goal, "goal_reached": goal_reached}

        return point_goal_obs, float(reward), bool(terminated), bool(truncated), info

    def _get_raw_goal_metrics(self) -> Tuple[float, float]:

        base_pos, base_quat = p.getBasePositionAndOrientation(self.base_env.robot_id, physicsClientId=self.client_id)
        base_pos_xy = np.array(base_pos[:2])

        vec_to_goal_world = self.goal_position - base_pos_xy
        dist_to_goal = np.linalg.norm(vec_to_goal_world)

        rot_matrix = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)
        robot_forward_vec = rot_matrix[:, 0][:2]

        angle_to_goal = np.arctan2(vec_to_goal_world[1], vec_to_goal_world[0]) - np.arctan2(robot_forward_vec[1], robot_forward_vec[0])

        # Normalize angle to [-pi, pi]
        angle_to_goal = (angle_to_goal + np.pi) % (2 * np.pi) - np.pi

        return dist_to_goal, angle_to_goal

    def _get_point_goal_observation(self) -> np.ndarray:

        # Get raw metrics first
        dist_to_goal, angle_to_goal = self._get_raw_goal_metrics()

        # Normalize for observation space [-1, 1]
        norm_dist = np.clip(dist_to_goal / 3.0, 0, 1.0) # Normalize by typical max distance
        norm_angle = angle_to_goal / np.pi

        goal_obs = np.array([norm_dist, norm_angle], dtype=np.float32)

        # Get the base proprioceptive observation
        proprio_obs = self.base_env._get_obs()

        # Combine into the final observation
        final_obs = np.concatenate([goal_obs, proprio_obs])
        return final_obs

    def _set_turn_mode_physics(self, is_turning: bool):
        """Dynamically applies or removes joint limits for turning."""
        leg_turn_limit_rad = np.deg2rad(15.0)
        for j_index in self.base_env.joint_indices:
            joint_info = p.getJointInfo(self.base_env.robot_id, j_index, physicsClientId=self.client_id)
            joint_name = joint_info[1].decode("utf-8")

            if is_turning and "Leg_Joint" in joint_name:
                p.changeDynamics(self.base_env.robot_id, j_index,
                                 jointLowerLimit=-leg_turn_limit_rad,
                                 jointUpperLimit=leg_turn_limit_rad,
                                 physicsClientId=self.client_id)
            else:
                p.changeDynamics(self.base_env.robot_id, j_index, **self.original_joint_limits[j_index])

    def close(self):
        self.base_env.close()

    def enable_gui(self, enable: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:

        current_render_mode = self.base_env.render_mode
        desired_mode = "GUI" if enable else "DIRECT"
        if current_render_mode == desired_mode:
            return self.reset()

        # --- IMPORTANT: Update the overlay's render_mode before recreating ---
        self.render_mode = desired_mode

        # Re-create the base environment with the new render mode
        self.base_env.close()
        self.base_env = BioloidAntLikeEnv(render_mode=desired_mode, max_steps=self.max_steps * self.command_duration_steps)
        
        # Re-create the visual shape if we are now in GUI mode
        if self.render_mode == "GUI":
             self.goal_marker_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.8], physicsClientId=self.client_id
            )

        self.client_id = self.base_env.client_id
        return self.reset()


if __name__ == "__main__":

    env = BioloidEnvPointGoal(render_mode="GUI")
    obs, info = env.reset()
    print("Point-Goal Observation dim:", obs.shape)

    for ep in range(3):
        ep_reward = 0.0
        obs, info = env.reset()
        for t in range(env.max_steps):
            # Take a random high-level action
            action = env.action_space.sample()
            print(f"Step {t}: Executing Command: {NavigationCommands(action).name}")

            obs, r, term, trunc, info = env.step(action)
            ep_reward += r

            print(f"  -> Reward: {r:.3f}, Dist: {info['dist_to_goal']:.2f}m")

            if term or trunc:
                print(f"Episode {ep+1} finished after {t+1} steps. Total Reward: {ep_reward:.2f}. Goal Reached: {info['goal_reached']}")
                break
    env.close()
