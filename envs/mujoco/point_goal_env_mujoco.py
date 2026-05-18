import math
from typing import Optional, Tuple, Dict, Any
from enum import IntEnum
import sys
import os

import gymnasium as gym
import numpy as np
import torch
import mujoco
from gymnasium import spaces


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from envs.mujoco.quadruped_mujoco_env import BioloidMujocoEnv
from brains.sac.sac_agent import soft_actor_critic_agent


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


def transform_observation_for_expert_mujoco(obs: np.ndarray, base_quat_pb: np.ndarray) -> np.ndarray:
    """
    Transforms the observation for the walker expert, matching the transformation done in PyBullet.
    base_quat_pb: [x, y, z, w]
    """
    # 1. Get Yaw from Quat (x,y,z,w)
    # math: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    x, y, z, w = base_quat_pb
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    # 2. Create inverse yaw quat
    cos_hy = math.cos(-yaw * 0.5)
    sin_hy = math.sin(-yaw * 0.5)
    inv_yaw_quat = np.array([0, 0, sin_hy, cos_hy]) # [x, y, z, w]
    
    # 3. Rotate angular velocity (obs[19:22])
    # ang_vel is [wx, wy, wz]
    # We use a helper to rotate a vector by a quaternion
    def rotate_vec_by_quat(v, q):
        # q is [x, y, z, w]
        qx, qy, qz, qw = q
        vx, vy, vz = v
        # res = v + 2 * cross(q_xyz, cross(q_xyz, v) + qw * v)
        tx = 2 * (qy * vz - qz * vy)
        ty = 2 * (qz * vx - qx * vz)
        tz = 2 * (qx * vy - qy * vx)
        return np.array([
            vx + qw * tx + (qy * tz - qz * ty),
            vy + qw * ty + (qz * tx - qx * tz),
            vz + qw * tz + (qx * ty - qy * tx)
        ])

    world_ang_vel = obs[19:22]
    rotated_ang_vel = rotate_vec_by_quat(world_ang_vel, inv_yaw_quat)
    obs[19:22] = rotated_ang_vel
    
    # 4. Rotate base orientation (obs[22:26])
    # Quat mult: res = q1 * q2
    def quat_mult(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ])
    
    rotated_base_quat = quat_mult(inv_yaw_quat, base_quat_pb)
    obs[22:26] = rotated_base_quat
    
    return obs


class NavigationCommands(IntEnum):
    WALK = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3


class BioloidMujocoEnvPointGoal(gym.Env):
    """
    MuJoCo high-level 'point-goal' environment for navigation.
    Matches the logic of BioloidEnvPointGoal in PyBullet.
    """
    def __init__(
        self,
        xml_path: str = os.path.join("assets", "mujoco", "Bioloid_Quadruped_Model", "Bioloid_Quadruped_Model.xml"),
        render_mode: str = None,
        max_steps: int = 1000,
        walker_path: str = os.path.join("models", "mujoco", "sac", "Walker.pth"),
        turn_left_path: str = os.path.join("models", "mujoco", "sac", "Left_Turner.pth"),
        turn_right_path: str = os.path.join("models", "mujoco", "sac", "Right_Turner.pth"),
        command_duration_steps: int = 30,
        goal_distance_threshold: float = 0.2,
    ):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.render_mode = render_mode
        
        # --- 1. Create underlying environment ---
        self.base_env = BioloidMujocoEnv(
            xml_path=xml_path, 
            render_mode=render_mode, 
            max_steps=max_steps * command_duration_steps
        )
        
        # --- 2. Load experts ---
        self.walker_agent = load_expert_agent(walker_path, self.base_env, self.device)
        self.turn_left_agent = load_expert_agent(turn_left_path, self.base_env, self.device)
        self.turn_right_agent = load_expert_agent(turn_right_path, self.base_env, self.device)
        
        # --- 3. Observation and Action Spaces ---
        self.action_space = spaces.Discrete(len(NavigationCommands))
        base_obs_dim = self.base_env.observation_space.shape[0]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 + base_obs_dim,), dtype=np.float32)
        
        # --- 4. State ---
        self.max_steps = max_steps
        self.command_duration_steps = command_duration_steps
        self.goal_dist_thresh = goal_distance_threshold
        self.step_count = 0
        self.goal_position = np.array([0.0, 0.0])
        self.last_dist_to_goal = 0.0
        self.last_angle_to_goal = 0.0
        self.np_random = np.random.RandomState()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None, set_new_goal: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random.seed(seed)
        
        self.base_env.reset(seed=seed, options=options)
        self.step_count = 0
        
        if set_new_goal:
            radius = self.np_random.uniform(1.5, 2.5)
            angle = self.np_random.uniform(-np.pi, np.pi)
            self.goal_position = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            
        self.last_dist_to_goal, self.last_angle_to_goal = self._get_raw_goal_metrics()
        obs = self._get_point_goal_observation()
        return obs, {}

    def _get_raw_goal_metrics(self) -> Tuple[float, float]:
        # MuJoCo qpos: [x, y, z, w, x, y, z, ...]
        base_pos_xy = self.base_env.data.qpos[:2]
        mj_quat = self.base_env.data.qpos[3:7] # w, x, y, z
        
        vec_to_goal = self.goal_position - base_pos_xy
        dist_to_goal = np.linalg.norm(vec_to_goal)
        
        # Calculate robot heading using quat
        # Heading vec is the rotated X-axis [1, 0, 0]
        w, x, y, z = mj_quat
        heading_x = 1.0 - 2.0 * (y**2 + z**2)
        heading_y = 2.0 * (x*y + w*z)
        
        robot_heading = math.atan2(heading_y, heading_x)
        goal_heading = math.atan2(vec_to_goal[1], vec_to_goal[0])
        
        angle_to_goal = goal_heading - robot_heading
        # Normalize to [-pi, pi]
        angle_to_goal = (angle_to_goal + np.pi) % (2.0 * np.pi) - np.pi
        
        return float(dist_to_goal), float(angle_to_goal)

    def _get_point_goal_observation(self) -> np.ndarray:
        dist, angle = self._get_raw_goal_metrics()
        # Normalized obs
        norm_dist = np.clip(dist / 3.0, 0.0, 1.0)
        norm_angle = angle / np.pi
        goal_obs = np.array([norm_dist, norm_angle], dtype=np.float32)
        
        proprio_obs = self.base_env._get_obs()
        return np.concatenate([goal_obs, proprio_obs])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        command = NavigationCommands(action)
        sub_step_reward = 0.0
        
        for _ in range(self.command_duration_steps):
            base_obs = self.base_env._get_obs()
            
            if command == NavigationCommands.WALK:
                # MuJoCo uses w,x,y,z natively for qpos. 
                # base_obs[22:26] is x,y,z,w (scaled for agent compatibility)
                expert_obs = transform_observation_for_expert_mujoco(base_obs.copy(), base_obs[22:26])
                expert_action = self.walker_agent.select_action(expert_obs, eval=True)
            elif command == NavigationCommands.TURN_LEFT:
                expert_action = self.turn_left_agent.select_action(base_obs, eval=True)
            elif command == NavigationCommands.TURN_RIGHT:
                expert_action = self.turn_right_agent.select_action(base_obs, eval=True)
            else: # STOP
                # Generic PD to hold position
                jpos = self.base_env.data.qpos[7:15]
                jvel = self.base_env.data.qvel[6:14]
                torque = -2.5 * jpos - 0.15 * jvel
                expert_action = np.clip(torque, -0.7, 0.7)
            
            _, r, term, trunc, _ = self.base_env.step(expert_action)
            self.base_env.render() # Keep the GUI smooth during the internal loop
            sub_step_reward += r
            if term: break
            
        point_goal_obs = self._get_point_goal_observation()
        dist_to_goal, angle_to_goal = self._get_raw_goal_metrics()
        
        # Reward Logic mirroring PyBullet
        dist_reduction = self.last_dist_to_goal - dist_to_goal
        angle_reduction = abs(self.last_angle_to_goal) - abs(angle_to_goal)
        
        goal_reached = dist_to_goal < self.goal_dist_thresh
        goal_bonus = 200.0 if goal_reached else 0.0
        
        # High-level Reward
        reward = (50.0 * dist_reduction) + (10.0 * angle_reduction) + goal_bonus - 0.1
        
        self.step_count += 1
        terminated = self.base_env._is_terminated() or goal_reached
        truncated = self.step_count >= self.max_steps
        
        self.last_dist_to_goal = dist_to_goal
        self.last_angle_to_goal = angle_to_goal
        
        info = {"dist_to_goal": dist_to_goal, "goal_reached": goal_reached}
        return point_goal_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.base_env.close()
