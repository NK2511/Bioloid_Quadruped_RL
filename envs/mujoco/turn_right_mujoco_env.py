import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import os
import os

class BioloidMujocoTurnRightEnv(gym.Env):
    """
    MuJoCo environment for the Bioloid Quadruped optimized for Turning Right.
    Matches 'envs/turn_right_env.py' exactly.
    """
    def __init__(
        self,
        xml_path: str = os.path.join("assets", "mujoco", "Bioloid_Quadruped_Model", "Bioloid_Quadruped_Model.xml"),
        render_mode: str = None,
        frame_skip: int = 4,
        torque_limit: float = 0.255, 
        max_steps: int = 500,
        w_turn_velocity: float = 1.5,
        w_movement: float = 0.1,
        w_height: float = 0.8,
        w_home: float = 0.3,
        w_joint_pose: float = 0.05,
        w_tilt: float = 0.5,
    ):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.frame_skip = frame_skip
        self.torque_limit = torque_limit
        self.max_steps = max_steps
        self.step_count = 0
        
        self.w_turn_velocity = w_turn_velocity
        self.w_movement = w_movement
        self.w_height = w_height
        self.w_home = w_home
        self.w_joint_pose = w_joint_pose
        self.w_tilt = w_tilt
        self.target_height = 0.119
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        # Obs: jpos(8), jvel(8), base_lin_vel(3), base_ang_vel(3), base_quat(4), height(1), touch_ratio(1) = 28
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

        # Matches joint_names in PyBullet env for graph compatibility
        self.joint_names = [
            'base_link_FR_Hip_Joint', 'FR_Hip_FR_Leg_Joint',
            'base_link_FL_Hip_Joint', 'FL_Hip_FL_Leg_Joint',
            'base_link_BR_Hip_Joint', 'BR_Hip_BR_Leg_Joint',
            'base_link_BL_Hip_Joint', 'BL_Hip_BL_Leg_Joint',
        ]
        
        self.viewer = None
        self.render_mode = render_mode

    def _get_obs(self):
        jpos = self.data.qpos[7:15].copy()
        jvel = self.data.qvel[6:14].copy()
        
        # Base state (World frame)
        base_lin_vel = self.data.qvel[:3].copy()
        base_ang_vel = self.data.qvel[3:6].copy()
        
        mj_quat = self.data.qpos[3:7]
        pb_quat = np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]], dtype=np.float32)
        
        base_height = np.array([self.data.qpos[2]], dtype=np.float32)
        touch_ratio = np.array([1.0], dtype=np.float32) # Dummy to match dim
        
        return np.concatenate([
            jpos, jvel, base_lin_vel, base_ang_vel, pb_quat, base_height, touch_ratio
        ]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = (action * self.torque_limit) / 2.5 
        
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        base_pos = self.data.qpos[:3]
        lin_vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        # 1. Turn Reward (Right is negative Z-ang velocity)
        turn_reward = np.clip(-ang_vel[2], -np.inf, 2.5)
        
        # 2. Penalties
        movement_penalty = np.linalg.norm(lin_vel[0:2])
        height_penalty = (base_pos[2] - self.target_height)**2
        home_penalty = np.linalg.norm(base_pos[0:2])
        
        jpos = self.data.qpos[7:15]
        joint_pose_penalty = np.sum(np.square(jpos))
        
        mj_quat = self.data.qpos[3:7] # w, x, y, z
        roll, pitch, _ = self._quat_to_euler(mj_quat)
        tilt_penalty = roll**2 + pitch**2
        
        reward = (self.w_turn_velocity * turn_reward - 
                  self.w_movement * movement_penalty - 
                  self.w_height * height_penalty - 
                  self.w_home * home_penalty - 
                  self.w_joint_pose * joint_pose_penalty - 
                  self.w_tilt * tilt_penalty)
        
        self.step_count += 1
        terminated = self._is_terminated(base_pos)
        truncated = self.step_count >= self.max_steps
        
        return obs, float(reward), terminated, truncated, {}

    def _quat_to_euler(self, q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def _is_terminated(self, base_pos):
        z = base_pos[2]
        too_low = z < max(0.04, 0.5 * self.target_height)
        moved_too_far = np.linalg.norm(base_pos[0:2]) > 0.5
        return too_low or moved_too_far

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = self.target_height
        self.step_count = 0
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
             self.viewer.close()
