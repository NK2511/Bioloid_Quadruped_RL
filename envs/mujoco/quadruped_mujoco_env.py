import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import os

class BioloidMujocoEnv(gym.Env):
    """
    MuJoCo environment for the Bioloid Quadruped optimized for Forward Walking.
    Matches the original 'envs/quadruped_env.py' logic and 240Hz physics.
    """
    def __init__(
        self,
        xml_path: str = os.path.join("assets", "mujoco", "Bioloid_Quadruped_Model", "Bioloid_Quadruped_Model.xml"),
        render_mode: str = None,
        frame_skip: int = 4,
        torque_limit: float = 0.255, 
        ctrl_cost_weight: float = 0.03, # Matches original quadruped_env.py
        contact_cost_weight: float = 2e-4,
        alive_bonus: float = 0.05,
        max_steps: int = 1000,
    ):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.frame_skip = frame_skip
        self.torque_limit = torque_limit
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self.alive_bonus = alive_bonus
        self.max_steps = max_steps
        self.step_count = 0
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        # Obs: jpos(8), jvel(8), base_lin_vel(3), base_ang_vel(3), base_quat(4), height(1), touch_ratio(1) = 28
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        
        self.leg_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in 
                             ["FR_Leg_geom", "FL_Leg_geom", "BR_Leg_geom", "BL_Leg_geom"]]
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

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
        # Joint states (8x pos, 8x vel)
        jpos = self.data.qpos[7:15].copy()
        jvel = self.data.qvel[6:14].copy()
        
        # Base state (World frame)
        base_lin_vel = self.data.qvel[:3].copy()
        base_ang_vel = self.data.qvel[3:6].copy()
        
        # Base Orientation (Quaternion)
        # MuJoCo uses (w, x, y, z), but PyBullet usually uses (x, y, z, w).
        # We'll use (x, y, z, w) to match the original model's expectation.
        mj_quat = self.data.qpos[3:7] # w, x, y, z
        pb_quat = np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]], dtype=np.float32)
        
        base_height = np.array([self.data.qpos[2]], dtype=np.float32)
        
        # Contact state sensing (Simplified proxy for 'feet_on_ground_ratio')
        contacts = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 in self.leg_geom_ids and contact.geom2 == self.floor_geom_id) or \
               (contact.geom2 in self.leg_geom_ids and contact.geom1 == self.floor_geom_id):
                contacts += 1
        touch_ratio = np.array([contacts / 4.0], dtype=np.float32)
        
        return np.concatenate([
            jpos, jvel, base_lin_vel, base_ang_vel, pb_quat, base_height, touch_ratio
        ]).astype(np.float32)

    def step(self, action):
        # Apply scaled torque control
        # PB setJointMotorControlArray uses 0.255 torque limit.
        # MuJoCo actuators are defined in Bioloid_Quadruped_Model.xml.
        # We manually scale control to match the torque limit of the URDF.
        action = np.clip(action, -1.0, 1.0)
        mj_ctrl = (action * self.torque_limit) / 2.5 # 2.5 is the actuator gain in XML
        self.data.ctrl[:] = mj_ctrl
        
        prev_x = self.data.qpos[0]
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # 1. Forward Progress Reward (Exactly like original quadruped_env.py)
        # Note: timestep is 1/240, frame_skip is 4 -> dt = 1/60 (60Hz control)
        dt = self.model.opt.timestep * self.frame_skip
        curr_x = self.data.qpos[0]
        forward_reward = (curr_x - prev_x) / max(dt, 1e-8)
        
        # 2. Control Cost
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
        
        # 3. Contact Cost (Simplified)
        contact_cost = self.contact_cost_weight * np.sum(np.square(self.data.cfrc_ext))
        
        # 4. Alive Bonus
        alive_bonus = self.alive_bonus
        
        reward = forward_reward + alive_bonus - ctrl_cost - contact_cost
        
        self.step_count += 1
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps
        
        return obs, float(reward), terminated, truncated, {}

    def _is_terminated(self):
        # Fall detection (Matches PyBullet's logic)
        z = self.data.qpos[2]
        return z < 0.05 # Terminate if base height falls below 5cm

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Set initial height to preventing falling on first frame
        self.data.qpos[2] = 0.119 
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
