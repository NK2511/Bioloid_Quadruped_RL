import math
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class BioloidAntLikeEnvTurnOnly(gym.Env):
    """
    A specialized environment to train a robot to turn RIGHT in place.
    """

    metadata = {"render_modes": ["GUI", "DIRECT"]}

    def __init__(
        self,
        urdf_path: str = r"assets\Bioloid_Quadruped_Model.urdf",
        render_mode: str = "DIRECT",
        frame_skip: int = 4,
        time_step: float = 1.0 / 240.0,

        body_height_m: float = 0.119,
        total_mass_kg: float = 0.85,
        leg_link_length_m: float = 0.0815,
        max_steps: int = 500,
        w_turn_velocity: float = 1.5,
        w_movement: float = 0.1,
        w_height: float = 0.8,
        w_home: float = 0.3,
        w_joint_pose: float = 0.05,
        w_tilt: float = 0.5,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.urdf_path = urdf_path
        self.body_height_m = float(body_height_m)
        self.total_mass_kg = float(total_mass_kg)
        self.leg_link_length_m = float(leg_link_length_m)

        if self.render_mode == "GUI":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        self.frame_skip = int(frame_skip)
        self.time_step = float(time_step)
        self.torque_limit = 0.255
        self.max_steps = int(max_steps)

        # Store reward weights
        self.w_turn_velocity = float(w_turn_velocity)
        self.w_movement = float(w_movement)
        self.w_height = float(w_height)
        self.w_home = float(w_home)
        self.w_joint_pose = float(w_joint_pose)
        self.w_tilt = float(w_tilt)

        self.joint_names: List[str] = [
            'base_link_FR_Hip_Joint', 'FR_Hip_FR_Leg_Joint', 'base_link_FL_Hip_Joint', 'FL_Hip_FL_Leg_Joint',
            'base_link_BR_Hip_Joint', 'BR_Hip_BR_Leg_Joint', 'base_link_BL_Hip_Joint', 'BL_Hip_BL_Leg_Joint',
        ]
        self.robot_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.joint_indices: List[int] = []

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.joint_names),), dtype=np.float32)
        # Obs: [original_obs(28)] = 28 dims. The target speed is removed.
        obs_dim = 28
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.np_random = np.random.RandomState()
        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random.seed(seed)
        self.step_count = 0

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        start_pos = [0.0, 0.0, float(self.body_height_m)]
        start_quat = [0.0, 0.0, 0.0, 1.0]
        self.robot_id = p.loadURDF(self.urdf_path, start_pos, start_quat, useFixedBase=False, physicsClientId=self.client_id)

        self.joint_indices = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        name_to_joint_index: Dict[str, int] = {p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)[1].decode("utf-8"): j for j in range(num_joints)}
        for name in self.joint_names:
            if name in name_to_joint_index:
                self.joint_indices.append(name_to_joint_index[name])
        
        # --- Set Hard Joint Limits ---
        foot_joint_limit_rad = np.deg2rad(15.0)
        for j in self.joint_indices:
            joint_info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            joint_name = joint_info[1].decode("utf-8")

            p.resetJointState(self.robot_id, j, 0.0, 0.0, physicsClientId=self.client_id)

            # Apply the hard limit ONLY to the lower leg joints (foot joints).
            if "Leg_Joint" in joint_name:
                p.changeDynamics(self.robot_id, j, jointLowerLimit=-foot_joint_limit_rad, jointUpperLimit=foot_joint_limit_rad, physicsClientId=self.client_id)

            p.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=j, controlMode=p.VELOCITY_CONTROL, force=0.0)

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        torques = (action * self.torque_limit).tolist()
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id, jointIndices=self.joint_indices, controlMode=p.TORQUE_CONTROL, forces=torques)

        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)

        obs = self._get_obs()
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)

        max_rewardable_speed = 2.5
        turn_reward = np.clip(-ang_vel[2], -np.inf, max_rewardable_speed)

        # Penalties for stability and staying in place
        movement_penalty = np.linalg.norm(lin_vel[0:2]) # Penalize XY translational velocity
        height_penalty = (base_pos[2] - self.body_height_m)**2 # Penalize deviation from target height
        home_penalty = np.linalg.norm(base_pos[0:2])


        joint_states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self.client_id)
        joint_pos = np.array([state[0] for state in joint_states])
        joint_pose_penalty = np.sum(np.square(joint_pos))
        

        roll, pitch, _ = p.getEulerFromQuaternion(base_quat)
        tilt_penalty = roll**2 + pitch**2

        # Combine rewards using the weights defined in __init__
        reward = self.w_turn_velocity * turn_reward - self.w_movement * movement_penalty - self.w_height * height_penalty - self.w_home * home_penalty - self.w_joint_pose * joint_pose_penalty - self.w_tilt * tilt_penalty

        self.step_count += 1
        terminated = self._is_terminated(base_pos)
        truncated = self.step_count >= self.max_steps

        return obs, float(reward), bool(terminated), bool(truncated), {}

    def _get_obs(self) -> np.ndarray:
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)

        jpos, jvel = [], []
        for j in self.joint_indices:
            js = p.getJointState(self.robot_id, j)
            jpos.append(js[0])
            jvel.append(js[1])

        # This is the standard 28-dim observation from the walking environment
        original_obs = np.concatenate([
            np.asarray(jpos, dtype=np.float32), np.asarray(jvel, dtype=np.float32),
            np.asarray(lin_vel, dtype=np.float32), np.asarray(ang_vel, dtype=np.float32),
            np.asarray(base_quat, dtype=np.float32), np.array([base_pos[2]], dtype=np.float32),
            np.array([1.0], dtype=np.float32), # feet_on_ground_ratio (not used but keeps dim consistent)
        ])

        return original_obs

    def _is_terminated(self, base_pos) -> bool:
        base_z = float(base_pos[2])
        
        # --- Termination Conditions ---
        # 1. Robot has fallen or collapsed
        too_low = base_z < max(0.04, 0.5 * self.body_height_m)
        # 2. Robot has drifted too far from the origin
        moved_too_far = np.linalg.norm(base_pos[0:2]) > 0.5
        # The over-extension termination is no longer needed as it's now a physical constraint.

        return bool(too_low or moved_too_far)

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def enable_gui(self, enable: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        desired = "GUI" if enable else "DIRECT"
        if desired == self.render_mode:
            return self.reset()
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)
        self.render_mode = desired
        if self.render_mode == "GUI":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        return self.reset()

