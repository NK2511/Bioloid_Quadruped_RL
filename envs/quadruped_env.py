import math
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces


class BioloidAntLikeEnv(gym.Env):
    """
    Ant-like PyBullet environment for an 8-DoF quadruped (Bioloid model).

    Key design choices to mimic AntBulletEnv behavior:
    - Continuous action space (8,) with torque control and a torque limit similar to Ant power.
    - Reward = forward_progress/dt + alive_bonus - ctrl_cost - contact_cost.
    - Termination on collapse (height or excessive tilt).
    - Frame skipping (multiple physics steps per action).
    - Observation blends joint states, base velocities, orientation, height, and contact features (~28 dims).

    Gymnasium API:
      - reset(seed) -> (obs, info)
      - step(action) -> (obs, reward, terminated, truncated, info)
    """

    metadata = {"render_modes": ["GUI", "DIRECT"]}

    def __init__(
        self,
        urdf_path: str = r"C:\\Users\\nandh\\Downloads\\Bioloid_Quadruped_Model\\Bioloid_Quadruped_Model.urdf",
        render_mode: str = "DIRECT",
        frame_skip: int = 4,
        time_step: float = 1.0 / 240.0,
        # Physical robot parameters (from user):
        body_height_m: float = 0.119,          # base height from ground at neutral
        hip_link_length_m: float = 0.05945,    # in XY plane
        leg_link_length_m: float = 0.0815,     # downward link length
        total_mass_kg: float = 0.85,           # total mass target
        # Control/reward weights tuned for a smaller robot:
        torque_limit: Optional[float] = None,  # if None, computed from mass & leg length
        ctrl_cost_weight: float = 0.03,        # slightly smaller than Ant default
        contact_cost_weight: float = 2e-4,     # scaled for ~0.85kg
        alive_bonus: float = 0.05,
        max_steps: int = 1000,
    ) -> None:
        super().__init__()

        assert render_mode in ("GUI", "DIRECT"), "render_mode must be 'GUI' or 'DIRECT'"
        self.render_mode = render_mode
        self.urdf_path = urdf_path

        # Store physical parameters
        self.body_height_m = float(body_height_m)
        self.hip_link_length_m = float(hip_link_length_m)
        self.leg_link_length_m = float(leg_link_length_m)
        self.total_mass_kg = float(total_mass_kg)

        # Connect to physics
        if self.render_mode == "GUI":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)

        # Simulation params
        self.frame_skip = int(frame_skip)
        self.time_step = float(time_step)
        # If torque_limit not provided, compute a conservative default based on size/mass
        self.torque_limit = float(torque_limit) if torque_limit is not None else self._compute_default_torque_limit(self.total_mass_kg, self.leg_link_length_m)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.contact_cost_weight = float(contact_cost_weight)
        self.alive_bonus = float(alive_bonus)
        self.max_steps = int(max_steps)

        # Joint and foot names (override if your URDF differs)
        self.joint_names: List[str] = [
            'base_link_FR_Hip_Joint', 'FR_Hip_FR_Leg_Joint',
            'base_link_FL_Hip_Joint', 'FL_Hip_FL_Leg_Joint',
            'base_link_BR_Hip_Joint', 'BR_Hip_BR_Leg_Joint',
            'base_link_BL_Hip_Joint', 'BL_Hip_BL_Leg_Joint',
        ]
        self.foot_link_names: List[str] = [
            'FR_Leg_Link', 'FL_Leg_Link', 'BR_Leg_Link', 'BL_Leg_Link'
        ]

        self.robot_id: Optional[int] = None
        self.plane_id: Optional[int] = None
        self.joint_indices: List[int] = []
        self.foot_link_indices: List[int] = []

        # Action/observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(self.joint_names),), dtype=np.float32
        )

        # We'll build a 28-dim observation similar in spirit to AntBulletEnv:
        # [joint_pos(8), joint_vel(8), base_lin_vel(3), base_ang_vel(3), base_quat(4), base_height(1), feet_on_ground_ratio(1)] = 28
        obs_dim = 8 + 8 + 3 + 3 + 4 + 1 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode state
        self.step_count = 0
        self.prev_base_x = 0.0
        self.np_random = np.random.RandomState()

        # Do initial reset to populate indices and state
        self.reset()

    # ----------------------------- Gymnasium API -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.np_random.seed(seed)
        self.step_count = 0

        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Spawn robot at nominal body height
        start_pos = [0.0, 0.0, float(self.body_height_m)]
        start_quat = [0.0, 0.0, 0.0, 1.0]
        self.robot_id = p.loadURDF(
            self.urdf_path, start_pos, start_quat, useFixedBase=False, physicsClientId=self.client_id
        )

        # Rescale masses to match total_mass_kg while preserving relative distribution
        try:
            if self.total_mass_kg is not None and self.total_mass_kg > 0:
                self._rescale_total_mass(self.total_mass_kg)
        except Exception:
            # Gracefully ignore if URDF/dynamics prevent mass change
            pass

        # Map joints and feet
        self.joint_indices = []
        self.foot_link_indices = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        name_to_joint_index: Dict[str, int] = {}
        link_name_set = set(self.foot_link_names)

        for j in range(num_joints):
            ji = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            joint_name = ji[1].decode("utf-8")
            link_name = ji[12].decode("utf-8")
            name_to_joint_index[joint_name] = j
            if link_name in link_name_set:
                self.foot_link_indices.append(j)

        for name in self.joint_names:
            if name in name_to_joint_index:
                self.joint_indices.append(name_to_joint_index[name])

        # Fallback: if named joints not found, take the first 8 revolute/prismatic joints
        if len(self.joint_indices) != len(self.joint_names):
            self.joint_indices = []
            for j in range(num_joints):
                ji = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
                joint_type = ji[2]
                if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                    self.joint_indices.append(j)
                if len(self.joint_indices) == len(self.joint_names):
                    break

        # Reset joint states and disable default motors
        for j in self.joint_indices:
            p.resetJointState(self.robot_id, j, 0.0, 0.0, physicsClientId=self.client_id)
            # Disable velocity motors so TORQUE_CONTROL takes effect
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=j,
                controlMode=p.VELOCITY_CONTROL,
                force=0.0,
                physicsClientId=self.client_id,
            )

        # Let the robot settle briefly
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client_id)

        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        self.prev_base_x = float(base_pos[0])

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "joint_indices": self.joint_indices,
            "foot_link_indices": self.foot_link_indices,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply torques scaled by torque_limit
        torques = (action * self.torque_limit).tolist()
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.client_id,
        )

        # Simulate with frame skip
        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.client_id)

        obs = self._get_obs()

        # Reward terms
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        base_x = float(base_pos[0])
        dt = self.time_step * self.frame_skip
        forward_progress = (base_x - self.prev_base_x) / max(dt, 1e-8)
        self.prev_base_x = base_x

        forward_reward = forward_progress
        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        contact_cost = self.contact_cost_weight * self._sum_contact_forces()

        reward = forward_reward + self.alive_bonus - ctrl_cost - contact_cost

        self.step_count += 1
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps

        info: Dict[str, Any] = {
            "forward_reward": forward_reward,
            "alive_bonus": self.alive_bonus,
            "ctrl_cost": ctrl_cost,
            "contact_cost": contact_cost,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def enable_gui(self, enable: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Toggle GUI on/off. Reconnects physics and resets the environment.
        Returns: (obs, info) from the post-toggle reset.
        Note: Toggling will reset the episode.
        """
        desired = "GUI" if enable else "DIRECT"
        if desired == self.render_mode:
            return self.reset()
        if p.isConnected(self.client_id):
            try:
                p.disconnect(self.client_id)
            except Exception:
                pass
        self.render_mode = desired
        if self.render_mode == "GUI":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.time_step, physicsClientId=self.client_id)
        return self.reset()

    # ----------------------------- Helpers -----------------------------
    def _get_obs(self) -> np.ndarray:
        # Joint positions/velocities for the controlled joints
        jpos = []
        jvel = []
        for j in self.joint_indices:
            js = p.getJointState(self.robot_id, j, physicsClientId=self.client_id)
            jpos.append(js[0])
            jvel.append(js[1])

        # Base state
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.client_id)

        base_lin_vel = np.asarray(lin_vel, dtype=np.float32)
        base_ang_vel = np.asarray(ang_vel, dtype=np.float32)
        base_quat = np.asarray(base_quat, dtype=np.float32)  # x, y, z, w
        base_height = np.float32(base_pos[2])

        feet_on_ground_ratio = np.float32(self._feet_on_ground_ratio())

        obs = np.concatenate([
            np.asarray(jpos, dtype=np.float32),
            np.asarray(jvel, dtype=np.float32),
            base_lin_vel,
            base_ang_vel,
            base_quat,
            np.array([base_height], dtype=np.float32),
            np.array([feet_on_ground_ratio], dtype=np.float32),
        ], axis=0)
        return obs

    def _is_terminated(self) -> bool:
        # Terminate on collapse or excessive tilt, scaled to robot size
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.client_id)
        base_z = float(base_pos[2])
        roll, pitch, _ = p.getEulerFromQuaternion(base_quat)

        # Thresholds relative to nominal body height
        too_low = base_z < max(0.04, 0.5 * self.body_height_m)
        too_high = base_z > 2.5 * self.body_height_m
        too_tilted = (abs(roll) > 1.0) or (abs(pitch) > 1.0)
        return bool(too_low or too_high or too_tilted)

    def _feet_on_ground_ratio(self) -> float:
        # Count number of feet in contact with the plane
        if not self.foot_link_indices:
            return 0.0
        in_contact = 0
        for link_idx in self.foot_link_indices:
            cps = p.getContactPoints(
                bodyA=self.robot_id, bodyB=self.plane_id, linkIndexA=link_idx, linkIndexB=-1, physicsClientId=self.client_id
            )
            if len(cps) > 0:
                in_contact += 1
        return float(in_contact) / float(len(self.foot_link_indices))

    def _sum_contact_forces(self) -> float:
        # Approximate contact cost from normal forces between robot links and plane or other bodies
        total = 0.0
        # With plane
        cps = p.getContactPoints(bodyA=self.robot_id, bodyB=self.plane_id, physicsClientId=self.client_id)
        for cp in cps:
            total += abs(cp[9])  # normal force
        # With any other bodies (rare here), loop over links for robustness
        num_bodies = p.getNumBodies(physicsClientId=self.client_id)
        for b in range(num_bodies):
            if b in (self.robot_id, self.plane_id):
                continue
            cps_other = p.getContactPoints(bodyA=self.robot_id, bodyB=b, physicsClientId=self.client_id)
            for cp in cps_other:
                total += abs(cp[9])
        return float(total)

    def _rescale_total_mass(self, target_mass: float) -> None:
        """Rescales all link masses to match target total mass while preserving ratios."""
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        # Collect current masses including base (-1)
        indices = [-1] + list(range(num_joints))
        masses = []
        for idx in indices:
            md = p.getDynamicsInfo(self.robot_id, idx, physicsClientId=self.client_id)
            masses.append(float(md[0]))
        current_total = float(sum(masses))
        if current_total <= 0:
            return
        scale = float(target_mass) / current_total
        for idx, m in zip(indices, masses):
            try:
                p.changeDynamics(self.robot_id, idx, mass=m * scale, physicsClientId=self.client_id)
            except Exception:
                pass

    def _compute_default_torque_limit(self, total_mass_kg: float, leg_length_m: float) -> float:
        """Heuristic torque limit per joint based on size/mass.
        Rough estimate: torque per leg to hold body ~ (m*g*leg_length)/4. Multiply by a margin for agility.
        """
        g = 9.81
        per_leg = (total_mass_kg * g * max(leg_length_m, 1e-3)) / 4.0
        margin = 3.0  # allow stronger torques than static hold
        per_joint = per_leg * margin * 0.5  # distribute between two joints per leg
        return float(max(0.2, min(2.5, per_joint)))


# Optional: simple self-test when run as a script
if __name__ == "__main__":
    env = BioloidAntLikeEnv(render_mode="GUI")
    obs, info = env.reset()
    print("Observation dim:", obs.shape)
    ep_reward = 0.0
    for t in range(200):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        ep_reward += r
        if term or trunc:
            print("Episode end at step", t, "reward=", ep_reward)
            obs, info = env.reset()
            ep_reward = 0.0
    env.close()
