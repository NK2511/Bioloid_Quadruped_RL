import argparse
import time
from typing import List
import sys
import numpy as np
import torch
import pybullet as p
import os

"""
Teleoperation Script
====================

This script allows interactive control of the Bioloid quadruped robot using the keyboard.
It utilizes pre-trained low-level skills (walking, turning) that are triggered by user input.

Controls:
    - UP ARROW:     Execute 'Walk Forward' policy.
    - LEFT ARROW:   Execute 'Turn Left' policy.
    - RIGHT ARROW:  Execute 'Turn Right' policy.
    - 'r':          Reset robot position.

The script instantiates the `BioloidAntLikeEnv` and loads the expert agents from `models/`.
"""

# Add the project's root directory and the 'env' directory to the Python path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT) # For sac_agent
# sys.path.append(os.path.join(PROJECT_ROOT, 'envs')) # Not needed if importing from envs package
from envs.quadruped_env import BioloidAntLikeEnv
from sac.sac_agent import soft_actor_critic_agent


def transform_observation(obs: np.ndarray, base_quat_world: list) -> np.ndarray:
    """
    Transforms world-frame observations into a canonical, robot-centric frame.
    This "tricks" a model trained only on the X-axis to walk in its current
    forward direction.
    """
    _, _, yaw = p.getEulerFromQuaternion(base_quat_world)
    inverse_yaw_quat = p.getQuaternionFromEuler([0, 0, -yaw])
    world_ang_vel = obs[19:22]
    rotated_ang_vel, _ = p.multiplyTransforms([0, 0, 0], inverse_yaw_quat, world_ang_vel, [0, 0, 0, 1])
    _, rotated_base_quat = p.multiplyTransforms([0, 0, 0], inverse_yaw_quat, [0, 0, 0], base_quat_world)
    obs[19:22] = rotated_ang_vel
    obs[22:26] = rotated_base_quat
    return obs


def load_agent_from_checkpoint(model_path: str, env, device: torch.device) -> soft_actor_critic_agent:
    """
    Loads a SAC agent's policy from a checkpoint file.
    """
    if not model_path:
        raise ValueError("Model path cannot be empty.")
        
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise

    # Determine hidden size from checkpoint, default to 256
    hidden_size = checkpoint.get("hidden_size", 256)
    
    # The stopper was trained on a 29-dim obs space, but the walker on 28.
    # The core proprioceptive states are the same, so we can use the same agent
    # structure and just feed it the correct slice of observations.
    num_inputs = env.observation_space.shape[0]

    agent = soft_actor_critic_agent(
        num_inputs,
        action_space=env.action_space,
        device=device,
        hidden_size=hidden_size,
        seed=0, lr=0.0, gamma=0.0, tau=0.0, alpha=0.0,
    )

    # The checkpoint can be a dictionary with an 'actor' key or just the state_dict
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        agent.policy.load_state_dict(checkpoint['actor'])
    else:
        agent.policy.load_state_dict(checkpoint)
        
    print(f"Successfully loaded model with hidden_size={hidden_size} from: {model_path}")
    agent.policy.eval()
    return agent

class DebugDisplay:
    """A helper class to manage the real-time text display in PyBullet."""
    def __init__(self, client_id: int):
        pass # Telemetry display is disabled.

    def reset(self, base_state: tuple):
        pass # Telemetry display is disabled.

    def update(self, current_pos: np.ndarray, current_quat: np.ndarray, active_model: str):
        pass # Telemetry display is disabled.

def set_turn_mode_physics(env: BioloidAntLikeEnv, is_turning: bool, original_limits: dict):
    """
    Dynamically applies or removes joint limits for turning.
    - Leg joints are limited to +/- 15 degrees during a turn.
    - Hip joints use their original, full range of motion.
    """
    leg_turn_limit_rad = np.deg2rad(15.0)

    for j_index in env.joint_indices:
        joint_info = p.getJointInfo(env.robot_id, j_index, physicsClientId=env.client_id)
        joint_name = joint_info[1].decode("utf-8")

        # Apply tight limits to leg joints only when turning.
        # Hip joints and all other joints will use their original limits.
        if is_turning and "Leg_Joint" in joint_name:
            p.changeDynamics(env.robot_id, j_index,
                             jointLowerLimit=-leg_turn_limit_rad,
                             jointUpperLimit=leg_turn_limit_rad,
                             physicsClientId=env.client_id)
        else:
            # Restore the original, wider limits for all joints when not turning
            p.changeDynamics(env.robot_id, j_index, **original_limits[j_index])


def main():
    """
    Main function to load and interactively evaluate walker and stopper agents.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate the environment in GUI mode
    # Increase max_steps to a very large number to prevent the episode from
    # truncating automatically. This allows for indefinite interactive control.
    env = BioloidAntLikeEnv(
        render_mode="GUI",
        max_steps=1_000_000  # Effectively infinite for interactive sessions
    )

    # 2. Load all expert agents
    try:
        walker_agent = load_agent_from_checkpoint(r"models\Walker.pth", env, device)
        turn_left_agent = load_agent_from_checkpoint(r"models\Left_Turner.pth", env, device)
        turn_right_agent = load_agent_from_checkpoint(r"models\Right_Turner.pth", env, device)
    except (ValueError, FileNotFoundError) as e:
        print(f"Failed to load models. Exiting. Error: {e}")
        env.close()
        return

    print("\n--- Interactive Control Ready ---")
    print("PRESS and HOLD the UP ARROW key to walk.")
    print("PRESS and HOLD the LEFT/RIGHT ARROW keys to turn.")
    print("PRESS 'r' to reset the robot's position.")
    print("Close the PyBullet window to exit.")

    # --- Store original joint limits before starting ---
    original_joint_limits = {}
    for j_index in env.joint_indices:
        info = p.getJointInfo(env.robot_id, j_index, physicsClientId=env.client_id)
        original_joint_limits[j_index] = {'jointLowerLimit': info[8], 'jointUpperLimit': info[9]}

    # --- State variables for the control logic ---
    state, _ = env.reset()

    # --- Setup the real-time display ---
    display = DebugDisplay(client_id=env.client_id)
    display.reset(p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id))

    # Main interactive loop - runs forever until the window is closed
    while p.isConnected(env.client_id):
        # Check for keyboard events
        keys = p.getKeyboardEvents(physicsClientId=env.client_id)
        is_walking = p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN
        is_turning_left = p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN
        is_turning_right = p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN
        reset_metrics_pressed = ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED

        if reset_metrics_pressed:
            print("Resetting robot position.")
            state, _ = env.reset()

        # --- Dynamically set the physics based on the current action ---
        is_turning = is_turning_left or is_turning_right
        set_turn_mode_physics(env, is_turning, original_joint_limits)


        current_action_name = 'stop/hold' # Default action
        action_taken = False
        terminated, truncated = False, False

        if is_turning_left:
            action = turn_left_agent.select_action(state, eval=True)
            action_taken = True
            current_action_name = 'turn_left'
        elif is_turning_right:
            action = turn_right_agent.select_action(state, eval=True)
            action_taken = True
            current_action_name = 'turn_right'
        elif is_walking:

            base_pos, base_quat_world = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
            transformed_state = transform_observation(state.copy(), base_quat_world)
            action = walker_agent.select_action(transformed_state, eval=True)
            action_taken = True
            current_action_name = 'walk'
        else:

            joint_states = p.getJointStates(env.robot_id, env.joint_indices, physicsClientId=env.client_id)
            current_positions = np.array([state[0] for state in joint_states])
            current_velocities = np.array([state[1] for state in joint_states])


            max_error = np.max(np.abs(current_positions))
            
            # Define a transition range for the gains.
            # Full power above 0.1 rad, fades to zero below that.
            transition_start = 0.1  # approx 5.7 degrees
            gain_scale = np.clip(max_error / transition_start, 0.0, 1.0)

            kp = 2.5 * gain_scale # Proportional gain
            kd = 0.15 * gain_scale # Derivative gain

            torque = -kp * current_positions - kd * current_velocities
            # Apply raw torque, clipped to the environment's physical limit.
            pd_torques = np.clip(torque, -env.torque_limit, env.torque_limit)

            # Bypass env.step() and apply torques directly for one physics step
            p.setJointMotorControlArray(
                bodyUniqueId=env.robot_id,
                jointIndices=env.joint_indices,
                controlMode=p.TORQUE_CONTROL,
                forces=pd_torques.tolist(),
                physicsClientId=env.client_id,
            )
            p.stepSimulation(physicsClientId=env.client_id)
            time.sleep(env.time_step) # Sleep for one physics step

        # --- Update the on-screen display ---
        current_pos, current_quat = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id)
        display.update(np.array(current_pos), np.array(current_quat), current_action_name)

        # Step the environment ONLY if an agent took an action
        # Apply torques for PD or use SAC action, BUT run env.step() ALWAYS:
        if action_taken:
            action_to_apply = action
        else:
            action_to_apply = pd_torques  # computed earlier

        # Use env.step() ALWAYS instead of manual p.stepSimulation()
        next_state, _, terminated, truncated, _ = env.step(action_to_apply)

        # Reduce sleep time:
        time.sleep(env.time_step)   # Instead of frame_skip*time_step


        # --- Handle Episode End ---
        # If the episode ends for any reason, reset the environment and the metrics.
        if truncated:
            print("Episode ended. Resetting position and metrics.")
            state, _ = env.reset()
            display.reset(p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id))
        # If the robot has fallen (terminated), attempt a PD-controlled recovery.
        elif terminated:
            print("Robot fell! Attempting PD-controlled recovery for 3 seconds...")
            recovery_steps = 240 * 3  # 3 seconds at 240Hz

            for _ in range(recovery_steps):
                joint_states = p.getJointStates(env.robot_id, env.joint_indices, physicsClientId=env.client_id)
                current_positions = np.array([s[0] for s in joint_states])
                current_velocities = np.array([s[1] for s in joint_states])

                # --- Apply Gain Scheduling to the Recovery Controller ---
                # This smooths out the recovery action as the robot nears a neutral pose.
                max_error = np.max(np.abs(current_positions))
                # Use a slightly larger transition range for the stronger recovery gains.
                transition_start = 0.15
                gain_scale = np.clip(max_error / transition_start, 0.0, 1.0)

                # Use stronger base gains for recovery, scaled down as it stabilizes.
                kp = 5.0 * gain_scale
                kd = 0.25 * gain_scale

                torque = -kp * current_positions - kd * current_velocities
                # Apply raw torque, clipped to the environment's physical limit.
                recovery_action = np.clip(torque, -env.torque_limit, env.torque_limit)

                # Apply torques directly using TORQUE_CONTROL
                p.setJointMotorControlArray(
                    bodyUniqueId=env.robot_id,
                    jointIndices=env.joint_indices,
                    controlMode=p.TORQUE_CONTROL,
                    forces=recovery_action.tolist(),
                    physicsClientId=env.client_id,
                )
                p.stepSimulation(physicsClientId=env.client_id)
                time.sleep(1. / 240.)

            print("Recovery attempt complete. Resuming interactive control.")
            # After recovery, get the new observation directly from the environment state.
            # This is crucial as env.step() was not called during recovery.
            state = env._get_obs()
            display.reset(p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env.client_id))
        else:
            state = next_state

    env.close()
    print("\n--- Simulation Closed ---")

if __name__ == "__main__":
    main()