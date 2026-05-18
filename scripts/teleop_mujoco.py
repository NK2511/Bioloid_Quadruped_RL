"""
MuJoCo Teleoperation Script
============================
Controls:
    UP ARROW    → Walk Forward
    LEFT ARROW  → Turn Left
    RIGHT ARROW → Turn Right
    R           → Reset robot position
    ESC / Q     → Quit
"""

import os, sys, time, torch, numpy as np
from pynput import keyboard as kb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from envs.mujoco.quadruped_mujoco_env     import BioloidMujocoEnv
from envs.mujoco.turn_left_mujoco_env     import BioloidMujocoTurnLeftEnv
from envs.mujoco.turn_right_mujoco_env    import BioloidMujocoTurnRightEnv
from brains.sac.sac_agent                 import soft_actor_critic_agent

# ── Paths ─────────────────────────────────────────────────────────────────────
WALKER_PATH     = os.path.join(PROJECT_ROOT, "models", "mujoco", "sac", "Straight_Walker.pth")
TURN_LEFT_PATH  = os.path.join(PROJECT_ROOT, "models", "mujoco", "sac", "Left_Turner.pth")
TURN_RIGHT_PATH = os.path.join(PROJECT_ROOT, "models", "mujoco", "sac", "Right_Turner.pth")

# ── Globals for key state ─────────────────────────────────────────────────────
_keys_held = set()
_quit      = False
_reset     = False

def _on_press(key):
    global _quit, _reset
    _keys_held.add(key)
    if key == kb.Key.esc or key == kb.KeyCode.from_char('q'):
        _quit = True
    if key == kb.KeyCode.from_char('r') or key == kb.KeyCode.from_char('R'):
        _reset = True

def _on_release(key):
    _keys_held.discard(key)

def get_command():
    """Returns 'walk', 'turn_left', 'turn_right', or 'idle'."""
    if kb.Key.up    in _keys_held: return "walk"
    if kb.Key.left  in _keys_held: return "turn_left"
    if kb.Key.right in _keys_held: return "turn_right"
    return "idle"

def load_agent(path, env, device):
    ckpt = torch.load(path, map_location=device)
    hidden = ckpt.get("hidden_size", 256)
    agent = soft_actor_critic_agent(
        env.observation_space.shape[0], env.action_space, device,
        hidden, 0, 0, 0, 0, 0
    )
    sd = ckpt.get("actor", ckpt)
    agent.policy.load_state_dict(sd, strict=False)
    agent.policy.eval()
    return agent

def main():
    global _quit, _reset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Environments ──────────────────────────────────────────────────────────
    print("Loading environments...")
    walker_env     = BioloidMujocoEnv()
    turn_left_env  = BioloidMujocoTurnLeftEnv()
    turn_right_env = BioloidMujocoTurnRightEnv()

    # ── Agents ────────────────────────────────────────────────────────────────
    print("Loading agents...")
    walker_agent     = load_agent(WALKER_PATH,     walker_env,     device)
    turn_left_agent  = load_agent(TURN_LEFT_PATH,  turn_left_env,  device)
    turn_right_agent = load_agent(TURN_RIGHT_PATH, turn_right_env, device)

    # ── Share the same underlying model/data for rendering ────────────────────
    model = walker_env.model
    data  = walker_env.data

    # Sync turn envs to use the same data
    turn_left_env.model  = model
    turn_left_env.data   = data
    turn_right_env.model = model
    turn_right_env.data  = data

    import mujoco.viewer as mjv
    print("Launching MuJoCo Viewer...")
    viewer = mjv.launch_passive(model, data)

    # ── Keyboard listener ─────────────────────────────────────────────────────
    listener = kb.Listener(on_press=_on_press, on_release=_on_release)
    listener.start()

    print("\n--- MuJoCo Teleop Ready ---")
    print("  UP ARROW    → Walk")
    print("  LEFT ARROW  → Turn Left")
    print("  RIGHT ARROW → Turn Right")
    print("  R           → Reset")
    print("  ESC / Q     → Quit\n")

    obs, _ = walker_env.reset()

    try:
        while not _quit and viewer.is_running():
            if _reset:
                obs, _ = walker_env.reset()
                _reset = False
                print("  > Reset.")

            cmd = get_command()

            if cmd == "walk":
                action = walker_agent.select_action(walker_env._get_obs(), eval=True)
            elif cmd == "turn_left":
                action = turn_left_agent.select_action(walker_env._get_obs(), eval=True)
            elif cmd == "turn_right":
                action = turn_right_agent.select_action(walker_env._get_obs(), eval=True)
            else:
                # Idle: apply zero torque
                action = np.zeros(walker_env.action_space.shape[0])

            obs, _, terminated, truncated, _ = walker_env.step(action)
            viewer.sync()

            if terminated or truncated:
                obs, _ = walker_env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        viewer.close()
        walker_env.close()
        print("Done.")

if __name__ == "__main__": main()
