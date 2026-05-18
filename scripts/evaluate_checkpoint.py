import os, sys, torch, numpy as np, tempfile, subprocess

# ─────────────────────────────────────────────────────────────────────────────
#  ✏️  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SIM  = "mujoco"      # "pybullet" or "mujoco"
TASK = "teleop"    # "walker", "turn_left", "turn_right", "point_goal", "teleop"
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_agent(path, env, device, is_discrete):
    from brains.sac.sac_agent_discrete import SACDiscreteAgent
    from brains.sac.sac_agent import soft_actor_critic_agent
    print(f"  > Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    hidden = ckpt.get("hidden_size", 256)
    if is_discrete:
        agent = SACDiscreteAgent(env.observation_space.shape[0], env.action_space, device, hidden, 0, 0, 0, 0)
    else:
        agent = soft_actor_critic_agent(env.observation_space.shape[0], env.action_space, device, hidden, 0, 0, 0, 0, 0)
    sd = ckpt.get("policy", ckpt.get("actor", ckpt))
    agent.policy.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    agent.policy.eval()
    return agent

def get_path_waypoints():
    print("\n" + "="*50 + "\n PATH NAVIGATION MODE\n" + "="*50)
    raw = input("Enter path waypoints (e.g. 1,1; 3,2; 5,0; 0,0): ")
    pts = []
    for pt in raw.split(';'):
        if not pt.strip(): continue
        try:
            x, y = pt.split(',')
            pts.append((float(x), float(y)))
        except:
            print(f"  Skipping invalid: {pt}")
    return pts

def create_pybullet_spheres(waypoints, client_id):
    """Creates non-collidable visual spheres at waypoints in PyBullet."""
    import pybullet as p
    sphere_ids = []
    for i, (x, y) in enumerate(waypoints):
        color = [0.2, 0.8, 0.2, 0.5] if i < len(waypoints) - 1 else [1.0, 0.2, 0.2, 0.6]
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.12, rgbaColor=color,
            physicsClientId=client_id
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis_id,
            baseCollisionShapeIndex=-1,   # No collision
            basePosition=[x, y, 0.12],
            physicsClientId=client_id
        )
        sphere_ids.append(body_id)
    return sphere_ids

def create_mujoco_xml_with_markers(base_xml_path, waypoints):
    """Injects non-touchable sphere bodies into the XML beside the original file."""
    with open(base_xml_path, 'r') as f:
        content = f.read()
    markers = ""
    for i, (x, y) in enumerate(waypoints):
        color = "0.2 0.8 0.2 0.4" if i < len(waypoints) - 1 else "1.0 0.2 0.2 0.5"
        markers += (
            f'<body name="marker_{i}" pos="{x} {y} 0.12" mocap="true">\n'
            f'  <geom type="sphere" size="0.12" rgba="{color}" contype="0" conaffinity="0"/>\n'
            f'</body>\n'
        )
    content = content.replace("</worldbody>", markers + "</worldbody>")
    temp_path = os.path.join(os.path.dirname(base_xml_path), "_eval_markers_temp.xml")
    with open(temp_path, 'w') as f:
        f.write(content)
    return temp_path

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Teleop: just delegate to teleop.py ───────────────────────────────────
    if TASK == "teleop":
        script = "teleop_mujoco.py" if SIM == "mujoco" else "teleop.py"
        print(f"Launching {script}...")
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "scripts", script)])
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== PATH EVALUATION: {SIM.upper()} — {TASK.upper()} ===\n")

    waypoints = []
    if TASK == "point_goal":
        waypoints = get_path_waypoints()

    # ── 1. Environment ────────────────────────────────────────────────────────
    print("\n1. Initializing Environment...")
    temp_xml = None
    viewer   = None

    if SIM == "mujoco":
        base_xml = os.path.join(PROJECT_ROOT, "assets", "mujoco",
                                "Bioloid_Quadruped_Model", "Bioloid_Quadruped_Model.xml")
        xml_to_use = create_mujoco_xml_with_markers(base_xml, waypoints) if waypoints else base_xml
        temp_xml   = xml_to_use if waypoints else None

        if TASK == "walker":
            from envs.mujoco.quadruped_mujoco_env import BioloidMujocoEnv as Env
            env = Env(xml_path=xml_to_use, render_mode="human")
        elif TASK == "turn_left":
            from envs.mujoco.turn_left_mujoco_env import BioloidMujocoTurnLeftEnv as Env
            env = Env(xml_path=xml_to_use, render_mode="human")
        elif TASK == "turn_right":
            from envs.mujoco.turn_right_mujoco_env import BioloidMujocoTurnRightEnv as Env
            env = Env(xml_path=xml_to_use, render_mode="human")
        else:  # point_goal
            from envs.mujoco.point_goal_env_mujoco import BioloidMujocoEnvPointGoal as Env
            env = Env(
                xml_path        = xml_to_use,
                render_mode     = "human",
                walker_path     = "models/mujoco/sac/Straight_Walker.pth",
                turn_left_path  = "models/mujoco/sac/Left_Turner.pth",
                turn_right_path = "models/mujoco/sac/Right_Turner.pth",
            )

        m = env.base_env.model if hasattr(env, 'base_env') else env.model
        d = env.base_env.data  if hasattr(env, 'base_env') else env.data
        import mujoco.viewer as mjv
        print("  > Launching MuJoCo Viewer...")
        viewer = mjv.launch_passive(m, d)

    else:  # pybullet
        if TASK == "walker":
            from envs.pybullet.quadruped_env import BioloidAntLikeEnv as Env
        elif TASK == "turn_left":
            from envs.pybullet.turn_left_env import BioloidAntLikeEnvTurnLeftOnly as Env
        elif TASK == "turn_right":
            from envs.pybullet.turn_right_env import BioloidAntLikeEnvTurnOnly as Env
        else:  # point_goal
            from envs.pybullet.point_goal_env import BioloidEnvPointGoal as Env

        env = Env(render_mode="GUI")

        # Draw waypoint spheres in PyBullet
        if waypoints:
            client_id = env.client_id if hasattr(env, 'client_id') else 0
            create_pybullet_spheres(waypoints, client_id)

    # ── 2. Agent ──────────────────────────────────────────────────────────────
    print("2. Setting up Agent...")
    names = {
        "walker": "Straight_Walker.pth", "turn_left": "Left_Turner.pth",
        "turn_right": "Right_Turner.pth", "point_goal": "Point_goal.pth"
    }
    agent = load_agent(
        os.path.join("models", SIM, "sac", names[TASK]),
        env, device, TASK == "point_goal"
    )

    # ── 3. Simulation Loop ────────────────────────────────────────────────────
    try:
        print("\n--- Starting Evaluation ---")
        obs, _ = env.reset(set_new_goal=False) if TASK == "point_goal" else env.reset()
        if viewer: viewer.sync()

        if TASK == "point_goal":
            for i, (wx, wy) in enumerate(waypoints):
                print(f"  >> Navigating to Waypoint {i+1}: ({wx}, {wy})")
                env.goal_position = np.array([wx, wy])
                reached, steps = False, 0
                while not reached and steps < 3000:
                    obs, _, term, trunc, info = env.step(agent.select_action(obs, True))
                    if viewer: viewer.sync()
                    if info.get("goal_reached"):
                        reached = True
                        print(f"  >> Waypoint {i+1} reached!")
                    if term or trunc: break
                    steps += 1
        else:
            term = trunc = False
            while not (term or trunc):
                obs, _, term, trunc, _ = env.step(agent.select_action(obs, True))
                if viewer: viewer.sync()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if viewer: viewer.close()
        env.close()
        if temp_xml and os.path.exists(temp_xml):
            os.remove(temp_xml)
        print("Done.")

if __name__ == "__main__": main()
