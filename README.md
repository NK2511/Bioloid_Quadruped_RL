# Quadruped Reinforcement Learning

This repository implements a **Hierarchical Reinforcement Learning (HRL)** framework for a Bioloid quadruped robot using PyBullet and PyTorch. It is designed to demonstrate how low-level motor skills can be composed to achieve complex high-level navigation tasks.

## Media

<img width="828" height="635" alt="Screenshot 2025-09-22 220335" src="https://github.com/user-attachments/assets/f028cb7f-0524-4230-aafe-e4cc95ab71fa" />
https://github.com/user-attachments/assets/28658172-1d26-478a-8d5b-c946233ce1be


## System Overview

The framework consists of two main levels:

1.  **Low-Level Skills (The "Body")**:
    -   Specialized Soft Actor-Critic (SAC) agents trained for specific primitive actions:
        -   **Walking Forward**: Moves the robot straight ahead.
        -   **Turning Left/Right**: Rotates the robot in place.
    -   These agents accept the robot's proprioceptive state (joint angles, velocities, orientation) and output direct joint torques.

2.  **High-Level Planner (The "Brain")**:
    -   A "Point-Goal" navigation agent that decides *which* skill to execute.
    -   It observes the distance and angle to a target waypoint and selects a discrete command (`WALK`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`) every ~0.5 seconds.
    -   This hierarchical approach simplifies long-horizon navigation by abstracting complex motor control into reusable behaviors.

## Project Structure

-   `envs/`: Custom Gymnasium environments for the quadruped.
    -   `quadruped_env.py`: Base physics environment (8-DoF).
    -   `point_goal_env.py`: Hierarchical navigation environment.
    -   `turn_*.py`: Specialized training environments for turning skills.
-   `models/`: Pre-trained model checkpoints.
-   `scripts/`: Training scripts for individual skills and the navigator.
-   `sac/`: Soft Actor-Critic (SAC) implementation.
-   `assets/`: Robot URDF and mesh files.
-   `evaluate_point_goal.py`: Script to evaluate the full navigation stack.
-   `teleop.py`: Interactive keyboard control script.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Evaluation (Navigation)
To observe the fully trained hierarchical agent navigating a sequence of waypoints:
```bash
python evaluate_point_goal.py
```
This script spawns the robot and a series of blue waypoints. The robot will autonomously switch between walking and turning to reach each goal.

### Teleoperation (Manual Control)
You can manually control the robot using your keyboard to test the low-level skills.
```bash
python teleop.py
```
*(Click on the PyBullet GUI window to ensure it captures your key presses)*

**Controls:**
-   **UP Arrow**: Walk Forward
-   **LEFT Arrow**: Turn Left
-   **RIGHT Arrow**: Turn Right
-   **'r' Key**: Reset Robot

> [!IMPORTANT]
> **Operational Note**: Please ensure stable control inputs for best performance.
> *   **Do not press multiple direction keys simultaneously** (e.g., Up + Left). The system cannot blend these distinct policies, and conflicting torque commands may cause instability.
> *   **Avoid rapid, erratic switching** between commands. Sudden changes in policy can destabilize the robot's dynamic balance.

### Training
To train the walker agent from scratch:
```bash
python scripts/train_walker.py --gui
```
(Other training scripts are available in the `scripts/` directory)

## License
MIT


