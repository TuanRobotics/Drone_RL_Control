# Learning Agile Quadrotor Flight: Narrow-Gate Passing with Extension to Racing Tracks
# Project Overview (English)

## Participants
- Dinh Ngoc Tuan, 291184
- Nguyen Thai Son,

### Motivation

Flying a quadrotor through narrow gates is a compact but challenging robotics benchmark: small tracking or approach errors can cause collisions, making the task highly sensitive to control accuracy and decision-making. It also naturally scales to drone racing, where the drone must pass multiple gates at high speed. These settings are closely related to real-world constrained-flight scenarios (e.g., confined-space inspection)

### Goals

+ Learn robust policies for narrow gate passing in simulation.
+ Extend to racing (multi-gate tracks).
+ Run an ablation study on:
+ Reward shaping
+ Curriculum learning
+ Reward shaping + curriculum
### Method

+ Simulator: gym-pybullet-drones (PyBullet-based Gym/Gymnasium environments for quadrotor RL and benchmarking). 

+ RL algorithms (project-dependent): e.g., PPO / SAC / TD3.

+ Reward shaping provides denser learning signals; we follow the classic policy-invariance perspective on shaping rewards.

Curriculum learning gradually increases task difficulty (spawn distance, gate size/yaw, speed), which is especially helpful under sparse rewards.

### Key Contributions
+ An RL training pipeline for narrow-gate navigation, with extension to racing.
+ Controlled comparison of Sparse, Shaping, Curriculum, and Shaping+Curriculum.
+ Evaluation with success rate, learning curves, time-to-threshold success, and stability/generalization.

### References
+ Falanga et al., Aggressive Quadrotor Flight through Narrow Gaps… (ICRA 2017).
+ Ng et al., Policy Invariance under Reward Transformations… (ICML 1999).
+ Florensa et al., Reverse Curriculum Generation for Reinforcement Learning (2017).
+ Panerati et al., Learning to Fly — a Gym Environment with PyBullet Physics… (2021).
+ Song et al., Autonomous Drone Racing with Deep Reinforcement Learning (IROS 2021).
+ Kaufmann et al., Champion-level drone racing using deep reinforcement learning (Nature 2023).

## Research Description (Updated 2025-12-08)
- Goal: learn end-to-end policies for stable flight through a narrow gate and extend to racing layouts. Compare sparse vs. shaped rewards and curriculum scheduling.
- Environments: `gym_pybullet_drones/envs/FlyThruGateAvitary.py` (single gate, 0.6 m × 0.4 m opening after scaling), `gym_pybullet_drones/envs/DroneRacingAviary.py` (multi-gate track). Gate mesh in `gym_pybullet_drones/assets/gate.obj` with scale set in `gate.urdf`.
- Agents: implementations in `agents/ppo_agent.py`, SAC/TD3 agents under `agents/`. Training scripts in `src/train_thrugate_ppo.py`, `src/train_thrugate_sac.py`, `src/train_thrugate_td3.py`, and racing script `src/train_racing_ppo.py`.
- Reward shaping vs curriculum: four experiment settings planned (below). Logs and checkpoints are written under `log_dir/`.

## Experimental Plan (minimum viable)
1) Sparse only (large reward only when gate is passed).  
2) Shaping only (dense progress signals, no curriculum).  
3) Curriculum only (progressive task difficulty, sparse/light shaping).  
4) Shaping + Curriculum (full pipeline).  
Report: learning curves (success rate), steps to reach X% success, stability.

## Demonstration (Video)
- Mandatory: add a video link showing successful flight (YouTube/Vimeo/cloud file).  
  Example placeholder: `[Add video link here]`.
- Recommended: embed a short GIF recorded from tests (see “Recording runs” below).

## Installation and Deployment
- Tested locally with Python 3.x and PyBullet GUI. For GUI, ensure OpenGL; for headless recording, use EGL/OSMesa.
- System packages (Ubuntu example):  
  ```bash
  sudo apt-get update && sudo apt-get install -y python3-venv ffmpeg libgl1
  ```
- Setup from scratch:  
  ```bash
  git clone <this-repo>
  cd drone_rl_control
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Optional: `export PYTHONPATH=$(pwd)` so scripts can import local packages.

## Running and Usage
- Train (FlyThruGate):
  - PPO: `python src/train_thrugate_ppo.py`
  - SAC: `python src/train_thrugate_sac.py`
  - TD3: `python src/train_thrugate_td3.py`
- Train racing (multi-gate): `python src/train_racing_ppo.py`
- Test a checkpoint (PPO example):  
  ```bash
  python src/test_thrugate_ppo.py --model_path /path/to/ppo_model_final.pt
  ```
- Recording runs: set `DEFAULT_RECORD_VIDEO = True` in `src/test_thrugate_ppo.py` (or pass `record=True` when constructing the env). GUI=True saves MP4 to `results/`; GUI=False saves PNG frames in `results/recording_*`. Convert frames/MP4 to GIF with ffmpeg, e.g. `ffmpeg -i results/video-*.mp4 -vf "fps=20,scale=640:-1:flags=lanczos" output.gif`.

## Results and Artifacts
- Current test summaries:  
  - SAC: `src/log_dir/results_thrugate_sac/test_summary/test_results_sac.txt`  
  - TD3: `src/log_dir/results_thrugate_td3/test_summary/test_results_td3.txt`  
  - PPO: add after running `src/test_thrugate_ppo.py` and saving summary.
- Checkpoints: under `log_dir/<algo>_training_thrugate/...`. Describe naming in your report.  
- Include links to raw logs/plots/checkpoints in cloud storage if needed.

## Dependency Management
- Python dependencies listed in `requirements.txt` (torch, gymnasium, pybullet, gym-pybullet-drones, etc.). Keep this file authoritative; avoid adding system packages here.
- Virtual environments and caches are ignored via `.gitignore` (covers venv, pycache, logs, build artifacts, IDE files). Add any new large data or weights paths to `.gitignore` before committing.

## Repository Pointers
- Envs: `gym_pybullet_drones/envs/` (BaseRLAviary, FlyThruGateAvitary, DroneRacingAviary).  
- Agents: `agents/` (PPO/SAC/TD3 networks and buffers).  
- Training/testing scripts: `src/`.  
- Assets: `gym_pybullet_drones/assets/` (gate mesh/URDF).  
- Results/logs: `log_dir/` (training runs) and `src/log_dir/` (test summaries).
