# 🚁 Autonomous UAV Landing Project
### Scale-Adaptive Dual-Expert Perception → RL-Based Expert Gating with PER

This repository documents my MS research project on vision-based autonomous UAV landing at the **LCASL Lab, Tennessee Tech University**. The work spans two phases: a published perception framework and an ongoing reinforcement learning extension.

> **Note:** This work is built on top of a private simulation stack collaboratively developed between the **LCASL Lab at TTU** and **UIUC**. The full stack remains private. This repository contains only my personal research contributions on top of that infrastructure.

---

## 🗺️ Project Arc

```
Phase 1 (Complete) :
  Scale-Adaptive Dual-Expert Perception Framework
  Published: AIAA 2026

Phase 2 (Ongoing) :
  RL-Based Expert Gating with Prioritized Experience Replay
  Replacing handcrafted geometric switching with a learned policy
```

---

## Phase 1 - Published Work

### "Expert Switching for Robust AAV Landing: A Dual-Detector Framework in Simulation"
*Tasnim et al. - AIAA 2026*
📄 Read the Paper here :
[Official Publication](https://doi.org/10.2514/6.2026-2171)
[arXiv Preprint](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=LB21Br0AAAAJ&citation_for_view=LB21Br0AAAAJ:YsMSGLbcyi4C)

#### Problem
During autonomous landing, a helipad appears tiny at high altitude and large near touchdown. A single detector cannot handle both extremes reliably - it either misses detections at far range or loses precision at close range.

#### Phase 1
A **scale-adaptive dual-expert perception framework**:
- Designed and implemented the dual-expert training pipeline on the HelipadCat dataset
- Implemented the geometric gating and temporal smoothing mechanism
- Built the visual servoing landing controller
- Integrated the full perception-control pipeline with the CARLA-GUAM simulation environment
- Ran and analyzed all 10 randomized landing trials

#### Simulation Environment
![Pygame Window](results/figures/pygame_window.png)
*Real-time inference during descent. Top row: Fusion output, YOLO Large expert, YOLO Small expert. Bottom row: CARLA camera views and SORT tracker outputs.*

#### Results
![Landing Scatter](results/figures/Landing_Scatter.png)
*Final landing positions across 10 randomized trials. The Dual Expert Model achieves the tightest clustering around the helipad center (red H) with mean error of 2.53m and std of 1.03m, outperforming both single-expert baselines.*

| Model | Mean Error (m) | Std Dev (m) |
|---|---|---|
| **Dual-Expert (ours)** | **2.53** | **1.03** |
| Near-range Expert | 5.53 | 5.57 |
| Far-range Expert | 5.60 | 3.82 |

The Dual-Expert model achieved **100% mission success rate** across all trials including the challenging 110m altitude case where the near-range expert fails completely.

---

## Phase 2 - Ongoing: RL-Based Expert Gating with PER

### Motivation
The geometric gating rule from Phase 1 is handcrafted and deterministic. The paper's own future work section identifies learning-based gating as the natural next step:

> *"A key direction for future research is the development of a reinforcement-learning-based gating mechanism capable of learning expert selection policies directly from landing performance rather than relying on handcrafted geometric rules."*

### Goal
Replace the L1-norm gating rule:
```python
# Current: handcrafted geometric rule
D_k(t) = |u_k(t) - c_x| + |v_k(t) - c_y|
k* = argmin_k D_k(t)
```

With a **learned RL policy** trained using Prioritized Experience Replay — prioritizing transitions where expert selection was uncertain or caused large landing error.

#### Automated Multi-Episode Pipeline (`episode_manager.py`)
Built a ROS node that runs fully automated landing trials with randomized starting positions:
```
x ~ Uniform(-95, -65) m
y ~ Uniform( 60,  90) m
z ∈ {70, 80, 90, 110} m
```
Before this, each trial required manual restart. Now 10 episodes run end-to-end automatically — essential for collecting diverse experience data for PER training.

#### GUAM Hot-Reset (`node_vehicle.py`)
Added mid-run reinitialization of the JAX-based physics integrator without restarting the simulation. GUAM runs a continuous integration loop — without resetting its internal `b_state`, it overrides every CARLA teleport with the old landing position within one tick.

#### Pose-Aware Environment Reset (`environment.py`)
Extended `reset()` to accept a target pose and teleport the CARLA actor to the correct altitude above the helipad, instead of a random road-level spawn point.

#### Landing Controller Tuning (`landing_controller.py`)
Tuned the visual servoing controller for stable landings from random start positions:
- **Adaptive P gain** — scales down for large positional errors, preventing overshoot on approach
- **Altitude-scaled descent** — fast at high altitude, automatically slows near ground based on bounding box size
- **Fixed 20x velocity attenuation bug** — see below

#### Key Bug Fixed: 20x Velocity Attenuation
After modifying the controller during Phase 1, the velocity callback in `node_vehicle.py` was using GUAM's internal simulation timestep (`dt=0.005s`) for position updates, but the callback fires at 10Hz (`dt=0.1s`). Result: commanding `-0.8 m/s` descent produced only `-0.04 m/s` actual motion. Episodes were taking 7-8 minutes each.

```python
# Before — wrong dt, 20x attenuation
dt = getattr(self.guam, "dt", 0.005)

# After — correct controller rate
dt = 0.1
```

**Impact:** Episode time reduced from ~8 minutes to ~60 seconds, making RL training feasible.

---

## 🏗️ Stack Overview

The system runs across four Docker containers communicating over ROS Noetic:

```
CARLA Simulator  ←→  env_sim (environment + controller)
                           ↕
                      jaxguam (NASA GUAM vehicle dynamics)
                           ↕
                      yolov8 (dual-expert perception)
```

## ⚙️ How the System Runs
When the simulation launches, four nodes start in sequence. First, node_carla.py initializes the CARLA environment - it spawns the ego vehicle at the configured starting location, sets up all onboard cameras and sensors, and begins ticking the world at 10Hz. It also listens for reset signals so it can teleport the vehicle between episodes. Second, node_vehicle.py starts the NASA GUAM physics integrator - it subscribes to velocity commands from the controller and continuously integrates them forward to produce realistic 6DOF flight dynamics, publishing the UAV's pose back to CARLA so the visual actor matches the physics state. Third, the YOLO perception nodes launch - two YOLOv8 experts run in parallel on the downward-facing camera feed, one specialized for far-range small helipads and one for near-range large helipads. A geometric gating module selects the best detection each frame and publishes a fused bounding box. Fourth, landing_controller.py subscribes to the fused detection and runs the visual servoing loop - it computes lateral error from the image center, aligns the UAV in XY, then locks position and begins descent scaled by the apparent helipad size, publishing velocity commands back to GUAM. Finally, episode_manager.py sits above all of this as the orchestrator - it samples a random starting pose, triggers a coordinated reset across CARLA and GUAM simultaneously, waits for the controller to signal landing complete, then starts the next episode automatically. This cycle repeats for however many episodes are configured, collecting experience data for PER training.

**Key ROS Topics:**

| Topic | Description |
|---|---|
| `/yolo_node/fused_array` | Dual-expert fused detections |
| `/controller_node/vel_cmd` | Velocity commands to UAV |
| `/jaxguam/pose` | UAV state feedback |
| `/episode/initial_pose` | Randomized starting position |
| `/episode/done` | Landing complete signal |
| `/episode/reset` | Reset controller state |

---

## 🔮 Next Steps

- [ ] Transition logging `(state, action, reward, next_state)` per step
- [ ] PER buffer implementation with priority sampling
- [ ] RL policy training to replace geometric gating
- [ ] Uncertainty estimation (MC Dropout) for YOLO experts
- [ ] Evaluation: learned gating vs geometric baseline

---


## Prereqs

- Ubuntu 20.04.6 LTS or 22.04.4 LTS (Other versions untested, but should work.)
- CUDA GPU for Pytorch and Unreal Engine, e.g., NVIDIA GeForce RTX series.
- Install Docker and Nvidia Docker Toolkit, see [doc/tools_installation.md](doc/tools_installation.md) for detailed instructions.
- Python packages
```bash
python3 -m pip install loguru
```

## Quick Start
Clone this repository with submodules.

```bash
git clone https://github.com/CPS-IL/rraaa-sim.git --recurse-submodules
cd rraaa-sim
git switch htasnim
git submodule update --init --recursive
#docker sudo access:
sudo chmod 666 /var/run/docker.sock
#docker access to host display
xhost +local:docker


python3 rraaa.py configs/single-static.yml
```

## 📚 References

- Tasnim et al. — *"Expert Switching for Robust AAV Landing"* — AIAA 2026
- Bitoun & Winkler — *"HelipadCat Dataset"* — IEEE TENCON 2020
- Dosovitskiy et al. — *"CARLA: An Open Urban Driving Simulator"* — CoRL 2017
- NASA — *"Generic Urban Air Mobility (GUAM)"*

---

## 👩‍💻 Author

**Humaira Tasnim**   
Graduate Student  
Mechanical and Nuclear Engineering  
Tennessee Tech University  


## Contact
  - [Humaira Tasnim](mailto:humairatasnim601@gmail.com)  
  - [Hyung-Jin Yoon](mailto:stargaze221@gmail.com)
  - [Ayoosh Bansal](mailto:ayooshb2@illinois.edu)
  - [Mikael Yeghiazaryan](mailto:myeghiaz@illinois.edu)
  - [Oswin So](mailto:oswinso@mit.edu) : JAX GUAM

