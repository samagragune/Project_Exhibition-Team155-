# Robotic Arm Cognitive Autonomy Layer & 3D Simulation v3.0

An AI-driven **Cognitive Autonomy Layer** for lower-cost robotic arms that replaces hardcoded state machines and brittle Vision-Language-Action (VLA) models with hierarchical multimodal reasoning, secondary 3D depth perception, calibrated inverse kinematics, and multi-scenario environments.

---

## 🚀 Key Autonomy Features & Roadmap Implementation

### 1. Camera Frustum Optimization & Compact Arm Scaling (Roadmap #2)
- **Elevated Frustum**: The top-down camera is elevated to `1.05m` with a wide `65°` FOV, capturing the full 50cm × 50cm workspace without link occlusion.
- **Proportional Scaling**: The arm is scaled compactly to `0.58` with base offset at `[-0.20, 0.25, -0.04]`, ensuring the base and shoulder never intrude into the top-down vision plane.
- **Calibrated Kinematic Offset**: Dynamically scales the end-effector tool offset (`effective_offset = 0.12 * (scale / 0.75)`) for millimeter-accurate fingertip contact at the table surface ($Z=0$).
- **Autonomous Survey Park Action (`park_arm`)**: The agent can autonomously call `park_arm(pose="survey")` to tuck the arm completely out of the camera's line of sight whenever it needs an unobstructed view.

### 2. Secondary Depth Sensor & 3D Spatial Fusion (Roadmap #3)
- **Metric Depth Buffer**: Extracts true line-of-sight metric depth from PyBullet's camera buffer, calculating physical elevation above table surface in centimeters.
- **2D-to-3D Object Mapping**: Maps SAM 2.1 segmented 2D contours to depth sensor point clusters, calculating:
  - Exact 3D Centroid: `(X, Y, Z)` in centimeters.
  - Top Surface Elevation: `surface_top_z_cm`.
  - Physical Object Height: `height_cm`.
  - Recommended Grasp Descent Height: `grasp_recommended_z_cm`.
- **Live Depth Sensor Heatmap**: Real-time colored visualization of elevation (`/capture_depth`) using a high-contrast turbo colormap.

### 3. Frontier Vision-Language Model Integration (Roadmap #4)
- Seamless multi-provider architecture supporting:
  - **Qwen2.5-VL**: Open-source frontier for 2D/3D visual grounding and normalized coordinate prediction.
  - **DeepSeek-VL2**: Dynamic high-resolution visual reasoning and MoE cognitive deduction.
  - **Google Gemini 2.0 Flash / Pro**: State-of-the-art multimodal reasoning with native tool calling.
  - **LM Studio / Ollama**: Local private workstation inference.
- **Spatial Chain-of-Thought Prompting**: Two-stage approach instructions (approach clearance height $Z_{\text{top}} + 12\text{cm} \to$ descend to grasp height $\to$ close claw $\to$ lift) and atomic action execution.

### 4. Diverse Cognitive Scenarios vs VLAs (Roadmap #5)
Demonstrates the strengths of decoupled cognitive autonomy over brittle end-to-end VLAs across 4 distinct working scenarios:
- 📦 **Standard Workspace Sorting (`sorting`)**: Red soda can, blue cubes, obstacle cylinders, and yellow calibration markers.
- 🔴 **Mars Rover Sample Curation (`mars_rover`)**: Martian red regolith terrain, target hematite core sample, olivine crystal mineral, basalt discard rock, and hermetic rover carousel sample containers.
- 🧪 **Chemical & Bio-Synthesis Lab (`chemistry_lab`)**: Sterile cleanroom bench, concentrated acid vial, base neutralizer, titration reaction beaker, and hazardous waste disposal basin.
- ⚡ **High-Precision Electronics Assembly (`manufacturing_plant`)**: Anti-static ESD mat, microcontroller QFN chip, valid electrolytic capacitor, defective bulged capacitor, PCB socket, and QA reject chute.
- **Dynamic Scenario Switcher**: Live switching via `POST /switch_scenario` or web interface.

### 5. Cyber-Aerospace Web Cockpit & Telemetry (Roadmap #6)
- Modern aerospace dark glassmorphism dashboard served directly via FastAPI:
  - **Dual-Feed Viewport**: Switch between Overhead RGB Camera, 3D Depth Heatmap, or Dual Monitor layout.
  - **3D Perception Inspector**: Real-time table displaying detected objects with 3D centroids, dimensions, top elevations, and one-click grasp targeting.
  - **Autonomy Chat & Reasoning Console**: Natural language command prompt with streaming thought/reasoning logs, tool chips, and quick action buttons.
  - **Scenario Switching Hub**: Switch environments with 1 click.
  - **Manual Jog & Diagnostic Controls**: Sliders and toggle controls for $X, Y, Z$ and claw state.

---

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
cd robo-arm-master

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Cockpit & Simulator

```bash
# Start the unified simulator and web dashboard
python simulator.py
```
Open your browser and navigate to: **`http://127.0.0.1:8080`**

### Running the Rich TUI Orchestrator (CLI Mode)

```bash
# In a second terminal, launch the interactive CLI loop
python orchestrator.py
```

Useful slash commands in the TUI:
- `/scenario mars_rover` — switch to Mars Rover scenario
- `/scenario chemistry_lab` — switch to Chemistry Lab scenario
- `/scenario manufacturing_plant` — switch to Electronics Plant scenario
- `/scenario sorting` — switch to standard sorting scenario
- `/model qwen2.5-vl` — switch to Qwen2.5-VL model preset
- `/model deepseek-vl2` — switch to DeepSeek-VL2 model preset
- `/model gemini-2.0` — switch to Gemini 2.0 preset

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Cockpit Dashboard HTML |
| `GET` | `/capture` | Overhead Camera RGB JPEG |
| `GET` | `/capture_depth` | Secondary Depth Sensor Heatmap JPEG |
| `GET` | `/telemetry` | Real-time $(X, Y, Z)$ coordinates, claw state, and active scenario |
| `GET` | `/scenarios` | List supported cognitive environments |
| `POST` | `/switch_scenario` | Switch active scenario (`{"scenario": "mars_rover"}`) |
| `POST` | `/execute_tool` | Dispatches `move_arm`, `set_claw`, or `park_arm` |
| `POST` | `/api/manual_move` | Direct coordinate jog control from UI |
| `POST` | `/api/scan` | Runs SAM 2.1 segmentation + 3D depth fusion |
| `POST` | `/api/chat` | Natural language autonomy step (`{"command": "..."}`) |

---

## 🧪 Automated Verification Suite

Run the full automated test suite verifying kinematics, depth fusion, model configurations, and all 4 scenarios:

```bash
python test_autonomy_suite.py
python test_web_server.py
```
