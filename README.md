# Robotic Arm Vision & Actuation System — 3D Simulation

A fully simulated environment for developing, testing, and iterating on an AI agent pipeline that controls a robotic arm. Replaces the physical ESP32-CAM + hardware arm with a PyBullet physics simulation that produces photorealistic camera feeds and executes every tool call through inverse kinematics.

## Pipeline Overview

```
User Command
    │
    ▼
┌─────────────────┐
│  orchestrator.py │  Rich TUI chat loop, coordinates the pipeline
└────────┬────────┘
         │
         ▼
┌─────────────────┐     JPEG frame
│  get_image.py   │ ───► from simulator HTTP endpoint (127.0.0.1:8080)
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│ sam2_processor.py│  SAM 2.1 segmentation + noise filtering + centroid calc
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     tool call (move_arm / set_claw)
│ agent_harness.py │  LLM reasoning via OpenAI-compatible API
└────────┬─────────┘
         │  HTTP POST
         ▼
┌──────────────────┐
│   simulator.py   │  PyBullet physics + IK solver + FastAPI server
└──────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (SAM 2.1 inference)
- X11 / display server required for PyBullet GUI

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd 3D-simulation-testing

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Start the simulator backend (must be running first)
python simulator.py

# Start the orchestrator TUI
python orchestrator.py              # online mode (hits simulator on localhost:8080)
python orchestrator.py --offline    # uses test_artifacts/*.jpg instead

# Standalone modules
python agent_harness.py             # standalone agent test
python sam2_processor.py            # standalone SAM 2.1 processing
python get_image.py                 # standalone camera fetch
```

## Architecture

### Core Modules

| Module | Purpose |
|---|---|
| `orchestrator.py` | Main entry point. Rich TUI chat loop that coordinates the full pipeline: fetch image → segment → reason → execute |
| `simulator.py` | PyBullet physics world + FastAPI HTTP server. Receives tool calls, solves IK, renders camera frames |
| `agent_harness.py` | LLM client (OpenAI-compatible). Manages conversation history, validates tool calls against workspace bounds, streams reasoning |
| `sam2_processor.py` | SAM 2.1 segmentation module. Detects objects, computes centroids in cm, applies noise filtering and color-based calibration |
| `get_image.py` | HTTP client that fetches JPEG frames from the simulator (or ESP32-CAM in hardware mode) |
| `config.py` | All configuration: camera settings, SAM 2.1 params, agent/LLM settings, workspace bounds, simulator settings |

### Simulator (`simulator.py`)

The simulator is a drop-in replacement for the physical hardware backend:

- **Physics Engine**: PyBullet with UR5 robot arm + Robotiq 140 gripper
- **IK Solver**: PyBullet's `calculateInverseKinematics` with joint limits and orientation constraints
- **Camera**: Top-down view (60° FOV, 640x480), renders to JPEG via `getCameraImage`
- **HTTP API**: FastAPI server with `/capture` (JPEG frame) and `/execute_tool` (POST tool calls)
- **Grasp Physics**: Fixed constraint attaches objects when gripper closes within proximity threshold
- **ESP32 Noise Simulation**: Optional scanline artifacts to match real camera characteristics

### Agent Pipeline

1. **Capture**: Fetch latest frame from simulator
2. **Perception**: SAM 2.1 detects objects, computes centroids calibrated against yellow corner markers
3. **Reasoning**: LLM receives current frame + previous frame + SAM metadata + user command
4. **Tool Call**: Agent outputs a single tool call (`move_arm` or `set_claw`)
5. **Execution**: Orchestrator dispatches tool call to simulator via HTTP POST
6. **Loop**: Auto-triggers next iteration (max 10 turns) until agent finishes or requests user input

## Configuration

All settings are in `config.py`:

### `simulator_CONFIG`
| Key | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Simulator server address |
| `PORT` | `8080` | HTTP server port |
| `CAMERA_RESOLUTION` | `(640, 480)` | Render resolution |
| `CAMERA_FOV` | `60.0` | Field of view in degrees |
| `CAMERA_EYE_POSITION` | `[0.25, 0.25, 0.85]` | Camera position in world space (m) |
| `CAMERA_TARGET_POSITION` | `[0.25, 0.25, 0.0]` | Camera look-at target (m) |
| `ESP32_NOISE` | `False` | Enable scanline noise simulation |
| `GUI_ENABLED` | `True` | Show PyBullet 3D window |

### `agent_harness_CONFIG`
| Key | Default | Description |
|---|---|---|
| `LM_STUDIO_URL` | `http://26.50.165.63:1234/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `gemma-4-12b-it@q4_k_m` | Model to use |
| `TEMPERATURE` | `0.55` | Sampling temperature |
| `BOUNDS` | X:[0,50] Y:[0,50] Z:[0,30] | Workspace limits in cm |

### `sam2_processor_CONFIG`
| Key | Default | Description |
|---|---|---|
| `MODEL_PATH` | `sam2.1_l.pt` | SAM 2.1 model weights |
| `PIXELS_PER_CM` | `30.0` | Default scale factor |
| `FILTER_SCANLINES` | `True` | Apply median blur for ESP32 noise |
| `CONFIDENCE_THRESHOLD` | `0.85` | Minimum detection confidence |

### `get_image_CONFIG`
| Key | Default | Description |
|---|---|---|
| `ESP32_IP` | `127.0.0.1:8080` | Simulator endpoint (overrides ESP32 IP) |
| `ENDPOINT` | `/capture` | HTTP endpoint for JPEG |
| `TIMEOUT_SECONDS` | `5` | HTTP request timeout |
| `MAX_RETRIES` | `3` | Retry attempts on failure |

## Workspace

- **Size**: 50cm x 50cm (X: 0–50cm, Y: 0–50cm), Z height ceiling: 30cm
- **Camera**: Fixed top-down mount at 85cm above center
- **Arm**: UR5 with Robotiq 140 gripper, mounted off-camera to the left
- **Calibration**: 4 yellow corner markers at (5,5), (45,5), (5,45), (45,45) cm for dynamic pixel-to-cm scaling

## File Structure

```
3D-simulation-testing/
├── orchestrator.py          # Main entry, Rich TUI chat loop
├── config.py                # All configuration (camera, SAM2, agent, simulator)
├── agent_harness.py         # LLM client, tool validation, conversation history
├── sam2_processor.py        # SAM 2.1 inference, centroid calculation, noise filtering
├── get_image.py             # HTTP fetch from simulator/ESP32-CAM
├── simulator.py             # PyBullet physics + FastAPI HTTP server
├── sam2.1_l.pt              # SAM 2.1 large model weights (download separately)
├── requirements.txt         # Python dependencies
├── AGENTS.md                # Developer notes and architecture reference
├── handshake/               # Runtime: raw frames and annotated outputs
├── test_artifacts/          # Offline test images
└── models/
    └── ur5_pybullet/        # UR5 + Robotiq 140 3D model (auto-cloned on first run)
```

## Tool Calls

The agent can issue two types of tool calls:

### `move_arm(x_cm, y_cm, z_cm)`
Moves the arm end-effector to specified 3D coordinates in centimeters.
- X: [0.0, 50.0] cm — left to right
- Y: [0.0, 50.0] cm — front to back
- Z: [0.0, 30.0] cm — height (0 = table surface)

### `set_claw(state)`
Controls the gripper end-effector.
- `state`: `"open"` or `"closed"`

## Hardware Mode (ESP32-CAM)

This simulation can be switched to hardware mode by updating `config.py`:
- Set `simulator_CONFIG` port to ESP32-CAM IP
- Update `get_image_CONFIG["ESP32_IP"]` to the camera's network address
- The orchestrator and agent code remain **unchanged** — the simulator is a drop-in replacement

## License

Proprietary / Internal use.
