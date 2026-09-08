# config.py
# Unified configuration for Robotic Arm Cognitive Autonomy Layer
# Covers: Perception (SAM 2.1 + 3D Depth Fusion), Agent Harness (Frontier Models: DeepSeek/Qwen/Gemini),
# Simulator Backend (PyBullet + Multi-Scenario Worlds), and Hardware Interfaces.

import os

# =========================================================================================================================
# 🔍 CONFIG FOR `sam2_processor.py` (PERCEPTION & SENSOR FUSION)
# =========================================================================================================================
sam2_processor_CONFIG = {
    # Model & File Paths
    "MODEL_PATH": "sam2.1_l.pt",
    "INPUT_IMAGE": "handshake/raw_capture.jpg",
    "OUTPUT_IMAGE": "handshake/annotated_output.jpg",
    "DEPTH_HEATMAP_IMAGE": "handshake/depth_heatmap.jpg",
    
    # 2D/3D Calibration & Scale
    "PIXELS_PER_CM": 25.0,                  # Calibrated scale factor at 1.05m elevation
    "FILTER_SCANLINES": True,               # ESP32/camera sensor noise mitigation
    
    # SAM 2.1 Detection Tuning
    "CONFIDENCE_THRESHOLD": 0.80,
    "FILTER_NESTED": False,
    "NESTED_OVERLAP_RATIO": 0.85,
    "MIN_MASK_AREA_PX": 30,
    "MIN_MASK_CIRCULARITY": 0.02,
    
    # Secondary Depth Sensor Fusion
    "DEPTH_FUSION_ENABLED": True,
    "DEPTH_FILTER_PERCENTILE": 75.0,        # Percentile sampling to extract object top surface
    "DEFAULT_OBJECT_Z_MARGIN_CM": 1.5       # Clearance margin above surface
}

# =========================================================================================================================
# 🧠 CONFIG FOR `agent_harness.py` (COGNITIVE REASONING AGENT)
# =========================================================================================================================
sysPrompt = """You are an advanced AI Spatial Reasoning & Autonomy Agent controlling an embodied robotic arm over a 50cm x 50cm workspace.

Unlike brittle Vision-Language-Action (VLA) models that fail under distribution shifts, you use high-level spatial perception, multi-sensor depth fusion, and step-by-step cognitive deduction.

### 📐 WORKSPACE BOUNDARIES & COORDINATE SYSTEM:
- X: [0.0 cm to 50.0 cm] — Left to Right. (Center = 25.0 cm)
- Y: [0.0 cm to 50.0 cm] — Front to Back. (Center = 25.0 cm)
- Z: [0.0 cm to 28.0 cm] — Vertical Height above table surface.
  * Z = 0.0 cm: Table surface level.
  * Z = 25.0 - 28.0 cm: Safe transit height ceiling.

### 👁️ MULTI-SENSOR PERCEPTION & 3D DEPTH GROUNDING:
You receive:
1. Annotated top-down camera visual frame with detected objects and bounding boxes.
2. 3D spatial metadata for each object, derived from our secondary depth sensor:
   - `centroid_cm`: {"x", "y", "z"} -> Exact 3D center in centimeters.
   - `surface_top_z_cm` -> Physical elevation of the top surface above table.
   - `height_cm` -> Object physical thickness/height.
   - `grasp_recommended_z_cm` -> Ideal end-effector descent height for grasping.

### 🦾 RULES OF MANIPULATION & SAFE EXECUTION:
1. **ACTION ATOMICITY**: ISSUE ONLY ONE TOOL CALL PER TURN.
2. **TWO-STAGE APPROACH FOR GRASPING**:
   - Never plunge directly into an object horizontally.
   - Step A: Move to pre-grasp clearance position: (target_x, target_y, target_z + 12.0 cm).
   - Step B: Descend vertically to `grasp_recommended_z_cm` (or surface_top_z_cm * 0.6).
   - Step C: Call `set_claw(state="closed")`.
   - Step D: Lift vertically back to safe transit height (Z ~ 18.0 - 22.0 cm).
3. **CAMERA OCCLUSION & PARKING**:
   - If the arm occludes the camera or you need an unobstructed full view of the workspace, call `park_arm(pose="survey")`.
   - After completing all tasks, you can call `park_arm(pose="rest")`.
4. **DOMAIN CONTEXTS & EMBODIMENT**:
   - **Mars Rover**: Prioritize collecting target hematite/silica core samples into rover carousel slots; avoid loose sand.
   - **Chemistry Lab**: Handle reagent vials delicately, transfer into titration beaker, place contaminated items into hazardous waste bin.
   - **Electronics Assembly**: Align microchips to sockets, discard damaged capacitors to reject bin.
5. **ANOMALY HANDLING**:
   - If after 2 attempts an action fails or is ambiguous, explain clearly to the user and ask for guidance.
"""

ROBOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_arm",
            "description": "Moves the robotic arm end-effector to specified 3D workspace coordinates (X, Y, Z) in centimeters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_cm": {
                        "type": "number",
                        "description": "Target X coordinate in cm [0.0 to 50.0]."
                    },
                    "y_cm": {
                        "type": "number",
                        "description": "Target Y coordinate in cm [0.0 to 50.0]."
                    },
                    "z_cm": {
                        "type": "number",
                        "description": "Target Height (Z) coordinate in cm [0.0 to 28.0]."
                    }
                },
                "required": ["x_cm", "y_cm", "z_cm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_claw",
            "description": "Opens or closes the robotic end-effector claw/gripper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed"],
                        "description": "State to set the claw to ('open' or 'closed')."
                    }
                },
                "required": ["state"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "park_arm",
            "description": "Parks the arm out of the camera's line of sight or to safe rest pose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pose": {
                        "type": "string",
                        "enum": ["survey", "rest"],
                        "description": "'survey' tucks arm completely away to clear camera view; 'rest' folds arm into compact base home."
                    }
                },
                "required": ["pose"]
            }
        }
    }
]

# Frontier Model Presets & Providers
FRONTIER_PRESETS = {
    "qwen2.5-vl": {
        "provider": "openai_compatible",
        "base_url": os.getenv("QWEN_API_BASE", "http://127.0.0.1:1234/v1"),
        "api_key": os.getenv("QWEN_API_KEY", "lm-studio"),
        "model_name": "qwen2.5-vl-7b-instruct",
        "description": "Open-source frontier for 2D/3D visual grounding & spatial coordinates"
    },
    "deepseek-vl2": {
        "provider": "openai_compatible",
        "base_url": os.getenv("DEEPSEEK_API_BASE", "http://127.0.0.1:1234/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", "lm-studio"),
        "model_name": "deepseek-vl2-small",
        "description": "Frontier dynamic high-res visual reasoning & MoE cognitive deduction"
    },
    "gemini-2.0": {
        "provider": "openai_compatible",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
        "model_name": "gemini-2.0-flash",
        "description": "State-of-the-art multimodal spatial perception & ultra-fast reasoning"
    },
    "lm-studio": {
        "provider": "openai_compatible",
        "base_url": os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1"),
        "api_key": "lm-studio",
        "model_name": "gemma-4-12b-it@q4_k_m",
        "description": "Local workstation inference endpoint via LM Studio"
    },
    "ollama": {
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "model_name": "qwen2.5-vl:latest",
        "description": "Local Ollama vision runner"
    }
}

ACTIVE_MODEL_PRESET = os.getenv("ROBO_MODEL_PRESET", "lm-studio")

agent_harness_CONFIG = {
    "ACTIVE_PRESET": ACTIVE_MODEL_PRESET,
    "PRESETS": FRONTIER_PRESETS,
    "LM_STUDIO_URL": FRONTIER_PRESETS[ACTIVE_MODEL_PRESET]["base_url"],
    "API_KEY": FRONTIER_PRESETS[ACTIVE_MODEL_PRESET]["api_key"],
    "MODEL_NAME": FRONTIER_PRESETS[ACTIVE_MODEL_PRESET]["model_name"],
    "TEMPERATURE": 0.45,
    "SYSTEM_PROMPT": sysPrompt,
    "TEST_IMAGE_PATH": "test_artifacts/test_image6.jpg",
    
    # Absolute Physical Guardrails (Centimeters)
    "BOUNDS": {
        "X_MIN": 0.0, "X_MAX": 50.0,
        "Y_MIN": 0.0, "Y_MAX": 50.0,
        "Z_MIN": 0.0, "Z_MAX": 28.0
    }
}

# ==============================================================================
# 🎮 SIMULATOR BACKEND CONFIGURATION (PyBullet + Depth + Multi-Scenarios)
# ==============================================================================
simulator_CONFIG = {
    "HOST": "127.0.0.1",
    "PORT": 8080,
    
    # Frustum & Camera Optimization:
    # Elevated to 1.05m with 65 FOV gives complete workspace visibility without arm link occlusion
    "CAMERA_RESOLUTION": (640, 480),
    "CAMERA_FOV": 65.0,
    "CAMERA_EYE_POSITION": [0.25, 0.25, 1.05],
    "CAMERA_TARGET_POSITION": [0.25, 0.25, 0.0],
    "CAMERA_UP_VECTOR": [0, 1, 0],
    "CAMERA_NEAR": 0.1,
    "CAMERA_FAR": 2.0,
    
    # Arm Kinematics & Scaling:
    # Compact scale 0.58 makes arm agile and prevents it from dominating/blocking the camera frustum
    "ARM_SCALE": 0.58,
    "ARM_BASE_POSITION": [-0.20, 0.25, -0.04],
    
    # Secondary Depth Sensor Simulation:
    "SECONDARY_DEPTH_SENSOR": {
        "ENABLED": True,
        "FOV": 65.0,
        "NEAR": 0.1,
        "FAR": 2.0,
        "RESOLUTION": (640, 480)
    },
    
    # Scenarios & Visuals:
    "SCENE_PRESET": "sorting",
    "SUPPORTED_SCENARIOS": ["sorting", "mars_rover", "chemistry_lab", "manufacturing_plant"],
    "ESP32_NOISE": False,
    "GUI_ENABLED": True
}

get_image_CONFIG = {
    "ESP32_IP": f"{simulator_CONFIG['HOST']}:{simulator_CONFIG['PORT']}",
    "ENDPOINT": "/capture",
    "DEPTH_ENDPOINT": "/capture_depth",
    "TIMEOUT_SECONDS": 5,
    "OUTPUT_FILENAME": "raw_capture.jpg",
    "SAVE_DIR": "handshake/"
}