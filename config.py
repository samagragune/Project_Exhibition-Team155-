# =========================================================================================================================
# config for `sam2_processor.py`
# =========================================================================================================================

import os

sam2_processor_CONFIG = {
    # File Paths
    "MODEL_PATH": "sam2.1_l.pt",
    "INPUT_IMAGE": "handshake/raw_capture.jpg",       # Default image to load during test runs
    "OUTPUT_IMAGE": "handshake/annotated_output.jpg", # File name for the rendered output
    
    # Calibration & Hardware Settings
    "PIXELS_PER_CM": 6.53,                # ~640px / (2*0.85m*tan(30°)*100) = 640/98cm at 0.85m/60°FOV
    "FILTER_SCANLINES": True,               # Apply vertical median filter to mitigate ESP32 noise
    
    # SAM 2.1 Detection Parameters
    "CONFIDENCE_THRESHOLD": 0.85,           # Minimum confidence for object detection
    "FILTER_NESTED": False,                  # Remove child/sub-masks inside larger objects
    "NESTED_OVERLAP_RATIO": 0.85,           # Overlap threshold (85%) to consider a mask "nested"
    "MIN_MASK_AREA_PX": 30,
    "MIN_MASK_CIRCULARITY": 0.02,
    
    # Secondary Depth Sensor Fusion
    "DEPTH_FUSION_ENABLED": True,
    "DEPTH_FILTER_PERCENTILE": 75.0,        # Percentile sampling to extract object top surface
    "DEFAULT_OBJECT_Z_MARGIN_CM": 1.5       # Clearance margin above surface
}
# -------------------------------------------------------------------------------------------------------------------------


# =========================================================================================================================
# config for `agent_harness.py`
# =========================================================================================================================
sysPrompt = """You are an AI spatial reasoning agent controlling a physical 3D robotic arm over a 50cm x 50cm workspace plane.

### WORKSPACE BOUNDARIES & HARD LIMITS:
All coordinates passed to `move_arm` MUST stay strictly within these physical boundaries:
- X Coordinate: [MIN: 0.0 cm, MAX: 50.0 cm]
- Y Coordinate: [MIN: 0.0 cm, MAX: 50.0 cm]
- Z Coordinate (Height): [MIN: 0.0 cm, MAX: 25.0 cm]
  * Z = 0.0 cm: Table surface level.
  * Z = 25.0 cm: Maximum height ceiling.

### HARDWARE, PERCEPTION & HEIGHT CONSTRAINTS:
1. PERCEPTION: Vision uses SAM 2.1 calibrated by corner markers. Treat detected coordinates (in cm) as target positions.
2. HEIGHT ESTIMATION: Set Z high (~15.0cm - 20.0cm) when moving over objects. Descend to ~2.0cm-4.0cm only when grasping.
3. ACTION ATOMICITY & SINGLE-TOOL RULE: ISSUE ONLY ONE TOOL CALL PER TURN.
4. If the robotic arm is covering your camera view, call park_arm(pose="survey") to tuck it out of the way and get a clear view.
5. In any anomaly or confusing situation, do not keep trying for more than 2 tries. Explain to the user and ask for guidance.
6. If you cannot see objects because the arm itself is in the way, park it first before proceeding.
7. CRITICAL: Tool execution feedback (e.g., status=success) only confirms the hardware received and executed the exact command — it does NOT verify the intended outcome. You MUST confirm success by examining the CURRENT FRAME image and the SAM 2.1 metadata. If the image shows the cube was pushed away or the claw grabbed nothing, you MUST retry with corrected coordinates.

### TERMINATION RULE:
- After completing your assigned task, output a text response summarizing what you did. Do NOT call park_arm or any other tool again once the task is complete. The autonomous loop will continue until you return a text response with no tool call.

### AVAILABLE TOOLS:
- `move_arm(x_cm: float, y_cm: float, z_cm: float)`: Moves the arm end-effector to specific 3D coordinates in centimeters.
- `set_claw(state: str)`: Controls end-effector claw ('open' or 'closed').
- `park_arm(pose: str)`: Parks the arm to clear the camera view ('survey') or fold into rest position ('rest').
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
                        "description": "Target Height (Z) coordinate in cm [0.0 to 25.0]."
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
            "description": "Opens or closes the robotic end-effector claw.",
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
            "description": "Parks the arm to clear the camera view or fold into rest position.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pose": {
                        "type": "string",
                        "enum": ["survey", "rest"],
                        "description": " 'survey' tucks arm out of camera view; 'rest' folds into compact home."
                    }
                },
                "required": ["pose"]
            }
        }
    }
]

agent_harness_CONFIG = {
    "LM_STUDIO_URL": "http://26.50.165.63:1234/v1",
    "API_KEY": "lm-studio",
    "MODEL_NAME": "gemma-4-12b-it@q4_k_m",
    "TEMPERATURE": 0.55,
    "SYSTEM_PROMPT": sysPrompt,
    "TEST_IMAGE_PATH": "test_artifacts/test_image6.jpg",
    
    # Absolute Physical Guardrails
    "BOUNDS": {
        "X_MIN": 0.0, "X_MAX": 50.0,
        "Y_MIN": 0.0, "Y_MAX": 50.0,
        "Z_MIN": 0.0, "Z_MAX": 30.0
    }
}

# Frontier Model Presets (OpenAI-compatible providers)
FRONTIER_PRESETS = {
    "lm-studio": {
        "base_url": agent_harness_CONFIG["LM_STUDIO_URL"],
        "api_key": agent_harness_CONFIG["API_KEY"],
        "model_name": agent_harness_CONFIG["MODEL_NAME"],
        "description": "Local workstation inference endpoint via LM Studio"
    },
    "qwen2.5-vl": {
        "base_url": os.getenv("QWEN_API_BASE", "http://127.0.0.1:1234/v1"),
        "api_key": os.getenv("QWEN_API_KEY", "lm-studio"),
        "model_name": "qwen2.5-vl-7b-instruct",
        "description": "Open-source frontier for 2D/3D visual grounding & spatial coordinates"
    },
    "deepseek-vl2": {
        "base_url": os.getenv("DEEPSEEK_API_BASE", "http://127.0.0.1:1234/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", "lm-studio"),
        "model_name": "deepseek-vl2-small",
        "description": "Frontier dynamic high-res visual reasoning & MoE cognitive deduction"
    },
    "gemini-2.0": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
        "model_name": "gemini-2.0-flash",
        "description": "State-of-the-art multimodal spatial perception & ultra-fast reasoning"
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "model_name": "qwen2.5-vl:latest",
        "description": "Local Ollama vision runner"
    }
}

ACTIVE_MODEL_PRESET = os.getenv("ROBO_MODEL_PRESET", "lm-studio")
if ACTIVE_MODEL_PRESET not in FRONTIER_PRESETS:
    raise ValueError(
        f"Invalid model preset '{ACTIVE_MODEL_PRESET}'. "
        f"Set ROBO_MODEL_PRESET to one of: {list(FRONTIER_PRESETS.keys())}, "
        f"or unset the env var to default to 'lm-studio'."
    )

agent_harness_CONFIG["ACTIVE_PRESET"] = ACTIVE_MODEL_PRESET
agent_harness_CONFIG["PRESETS"] = FRONTIER_PRESETS

# -------------------------------------------------------------------------------------------------------------------------

# ==============================================================================
# 🎮 SIMULATOR BACKEND CONFIGURATION
# ==============================================================================
simulator_CONFIG = {
    "HOST": "127.0.0.1",
    "PORT": 8080,
    "CAMERA_RESOLUTION": (640, 480),
    "CAMERA_FOV": 60.0,
    "CAMERA_EYE_POSITION": [0.25, 0.25, 0.85],  # Zoomed out higher above middle of 50x50 workspace
    "CAMERA_TARGET_POSITION": [0.25, 0.25, 0.0],
    "CAMERA_UP_VECTOR": [0, 1, 0],
    "CAMERA_NEAR": 0.1,
    "CAMERA_FAR": 2.0,
    "ESP32_NOISE": False,
    "SCENE_PRESET": "sorting",
    "SUPPORTED_SCENARIOS": ["sorting", "mars_rover", "chemistry_lab", "manufacturing_plant"],
    "ARM_SCALE": 0.75,
    "GUI_ENABLED": True,
    "MAX_AUTO_TURNS": 30,
    "SECONDARY_DEPTH_SENSOR": {
        "ENABLED": True,
        "FOV": 60.0,
        "NEAR": 0.1,
        "FAR": 2.0,
        "RESOLUTION": (640, 480)
    }
}

get_image_CONFIG = {
    "ESP32_IP": f"{simulator_CONFIG['HOST']}:{simulator_CONFIG['PORT']}",
    "ENDPOINT": "/capture",
    "DEPTH_ENDPOINT": "/capture_depth",
    "TIMEOUT_SECONDS": 5,
    "OUTPUT_FILENAME": "raw_capture.jpg",
    "SAVE_DIR": "handshake/"
}

# testing prompt: 
# I want you to grab the blue box for me. first simply approach it and get in position to grab at height 2cm. but do not actually grab. approach slowly form above, fast swign into position will hit the block out of position. go to height of 20 cm directly above the block into position, then lower, then grab. After grabbing, lift the bue box to a height of 20 cm.