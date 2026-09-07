# =========================================================================================================================
# config for `sam2_processor.py`
# =========================================================================================================================
sam2_processor_CONFIG = {
    # File Paths
    "MODEL_PATH": "sam2.1_l.pt",
    "INPUT_IMAGE": "handshake/raw_capture.jpg",       # Default image to load during test runs
    "OUTPUT_IMAGE": "handshake/annotated_output.jpg", # File name for the rendered output
    
    # Calibration & Hardware Settings
    "PIXELS_PER_CM": 30.0,                  # Scale factor: pixels per centimeter
    "FILTER_SCANLINES": True,               # Apply vertical median filter to mitigate ESP32 noise
    
    # SAM 2.1 Detection Parameters
    "CONFIDENCE_THRESHOLD": 0.85,           # Minimum confidence for object detection
    "FILTER_NESTED": False,                  # Remove child/sub-masks inside larger objects
    "NESTED_OVERLAP_RATIO": 0.85,           # Overlap threshold (85%) to consider a mask "nested"
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
4. You may or may not somtimes have to move the robotic arm you are controlling ou tof the way of the camera to view the objects on the workspace. a good default is to move to (0, 0, max height) and then (0, 0, 0) as well if needed
5. In any onomoly or confusing situation, do not keep trying for more than 2 tries. it is more reliable to quit and coem to the user for help in situations where you don't know a direct or obvious answer

### AVAILABLE TOOLS:
- `move_arm(x_cm: float, y_cm: float, z_cm: float)`: Moves the arm end-effector to specific 3D coordinates in centimeters.
- `set_claw(state: str)`: Controls end-effector claw ('open' or 'closed').
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
    "ESP32_NOISE": False,                           # Enable hardware noise simulation
    "SCENE_PRESET": "sorting",                     # Scene presets: "sorting", "cluttered", "space"
    "GUI_ENABLED": True                            # Enable PyBullet live 3D window
}

get_image_CONFIG = {
    "ESP32_IP": f"{simulator_CONFIG['HOST']}:{simulator_CONFIG['PORT']}",
    "ENDPOINT": "/capture",
    "TIMEOUT_SECONDS": 5,
    "OUTPUT_FILENAME": "raw_capture.jpg",
    "SAVE_DIR": "handshake/"
}

# testing prompt: 
# I want you to grab the blue box for me. for now simply approach it and get in position to grab at height 2cm. but do not actually grab. wait for my approval that 2cm turned out to be the right height befor egrabbing. approach slowly form above, fast swign into position will hit the block out of position. go to height of 20 cm directly above the block into position, then lower, then wait for my approval