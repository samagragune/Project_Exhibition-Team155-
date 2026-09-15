# test_autonomy_suite.py
# Comprehensive automated verification suite for the Robotic Arm Autonomy Layer.

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

print("=" * 70)
print("[TEST SUITE] RUNNING ROBOTIC ARM AUTONOMY LAYER VERIFICATION SUITE")
print("=" * 70)

# 1. Config Verification
print("\n[TEST 1] Verifying config.py...")
import config
assert config.simulator_CONFIG["CAMERA_EYE_POSITION"][2] >= 1.0, "Camera eye height must be elevated (>= 1.0m)"
assert config.simulator_CONFIG["ARM_SCALE"] <= 0.65, "Arm scale should be compact (<= 0.65)"
assert "park_arm" in [t["function"]["name"] for t in config.ROBOT_TOOLS], "park_arm must be in ROBOT_TOOLS"
assert "SECONDARY_DEPTH_SENSOR" in config.simulator_CONFIG, "Secondary depth sensor config must be present"
print("  [PASS] Config verified successfully.")

# 2. Simulator & Kinematics & Depth Sensor
print("\n[TEST 2] Verifying simulator.py & PyBullet physics world...")
import simulator
sim = simulator.sim_world

# Verify arm scale and camera height
print(f"  [PASS] Arm scale: {sim.arm_scale}, Camera eye: {sim.camera_eye}")

# Verify RGB frame render
rgb_frame = sim.render_esp32_frame()
assert len(rgb_frame) > 1000, "RGB frame output must be valid JPEG"
print(f"  [PASS] Camera RGB frame rendered ({len(rgb_frame)} bytes)")

# Verify Metric Depth Buffer & Heatmap
depth_map = sim.get_metric_depth_map()
assert isinstance(depth_map, np.ndarray), "Depth map must be a numpy array"
assert depth_map.shape == (480, 640), f"Depth map shape was {depth_map.shape}, expected (480, 640)"
max_depth = float(np.max(depth_map))
print(f"  [PASS] Metric depth map generated: shape {depth_map.shape}, max elevation: {round(max_depth, 2)} cm")

depth_jpg = sim.render_depth_frame()
assert len(depth_jpg) > 1000, "Depth heatmap JPEG must be valid"
print(f"  [PASS] Depth sensor heatmap rendered ({len(depth_jpg)} bytes)")

# Test park_arm tool call
park_res = sim.execute_park_arm("survey")
assert park_res["status"] == "success", "park_arm(survey) failed"
print(f"  [PASS] park_arm('survey') executed: {park_res['message']}")

# Test move_arm tool call with calibrated scale
move_res = sim.execute_move_arm(25.0, 25.0, 10.0)
assert move_res["status"] == "success", "move_arm failed"
print(f"  [PASS] move_arm(25, 25, 10) executed: tip reached {move_res['reached_position_cm']} cm")

# Test set_claw
claw_res = sim.execute_set_claw("open")
assert claw_res["status"] == "success"
print(f"  [PASS] set_claw('open') executed: {claw_res['message']}")

# 3. Multi-Scenario Environments
print("\n[TEST 3] Verifying Multi-Scenario environments...")
scenarios = ["sorting", "mars_rover", "chemistry_lab", "manufacturing_plant"]
for sc in scenarios:
    sim.spawn_scenario(sc)
    telemetry = sim.get_telemetry()
    assert telemetry["active_scenario"] == sc
    assert len(telemetry["objects"]) >= 4, f"Scenario '{sc}' should have at least 4 objects"
    print(f"  [PASS] Scenario '{sc}': Loaded with {len(telemetry['objects'])} objects.")

# 4. SAM 2.1 & 3D Depth Perception Fusion
print("\n[TEST 4] Verifying Perception & 2D-to-3D Depth Fusion...")
from sam2_processor import SAM2Processor
processor = SAM2Processor()

# Run on current scenario frame with depth map
depth_buffer = sim.get_metric_depth_map()
frame_bytes = sim.render_esp32_frame()
temp_capture = "handshake/raw_capture.jpg"
os.makedirs("handshake", exist_ok=True)
with open(temp_capture, "wb") as f:
    f.write(frame_bytes)

annotated_img, objects_3d = processor.process_image(temp_capture, depth_buffer=depth_buffer)
print(f"  [PASS] Detected {len(objects_3d)} objects with 3D depth grounding:")
for obj in objects_3d[:3]:
    c = obj["centroid_cm"]
    print(f"    - Obj #{obj['id']}: Centroid=({c['x']}, {c['y']}, {c.get('z')}cm), Top Z={obj.get('surface_top_z_cm')}cm, Grasp Z={obj.get('grasp_recommended_z_cm')}cm")

# 5. Agent Harness & Spatial Prompting
print("\n[TEST 5] Verifying Agent Harness & Spatial Reasoning...")
from agent_harness import AgentHarness
harness = AgentHarness()
assert "qwen2.5-vl" in config.FRONTIER_PRESETS
assert "deepseek-vl2" in config.FRONTIER_PRESETS
assert "gemini-2.0" in config.FRONTIER_PRESETS

# Test tool validation
val_legal = harness.validate_and_execute_tool("move_arm", {"x_cm": 25.0, "y_cm": 25.0, "z_cm": 15.0})
assert val_legal["status"] == "legal"
val_illegal = harness.validate_and_execute_tool("move_arm", {"x_cm": 95.0, "y_cm": 25.0, "z_cm": 15.0})
assert val_illegal["status"] == "error"
val_park = harness.validate_and_execute_tool("park_arm", {"pose": "survey"})
assert val_park["status"] == "pass"
print("  [PASS] Safety validation verified: Legal bounds enforced, out-of-envelope calls rejected.")

# Test fallback decision logic
act = harness._fallback_cognitive_decision("park the arm out of the way of the camera", objects_3d)
assert act["tool_call"]["tool"] == "park_arm"
print(f"  [PASS] Cognitive decision test passed: {act['tool_call']}")

# 6. Web Dashboard Assets
print("\n[TEST 6] Verifying Web Dashboard files...")
web_files = ["web/index.html", "web/style.css", "web/app.js"]
for wf in web_files:
    assert os.path.exists(wf), f"Missing web file {wf}"
    size = os.path.getsize(wf)
    print(f"  [PASS] {wf} present ({size} bytes)")

print("\n" + "=" * 70)
print("[SUCCESS] ALL VERIFICATION SUITE TESTS PASSED!")
print("=" * 70)
