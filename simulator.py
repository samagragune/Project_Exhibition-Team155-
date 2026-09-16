# simulator.py
# PyBullet physics simulation + Secondary Depth Sensor + Multi-Scenario Worlds + FastAPI Server.
# Replaces the physical ESP32-CAM + robotic arm hardware backend.
# Endpoints: GET /capture (JPEG), POST /execute_tool (tool call dispatch)

import os
import time
import math
import shutil
import threading
import numpy as np
import cv2
import subprocess
import pybullet as p
import pybullet_data
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import config

def ensure_ur5_downloaded():
    """Downloads PyBullet UR5 model repository automatically if not present."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    ur5_path = os.path.join(models_dir, "ur5_pybullet")
    
    if not os.path.exists(ur5_path):
        print("[Simulator] Downloading PyBullet UR5 model repository...")
        os.makedirs(models_dir, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Alchemist77/pybullet-ur5-equipped-with-robotiq-140.git",
            ur5_path
        ], check=True)
        print("[Simulator] Download complete.")
    
    return os.path.join(ur5_path, "urdf", "ur5_robotiq_140.urdf")

cfg = config.simulator_CONFIG

app = FastAPI(title="Robotic Arm Hardware Simulator")

class WorldSimulator:
    """PyBullet physics world with UR5 arm, Robotiq 140 gripper, and simulated camera.
    Handles IK solving, grasp physics, and frame rendering."""

    def __init__(self):
        # 1. Connect GUI Physics Engine
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        
        # Increase constraint solver iterations to prevent gripper model breaking/jittering under load
        p.setPhysicsEngineParameter(numSolverIterations=3000, physicsClientId=self.physics_client)

        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=0,
            cameraPitch=-89,  # Direct top-down view
            cameraTargetPosition=[0.25, 0.25, 0.0]
        )

        # Initialize tracking variables
        self.object_ids = {}
        self.held_constraint = None
        self.held_object_id = None
        self.active_scenario = "sorting"

        # Camera configuration
        self.camera_eye = list(cfg.get("CAMERA_EYE_POSITION", [0.25, 0.25, 0.80]))
        self.camera_target = list(cfg.get("CAMERA_TARGET_POSITION", [0.25, 0.25, 0.0]))
        self.camera_up = list(cfg.get("CAMERA_UP_VECTOR", [0, 1, 0]))
        self.arm_scale = cfg.get("ARM_SCALE", 0.75)

        # 1b. Solid workspace plane (50cm x 50cm from 0,0 to 0.5,0.5)
        plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01])
        plane_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01], rgbaColor=[0.85, 0.85, 0.85, 1.0])
        self.plane_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plane_shape,
            baseVisualShapeIndex=plane_visual,
            basePosition=[0.25, 0.25, -0.01],
            physicsClientId=self.physics_client
        )

        # 2. Load UR5 Arm - Positioned at [-0.15, 0.25] (Off-camera to the left)
        ur5_urdf_file = ensure_ur5_downloaded()
        self.arm_id = p.loadURDF(
            ur5_urdf_file, 
            basePosition=[-0.15, 0.25, -0.05], 
            useFixedBase=True, 
            globalScaling=0.75,
            physicsClientId=self.physics_client
        )

        # 3. Discover Controllable Joints, Gripper Joints, and End Effector
        self.arm_joints = []
        self.gripper_joint_indices = []
        self.end_effector_idx = -1
        self.movable_joint_indices = []

        num_joints = p.getNumJoints(self.arm_id, physicsClientId=self.physics_client)
        for i in range(num_joints):
            info = p.getJointInfo(self.arm_id, i, physicsClientId=self.physics_client)
            joint_name = info[1].decode("utf-8").lower()
            joint_type = info[2]
            link_name = info[12].decode("utf-8").lower()

            if joint_type != p.JOINT_FIXED:
                self.movable_joint_indices.append(i)

            # Detect main arm motor joints
            if any(name in joint_name for name in ["shoulder", "elbow", "wrist"]):
                self.arm_joints.append(i)
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=1000,
                    physicsClientId=self.physics_client
                )

            # Detect gripper / finger joints dynamically
            if any(k in joint_name or k in link_name for k in ["finger", "gripper", "knuckle", "claw"]):
                self.gripper_joint_indices.append(i)

            # Detect end-effector link
            if "ee_link" in link_name or "tool0" in link_name:
                self.end_effector_idx = i

        if self.end_effector_idx == -1:
            self.end_effector_idx = self.arm_joints[-1] if self.arm_joints else 6

        # Fallback for gripper joints if model has non-standard joint names
        if not self.gripper_joint_indices:
            self.gripper_joint_indices = [j for j in self.movable_joint_indices if j not in self.arm_joints]

        # Folded rest position
        self.home_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        for idx, joint_idx in enumerate(self.arm_joints):
            if idx < len(self.home_joint_positions):
                p.resetJointState(self.arm_id, joint_idx, self.home_joint_positions[idx], physicsClientId=self.physics_client)

        print(f"[Simulator] UR5 ready. Arm Joints: {self.arm_joints} | Gripper Joints: {self.gripper_joint_indices}")

        # Spawn workspace items and calibration markers
        self._spawn_default_objects()
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
        print(f"[Simulator] Initialized. Active Scenario: '{self.active_scenario}', Arm Scale: {self.arm_scale}")

    def _spawn_default_objects(self):
        """Spawns the default scene: red soda can, blue cube, and 4 yellow
        calibration markers at workspace corners (5cm and 45cm)."""
        # 1. Red Soda Can
        can_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.08)
        can_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.08, rgbaColor=[0.9, 0.1, 0.1, 1.0])
        can_id = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=can_shape,
            baseVisualShapeIndex=can_visual,
            basePosition=[0.20, 0.20, 0.04],
            physicsClientId=self.physics_client
        )
        self.object_ids["red_soda_can"] = can_id

        # 2. Blue Cube
        blue_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
        blue_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=[0.1, 0.3, 0.9, 1.0])
        blue_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=blue_shape,
            baseVisualShapeIndex=blue_visual,
            basePosition=[0.32, 0.15, 0.025],
            physicsClientId=self.physics_client
        )
        self.object_ids["blue_cube"] = blue_id

        # 3. Spawn 4 Yellow Calibration Markers at Corners (5cm to 45cm)
        marker_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.012)
        corner_positions = [
            [0.05, 0.05, 0.01],   # (5 cm, 5 cm)
            [0.45, 0.05, 0.01],   # (45 cm, 5 cm)
            [0.05, 0.45, 0.01],   # (5 cm, 45 cm)
            [0.45, 0.45, 0.01]    # (45 cm, 45 cm)
        ]
        
        for idx, pos in enumerate(corner_positions):
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1.0, 1.0, 0.0, 1.0])
            marker_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=marker_shape,
                baseVisualShapeIndex=marker_visual,
                basePosition=pos,
                physicsClientId=self.physics_client
            )
            self.object_ids[f"calibration_marker_{idx}"] = marker_id

        for _ in range(30):
            p.stepSimulation(physicsClientId=self.physics_client)

    def spawn_scenario(self, scenario_name: str):
        """Clears and respawns objects for the target cognitive scenario."""
        # Remove existing dynamic objects
        for name, obj_id in list(self.object_ids.items()):
            if "calibration" not in name:
                try:
                    p.removeBody(obj_id, physicsClientId=self.physics_client)
                except Exception:
                    pass
                self.object_ids.pop(name, None)

        self.active_scenario = scenario_name
        spawn_methods = {
            "sorting": self._spawn_sorting,
            "mars_rover": self._spawn_mars_rover,
            "chemistry_lab": self._spawn_chemistry_lab,
            "manufacturing_plant": self._spawn_manufacturing_plant,
        }
        spawn_fn = spawn_methods.get(scenario_name, self._spawn_sorting)
        spawn_fn()
        print(f"[Simulator] Scenario '{scenario_name}' loaded successfully with {len(self.object_ids)} items.")

    def _spawn_sorting(self):
        """Standard sorting scenario: red soda can, blue cube, obstacle cylinder, yellow markers."""
        self._spawn_default_objects()

    def _spawn_mars_rover(self):
        """Mars rover scenario: regolith, hematite sample, olivine crystal, basalt rock, rover carousel."""
        regolith = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.02], rgbaColor=[0.45, 0.30, 0.20, 1.0])
        regolith_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=regolith, basePosition=[0.25, 0.25, -0.01], physicsClientId=self.physics_client)
        self.object_ids["martian_regolith"] = regolith_id

        hematite = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        hematite_v = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0.25, 0.15, 0.15, 1.0])
        self.object_ids["hematite_core_sample"] = p.createMultiBody(baseMass=0.3, baseCollisionShapeIndex=hematite, baseVisualShapeIndex=hematite_v, basePosition=[0.20, 0.18, 0.03], physicsClientId=self.physics_client)

        olivine = p.createVisualShape(p.GEOM_SPHERE, radius=0.025, rgbaColor=[0.0, 0.7, 0.4, 1.0])
        self.object_ids["olivine_crystal"] = p.createMultiBody(baseMass=0.15, baseVisualShapeIndex=olivine, basePosition=[0.35, 0.30, 0.025], physicsClientId=self.physics_client)

        basalt = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.04], rgbaColor=[0.2, 0.2, 0.22, 1.0])
        self.object_ids["basalt_discard_rock"] = p.createMultiBody(baseMass=0.4, baseVisualShapeIndex=basalt, basePosition=[0.40, 0.35, 0.02], physicsClientId=self.physics_client)

        carousel = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.02, rgbaColor=[0.15, 0.15, 0.15, 1.0])
        self.object_ids["rover_sample_carousel"] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=carousel, basePosition=[0.10, 0.10, 0.01], physicsClientId=self.physics_client)

        self._spawn_calibration_markers()

    def _spawn_chemistry_lab(self):
        """Chemistry lab scenario: clean bench, acid vial, base neutralizer, titration beaker, waste basin."""
        bench = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01], rgbaColor=[0.9, 0.9, 0.95, 0.3])
        self.object_ids["cleanroom_bench"] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=bench, basePosition=[0.25, 0.25, -0.005], physicsClientId=self.physics_client)

        acid = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.06)
        acid_v = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.06, rgbaColor=[0.9, 0.1, 0.3, 1.0])
        self.object_ids["acid_reagent_vial"] = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=acid, baseVisualShapeIndex=acid_v, basePosition=[0.22, 0.22, 0.03], physicsClientId=self.physics_client)

        base = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.06, rgbaColor=[0.1, 0.6, 0.9, 1.0])
        self.object_ids["base_neutralizer"] = p.createMultiBody(baseMass=0.1, baseVisualShapeIndex=base, basePosition=[0.30, 0.28, 0.03], physicsClientId=self.physics_client)

        beaker = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.05, rgbaColor=[0.7, 0.85, 0.95, 0.5])
        self.object_ids["titration_reaction_beaker"] = p.createMultiBody(baseMass=0.2, baseVisualShapeIndex=beaker, basePosition=[0.35, 0.18, 0.025], physicsClientId=self.physics_client)

        basin = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.03])
        basin_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.03], rgbaColor=[0.4, 0.0, 0.6, 0.7])
        self.object_ids["hazardous_waste_disposal_basin"] = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=basin, baseVisualShapeIndex=basin_v, basePosition=[0.45, 0.42, 0.03], physicsClientId=self.physics_client)

        self._spawn_calibration_markers()

    def _spawn_manufacturing_plant(self):
        """Electronics assembly scenario: ESD mat, microcontroller IC, valid cap, defective cap, PCB socket."""
        esd_mat = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01], rgbaColor=[0.0, 0.0, 0.3, 0.4])
        self.object_ids["esd_anti_static_mat"] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=esd_mat, basePosition=[0.25, 0.25, -0.005], physicsClientId=self.physics_client)

        ic_chip = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012, 0.012, 0.002], rgbaColor=[0.9, 0.85, 0.1, 1.0])
        self.object_ids["microcontroller_qfn_chip"] = p.createMultiBody(baseMass=0.01, baseVisualShapeIndex=ic_chip, basePosition=[0.28, 0.20, 0.002], physicsClientId=self.physics_client)

        valid_cap = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.03, rgbaColor=[0.1, 0.8, 0.3, 1.0])
        self.object_ids["valid_electrolytic_capacitor"] = p.createMultiBody(baseMass=0.05, baseVisualShapeIndex=valid_cap, basePosition=[0.33, 0.25, 0.015], physicsClientId=self.physics_client)

        defective_cap = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.03, rgbaColor=[0.8, 0.2, 0.2, 1.0])
        self.object_ids["defective_bulged_capacitor"] = p.createMultiBody(baseMass=0.05, baseVisualShapeIndex=defective_cap, basePosition=[0.40, 0.32, 0.015], physicsClientId=self.physics_client)

        pcb_socket = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.005], rgbaColor=[0.1, 0.1, 0.1, 1.0])
        self.object_ids["pcb_socket"] = p.createMultiBody(baseMass=0.3, baseVisualShapeIndex=pcb_socket, basePosition=[0.25, 0.25, 0.005], physicsClientId=self.physics_client)

        reject_chute = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.15, 0.02], rgbaColor=[0.4, 0.4, 0.4, 0.8])
        self.object_ids["qa_reject_chute"] = p.createMultiBody(baseMass=0, baseVisualShapeIndex=reject_chute, basePosition=[0.48, 0.25, 0.02], physicsClientId=self.physics_client)

        self._spawn_calibration_markers()

    def _spawn_calibration_markers(self):
        """Spawns 4 yellow calibration markers at workspace corners."""
        marker_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.012)
        corner_positions = [
            [0.05, 0.05, 0.01],
            [0.45, 0.05, 0.01],
            [0.05, 0.45, 0.01],
            [0.45, 0.45, 0.01]
        ]
        for idx, pos in enumerate(corner_positions):
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1.0, 1.0, 0.0, 1.0])
            marker_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=marker_shape,
                baseVisualShapeIndex=marker_visual,
                basePosition=pos,
                physicsClientId=self.physics_client
            )
            self.object_ids[f"calibration_marker_{idx}"] = marker_id

    def render_esp32_frame(self, resolution=None) -> bytes:
        """Renders a top-down JPEG frame from the PyBullet camera.
        Optionally applies ESP32 scanline noise artifacts."""
        res_w, res_h = resolution if resolution else cfg.get("CAMERA_RESOLUTION", (640, 480))

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_eye,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cfg.get("CAMERA_FOV", 60.0),
            aspect=float(res_w) / res_h,
            nearVal=cfg.get("CAMERA_NEAR", 0.1),
            farVal=cfg.get("CAMERA_FAR", 2.0)
        )

        # Try hardware renderer first, fall back to CPU TinyRenderer
        rgb_pixels = None
        for renderer in (p.ER_BULLET_HARDWARE_OPENGL, p.ER_TINY_RENDERER):
            try:
                _, _, rgb_pixels, _, _ = p.getCameraImage(
                    width=res_w,
                    height=res_h,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=renderer
                )
                if rgb_pixels and len(rgb_pixels) == res_w * res_h * 4:
                    break
            except Exception:
                rgb_pixels = None
                continue
        
        if not rgb_pixels or len(rgb_pixels) != res_w * res_h * 4:
            # Last resort: return a blank frame instead of crashing
            blank = np.zeros((res_h, res_w, 3), dtype=np.uint8)
            _, enc_img = cv2.imencode('.jpg', blank)
            return enc_img.tobytes()

        frame_rgba = np.reshape(rgb_pixels, (res_h, res_w, 4)).astype(np.uint8)
        frame_rgb = frame_rgba[:, :, :3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if cfg.get("ESP32_NOISE", False):
            scanlines = np.ones_like(frame_bgr, dtype=np.float32)
            scanlines[::4, :, :] = 0.90
            frame_bgr = (frame_bgr.astype(np.float32) * scanlines).clip(0, 255).astype(np.uint8)

        _, enc_img = cv2.imencode('.jpg', frame_bgr)
        return enc_img.tobytes()

    def get_metric_depth_map(self) -> np.ndarray:
        """Extracts a metric depth buffer from PyBullet (Z in camera space).
        Returns depth in centimeters above the table surface (Z=0 plane).
        Positive values indicate objects above the table plane.
        Uses the same camera parameters as render_esp32_frame for pixel alignment."""
        res_w, res_h = cfg.get("CAMERA_RESOLUTION", (640, 480))

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_eye,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cfg.get("CAMERA_FOV", 60.0),
            aspect=float(res_w) / res_h,
            nearVal=cfg.get("CAMERA_NEAR", 0.1),
            farVal=cfg.get("CAMERA_FAR", 2.0)
        )

        depth_buf = None
        for renderer in (p.ER_BULLET_HARDWARE_OPENGL, p.ER_TINY_RENDERER):
            try:
                _, depth_buf = p.getCameraImage(
                    width=res_w,
                    height=res_h,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix,
                    renderer=renderer
                )[2:4]
                if depth_buf is not None and len(depth_buf) == res_w * res_h:
                    break
            except Exception:
                depth_buf = None
                continue

        if depth_buf is None or len(depth_buf) != res_w * res_h:
            return np.zeros((res_h, res_w), dtype=np.float64)

        depth_array = np.reshape(depth_buf, (res_h, res_w)).astype(np.float64)
        # PyBullet returns depth bottom-up (row 0 = bottom). OpenCV images are top-down.
        # Flip vertically so depth[row, col] aligns with the RGB frame pixel coordinates.
        depth_array = np.flipud(depth_array)

        # Deproject from normalized [0,1] depth to metric distance from camera
        near, far = cfg.get("CAMERA_NEAR", 0.1), cfg.get("CAMERA_FAR", 2.0)
        depth_array = near * far / (far - depth_array * (far - near))

        # Camera looks straight down from self.camera_eye to self.camera_target.
        # World Z height of surface = camera_height - distance_from_camera
        table_height = 0.0
        elevation_m = self.camera_eye[2] - depth_array
        elevation_cm = (elevation_m - table_height) * 100.0
        elevation_cm = np.clip(elevation_cm, 0, None)

        return elevation_cm

    def render_depth_frame(self) -> bytes:
        """Renders the depth heatmap as a JPEG using turbo colormap."""
        depth_cm = self.get_metric_depth_map()
        max_depth = 12.0
        norm_depth = np.clip((depth_cm / max_depth) * 255.0, 0, 255).astype(np.uint8)

        depth_bgr = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)
        _, enc_img = cv2.imencode('.jpg', depth_bgr)
        return enc_img.tobytes()

    def execute_move_arm(self, x_cm: float, y_cm: float, z_cm: float):
        """Solves inverse kinematics to move the UR5 arm end-effector to the
        target (x, y, z) in centimeters. Applies 12cm tool-length offset so
        the fingertips land precisely on the target position."""
        # Target coordinates shifted forward slightly in world space so the 
        # actual fingertips (accounting for tool length) land on the target.
        target_pos = [x_cm / 100.0, y_cm / 100.0, z_cm / 100.0]
        print(f"\n[Tool Call Received] Moving UR5 end-effector to position (m): {target_pos}")

        downward_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        # Treat the entire gripper as an extended limb. 
        # Add a fixed 12cm forward/downward reach offset so the wrist flange 
        # stays back, allowing the fingertips to touch the table precisely.
        effective_length = 0.12 
        
        local_limb_vector = np.array([0.0, 0.0, -effective_length])
        rotation_matrix = np.array(p.getMatrixFromQuaternion(downward_orientation)).reshape(3, 3)
        world_limb_offset = rotation_matrix.dot(local_limb_vector)

        # Shift the target vector so the wrist compensates for the limb length
        adjusted_target_pos = [
            target_pos[0] - world_limb_offset[0],
            target_pos[1] - world_limb_offset[1],
            target_pos[2] - world_limb_offset[2]
        ]

        # Use the correct tool tip link (or end_effector_idx with explicit offset if needed)
        # For the Robotiq 140 model, target the precise operational tip
        ik_target_idx = self.end_effector_idx
        
        # Adjust target slightly down along the tool Z-axis so fingertips land on table plane accurately
        effective_tip_offset = [0.0, 0.0, 0.08]
        rotation_matrix = np.array(p.getMatrixFromQuaternion(downward_orientation)).reshape(3, 3)
        adjusted_target_pos = [
            target_pos[0] + rotation_matrix[0, 2] * effective_tip_offset[2],
            target_pos[1] + rotation_matrix[1, 2] * effective_tip_offset[2],
            target_pos[2] + rotation_matrix[2, 2] * effective_tip_offset[2]
        ]

        # Collect joint limits and ranges for stable constrained IK solving
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
        for j_idx in self.movable_joint_indices:
            info = p.getJointInfo(self.arm_id, j_idx, physicsClientId=self.physics_client)
            ll, ul = info[8], info[9]
            if ll > ul:  # Handle continuous joints if any
                ll, ul = -2 * math.pi, 2 * math.pi
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)
            rest_poses.append((ll + ul) / 2.0)

        # Shift the IK target upwards (compensating for the gripper tool length) 
        # so that when the arm reaches the IK target, the fingertips land precisely on the original target.
        # Only this final offset is used (first two adjustments are overwritten).
        gripper_length_offset = 0.12  # 12 cm tool length offset (legacy value, works for arm_scale=0.75)
        adjusted_target_pos = [
            target_pos[0] - gripper_length_offset,
            target_pos[1],
            target_pos[2]
        ]

        joint_poses = p.calculateInverseKinematics(
            self.arm_id,
            ik_target_idx,
            targetPosition=adjusted_target_pos,
            targetOrientation=downward_orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=500,
            residualThreshold=0.001,
            physicsClientId=self.physics_client
        )

        for joint_idx in self.arm_joints:
            dof_index = self.movable_joint_indices.index(joint_idx)
            if dof_index < len(joint_poses):
                target_angle = joint_poses[dof_index]
                
                if joint_idx == self.arm_joints[-1]:
                    target_angle = 1.57
                    
                p.setJointMotorControl2(
                    bodyUniqueId=self.arm_id,
                    jointIndex=joint_idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_angle,
                    force=1000,
                    maxVelocity=2.0,
                    physicsClientId=self.physics_client
                )

        for _ in range(240):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1.0 / 240.0)

        current_pos = p.getLinkState(self.arm_id, self.end_effector_idx, physicsClientId=self.physics_client)[0]
        print(f"[Simulator] Movement Complete. End Effector Position: {[round(c, 3) for c in current_pos]}\n")
        return {"status": "success", "reached_position_m": list(current_pos)}

    def execute_set_claw(self, state: str) -> dict:
        """Opens or closes the Robotiq 140 gripper. On close, checks proximity
        to nearby objects and attaches the closest one via a fixed physics
        constraint. Returns status dict with execution details."""
        state_str = str(state).strip().lower()
        is_closing = state_str in ["close", "closed"]

        print(f"\n[Tool Call Received] Setting Claw State: '{state_str.upper()}'")

        # Visual indicator color: RED for Closed, GREEN for Open
        indicator_color = [1.0, 0.0, 0.0, 1.0] if is_closing else [0.0, 1.0, 0.0, 1.0]
        p.changeVisualShape(self.arm_id, self.end_effector_idx, rgbaColor=indicator_color, physicsClientId=self.physics_client)

        if not is_closing:
            if getattr(self, 'held_constraint', None) is not None:
                p.removeConstraint(self.held_constraint, physicsClientId=self.physics_client)
                self.held_constraint = None
                self.held_object_id = None
                print("[Simulator] Claw Opened: Constraint Released.")

        # Filter gripper joints to drive only the primary driver knuckle/finger joints, avoiding locked mimic joints
        target_finger_pos = 0.65 if is_closing else 0.0  
        
        driver_gripper_joints = []
        for g_idx in self.gripper_joint_indices:
            info = p.getJointInfo(self.arm_id, g_idx, physicsClientId=self.physics_client)
            j_name = info[1].decode("utf-8").lower()
            # Target main driver knuckles which control the sub-links
            if "knuckle" in j_name or "finger_joint" in j_name:
                driver_gripper_joints.append(g_idx)
        
        if not driver_gripper_joints:
            driver_gripper_joints = self.gripper_joint_indices[:2] # Fallback to first few joints

        for g_idx in driver_gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_id,
                jointIndex=g_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_finger_pos,
                force=200,
                maxVelocity=1.0,
                physicsClientId=self.physics_client
            )

        for _ in range(120):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1.0 / 240.0)

        # Attach object with fixed physics constraint when closing near an object
        if is_closing and getattr(self, 'held_constraint', None) is None:
            ee_state = p.getLinkState(self.arm_id, self.end_effector_idx, physicsClientId=self.physics_client)
            ee_pos = ee_state[0]
            
            for obj_name, obj_id in self.object_ids.items():
                if "calibration" in obj_name:
                    continue
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client)
                dist = sum((e - o) ** 2 for e, o in zip(ee_pos, obj_pos)) ** 0.5
                
                if dist < 0.15:  # Increased tolerance slightly to ensure easy grabbing
                    self.held_constraint = p.createConstraint(
                        parentBodyUniqueId=self.arm_id,
                        parentLinkIndex=self.end_effector_idx,
                        childBodyUniqueId=obj_id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=[0, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0],
                        physicsClientId=self.physics_client
                    )
                    self.held_object_id = obj_id
                    print(f"[Simulator] Claw Closed: Successfully attached '{obj_name}' (distance: {round(dist*100, 1)} cm).")
                    break

        return {"status": "success", "message": f"Claw set to {state_str}"}

    def execute_park_arm(self, pose: str = "survey") -> dict:
        """Parks the arm in a safe pose to clear the camera frustum."""
        pose_str = str(pose).strip().lower()

        if pose_str == "survey":
            # Tuck arm back and left, completely clear of top-down camera frustum
            survey_pose = [-1.57, -1.8, 1.8, -1.57, 1.57, 0.0]
            for idx, joint_idx in enumerate(self.arm_joints):
                if idx < len(survey_pose):
                    p.setJointMotorControl2(
                        bodyUniqueId=self.arm_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=survey_pose[idx],
                        force=1000,
                        maxVelocity=2.5,
                        physicsClientId=self.physics_client
                    )
            msg = "Arm parked in 'survey' pose. Camera line of sight is 100% unobstructed."
        else:
            # Home folded resting pose
            for idx, joint_idx in enumerate(self.arm_joints):
                if idx < len(self.home_joint_positions):
                    p.setJointMotorControl2(
                        bodyUniqueId=self.arm_id,
                        jointIndex=joint_idx,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=self.home_joint_positions[idx],
                        force=1000,
                        maxVelocity=2.0,
                        physicsClientId=self.physics_client
                    )
            msg = "Arm folded into 'rest' home position."

        for _ in range(120):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1.0 / 240.0)

        current_pos = p.getLinkState(self.arm_id, self.end_effector_idx, physicsClientId=self.physics_client)[0]
        return {
            "status": "success",
            "message": msg,
            "pose": pose_str,
            "reached_position_m": list(current_pos)
        }

    def get_telemetry(self) -> dict:
        """Returns current end-effector position, claw state, and scenario."""
        ee_pos = p.getLinkState(self.arm_id, self.end_effector_idx, physicsClientId=self.physics_client)[0]
        objects_telemetry = []

        for obj_name, obj_id in self.object_ids.items():
            if "calibration" in obj_name:
                continue
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client)
            objects_telemetry.append({
                "name": obj_name,
                "position_cm": [round(pos[0] * 100.0, 1), round(pos[1] * 100.0, 1), round(pos[2] * 100.0, 1)]
            })

        return {
            "active_scenario": self.active_scenario,
            "end_effector_cm": {
                "x": round(ee_pos[0] * 100.0, 1),
                "y": round(ee_pos[1] * 100.0, 1),
                "z": round(ee_pos[2] * 100.0, 1)
            },
            "claw_state": "closed" if (self.held_constraint is not None) else "open",
            "held_object_id": self.held_object_id,
            "objects": objects_telemetry
        }

    def cleanup(self):
        """Shuts down the physics client. Call on exit to release resources."""
        if self.physics_client is not None:
            try:
                p.disconnect(physicsClientId=self.physics_client)
            except Exception:
                pass
            self.physics_client = None
            print("[Simulator] Physics client disconnected.")


# ==============================================================================
# 🌐 FASTAPI APPLICATION & UNIFIED WEB HOSTING
# ==============================================================================
_sim_world_instance = None
_sim_world_lock = threading.Lock()


def get_sim_world():
    """Lazy initializer for the simulation world. Prevents PyBullet from
    starting on module import (e.g., during tests or other imports).
    Thread-safe: only one WorldSimulator is ever created."""
    global _sim_world_instance
    if _sim_world_instance is None:
        with _sim_world_lock:
            if _sim_world_instance is None:
                _sim_world_instance = WorldSimulator()
    return _sim_world_instance

def cleanup_sim_world():
    """Explicitly shut down the simulation and release resources."""
    global _sim_world_instance
    if _sim_world_instance is not None:
        _sim_world_instance.cleanup()
        _sim_world_instance = None

app = FastAPI(title="Robotic Arm Hardware Simulator")

# Web hosting: serve static files and the cockpit dashboard
web_dir = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(web_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(web_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Robotic Arm Simulator: web/index.html not found.</h2>")

@app.get("/capture")
def capture_frame():
    jpeg_bytes = get_sim_world().render_esp32_frame()
    return Response(content=jpeg_bytes, media_type="image/jpeg")

@app.get("/capture_depth")
def capture_depth_frame():
    jpeg_bytes = get_sim_world().render_depth_frame()
    return Response(content=jpeg_bytes, media_type="image/jpeg")

@app.get("/telemetry")
def get_telemetry():
    return get_sim_world().get_telemetry()

@app.get("/scenarios")
def list_scenarios():
    return {
        "current": get_sim_world().active_scenario,
        "scenarios": [
            {"id": "sorting", "name": "Standard Sorting", "desc": "Red soda can, blue cube, obstacle cylinder, yellow calibration markers."},
            {"id": "mars_rover", "name": "Mars Rover Curation", "desc": "Martian regolith, hematite sample, olivine crystal, basalt rock."},
            {"id": "chemistry_lab", "name": "Chemistry Lab", "desc": "Clean bench, acid vial, base neutralizer, titration beaker."},
            {"id": "manufacturing_plant", "name": "Electronics Plant", "desc": "ESD mat, microcontroller IC chip, valid/bad capacitor, PCB socket."}
        ]
    }

@app.post("/switch_scenario")
def switch_scenario(payload: dict):
    target = payload.get("scenario", "sorting")
    supported = cfg.get("SUPPORTED_SCENARIOS", ["sorting", "mars_rover", "chemistry_lab", "manufacturing_plant"])
    if target in supported:
        get_sim_world().spawn_scenario(target)
        return {"status": "success", "active_scenario": target}
    return {"status": "error", "message": f"Unsupported scenario '{target}'"}

@app.post("/execute_tool")
def execute_tool_call(payload: dict):
    tool_name = payload.get("tool")
    args = payload.get("arguments", {})
    sw = get_sim_world()

    try:
        if tool_name == "move_arm":
            return sw.execute_move_arm(float(args["x_cm"]), float(args["y_cm"]), float(args["z_cm"]))
        elif tool_name == "set_claw":
            return sw.execute_set_claw(args["state"])
        elif tool_name == "park_arm":
            return sw.execute_park_arm(args.get("pose", "survey"))

        return {"status": "error", "message": f"Unknown tool name '{tool_name}'"}
    except KeyError as e:
        return {"status": "error", "message": f"Missing required argument: {e}"}

@app.post("/api/manual_move")
def manual_move(payload: dict):
    action = payload.get("action")
    sw = get_sim_world()
    if action == "move":
        x = float(payload.get("x", 25.0))
        y = float(payload.get("y", 25.0))
        z = float(payload.get("z", 10.0))
        return sw.execute_move_arm(x, y, z)
    elif action == "claw":
        state = payload.get("state", "open")
        return sw.execute_set_claw(state)
    elif action == "park":
        pose = payload.get("pose", "survey")
        return sw.execute_park_arm(pose)
    return {"status": "error", "message": f"Unknown manual action '{action}'"}

@app.post("/api/scan")
def scan_workspace():
    """Triggers SAM 2.1 segmentation + depth fusion on the current frame."""
    from sam2_processor import SAM2Processor
    sw = get_sim_world()
    processor = SAM2Processor(config=config.sam2_processor_CONFIG)

    raw_path = "handshake/raw_capture.jpg"
    os.makedirs("handshake", exist_ok=True)
    frame_bytes = sw.render_esp32_frame()
    
    if not frame_bytes or len(frame_bytes) < 1000:
        return {"status": "error", "message": "Camera frame capture failed or empty"}
    
    with open(raw_path, "wb") as f:
        f.write(frame_bytes)
    
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 1000:
        return {"status": "error", "message": "Failed to write capture file"}

    depth_map = sw.get_metric_depth_map()
    _, objects = processor.process_image(image_input=raw_path, depth_buffer=depth_map)
    return {"status": "success", "objects": objects}

@app.post("/api/chat")
def autonomy_chat_step(payload: dict):
    """Executes a full perception-reasoning-action cycle from a user natural language command.
    Loops autonomously after each tool call until the LLM returns a text response or max turns."""
    from sam2_processor import SAM2Processor
    from agent_harness import AgentHarness
    command = payload.get("command", "")
    sw = get_sim_world()

    processor = SAM2Processor(config=config.sam2_processor_CONFIG)
    harness = AgentHarness()
    max_turns = config.simulator_CONFIG.get("MAX_AUTO_TURNS", 30)

    raw_path = "handshake/raw_capture.jpg"
    output_image_path = config.sam2_processor_CONFIG.get("OUTPUT_IMAGE", "handshake/annotated_output.jpg")

    step_command = command
    final_response_text = ""
    final_reasoning = ""
    final_tool_call = {"tool": "text_response", "arguments": {}}
    final_execution_feedback = {"status": "info", "message": "No action taken."}
    final_objects = []
    turn_count = 0

    while turn_count < max_turns:
        # Capture frame
        os.makedirs("handshake", exist_ok=True)
        frame_bytes = sw.render_esp32_frame()
        
        if not frame_bytes or len(frame_bytes) < 1000:
            return {"status": "error", "message": "Camera frame capture failed or empty"}
        
        with open(raw_path, "wb") as f:
            f.write(frame_bytes)
        
        if not os.path.exists(raw_path) or os.path.getsize(raw_path) < 1000:
            return {"status": "error", "message": "Failed to write capture file"}

        # Run SAM 2.1 on current frame + depth fusion
        depth_map = sw.get_metric_depth_map()
        _, objects = processor.process_image(image_input=raw_path, depth_buffer=depth_map)
        final_objects = objects

        # Save previous annotated frame for before/after context
        if os.path.exists(output_image_path):
            prev_image_path = os.path.join(os.path.dirname(output_image_path), "previous_annotated_output.jpg")
            try:
                shutil.copy(output_image_path, prev_image_path)
            except Exception:
                pass

        # Agent reasoning with accumulated conversation history
        out = harness.decide_action(
            user_command=step_command,
            image_path=output_image_path,
            detected_objects=objects
        )

        final_response_text = out.get("response_text", "")
        final_reasoning = out.get("reasoning", "")
        final_tool_call = out.get("tool_call", {"tool": "text_response", "arguments": {}})
        final_execution_feedback = out.get("execution_feedback", {"status": "info", "message": "No feedback."})

        tool_name = final_tool_call.get("tool", "text_response")
        tool_args = final_tool_call.get("arguments", {})

        # Execute tool directly on the shared physics world
        if tool_name != "text_response":
            if tool_name == "move_arm":
                sw.execute_move_arm(float(tool_args["x_cm"]), float(tool_args["y_cm"]), float(tool_args["z_cm"]))
            elif tool_name == "set_claw":
                sw.execute_set_claw(tool_args["state"])
            elif tool_name == "park_arm":
                sw.execute_park_arm(tool_args.get("pose", "survey"))

            turn_count += 1
            if turn_count < max_turns:
                step_command = "Inspect the updated camera view and depth sensor readings. Execute the next atomic action to progress or finish the task."
        else:
            # LLM returned text response — task complete or needs user input
            break

    return {
        "response_text": final_response_text,
        "reasoning": final_reasoning,
        "tool_call": final_tool_call,
        "execution_feedback": final_execution_feedback,
        "detected_objects": final_objects,
        "turns_executed": turn_count
    }

@app.post("/api/set_model_preset")
def set_model_preset(payload: dict):
    """Switches the active LLM model preset."""
    from agent_harness import AgentHarness
    preset = payload.get("preset", "")
    harness = AgentHarness(cfg=config.agent_harness_CONFIG, tools=config.ROBOT_TOOLS)
    if harness.switch_model_preset(preset):
        return {"status": "success", "preset": preset, "model": harness.model_name}
    return {"status": "error", "message": f"Unknown preset '{preset}'"}

if __name__ == "__main__":
    import atexit
    # Pre-initialize in main thread to avoid PyBullet GUI thread issues
    get_sim_world()
    print(f"\n[Simulator] Starting Robotic Arm Simulator at http://{cfg['HOST']}:{cfg['PORT']}")
    atexit.register(cleanup_sim_world)
    uvicorn.run(app, host=cfg["HOST"], port=cfg["PORT"], log_level="warning")