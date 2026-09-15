# simulator.py
# PyBullet physics simulation + Secondary Depth Sensor + Multi-Scenario Worlds + FastAPI Server.
# Serves:
#   GET  /capture          (Top-down Camera RGB JPEG)
#   GET  /capture_depth    (Secondary Depth Sensor Heatmap JPEG)
#   GET  /telemetry        (End-effector XYZ cm, joint states, active scenario, held objects)
#   GET  /scenarios        (List supported cognitive demonstration scenarios)
#   POST /switch_scenario  (Live scenario environment switcher)
#   POST /execute_tool     (Dispatches move_arm, set_claw, park_arm)
#   POST /api/manual_move  (Web UI direct jog control)

import os
import time
import math
import numpy as np
import cv2
import subprocess
import pybullet as p
import pybullet_data
from fastapi import FastAPI, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import config

cfg = config.simulator_CONFIG

def ensure_ur5_downloaded():
    """Ensures the UR5 PyBullet model repository is available locally."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    ur5_path = os.path.join(models_dir, "ur5_pybullet")
    urdf_file = os.path.join(ur5_path, "urdf", "ur5_robotiq_140.urdf")
    if not os.path.exists(urdf_file):
        print("[Simulator] Downloading PyBullet UR5 model repository...")
        os.makedirs(models_dir, exist_ok=True)
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/Alchemist77/pybullet-ur5-equipped-with-robotiq-140.git",
            ur5_path
        ], check=True)
        print("[Simulator] Download complete.")
    
    return urdf_file


class WorldSimulator:
    """PyBullet physics world with UR5 arm, Robotiq 140 gripper, calibrated camera frustum,
    secondary depth sensor simulation, and dynamic multi-scenario environments."""

    def __init__(self):
        # 1. Connect GUI Physics Engine (or DIRECT if headless)
        try:
            self.physics_client = p.connect(p.GUI if cfg.get("GUI_ENABLED", True) else p.DIRECT)
        except Exception:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setPhysicsEngineParameter(numSolverIterations=3000, physicsClientId=self.physics_client)

        # Frustum & visualizer setup
        self.camera_eye = cfg.get("CAMERA_EYE_POSITION", [0.25, 0.25, 1.05])
        self.camera_target = cfg.get("CAMERA_TARGET_POSITION", [0.25, 0.25, 0.0])
        self.camera_fov = cfg.get("CAMERA_FOV", 65.0)
        self.camera_near = cfg.get("CAMERA_NEAR", 0.1)
        self.camera_far = cfg.get("CAMERA_FAR", 2.0)
        self.arm_scale = cfg.get("ARM_SCALE", 0.58)
        self.arm_base = cfg.get("ARM_BASE_POSITION", [-0.20, 0.25, -0.04])

        p.resetDebugVisualizerCamera(
            cameraDistance=1.3,
            cameraYaw=0,
            cameraPitch=-89,
            cameraTargetPosition=self.camera_target
        )

        # Tracking state
        self.object_ids = {}
        self.plane_id = None
        self.accessory_ids = []
        self.held_constraint = None
        self.held_object_id = None
        self.active_scenario = cfg.get("SCENE_PRESET", "sorting")
        self.last_target_pos = [0.25, 0.25, 0.15]

        # 2. Load UR5 Arm with calibrated compact scaling & base offset
        ur5_urdf_file = ensure_ur5_downloaded()
        self.arm_id = p.loadURDF(
            ur5_urdf_file,
            basePosition=self.arm_base,
            useFixedBase=True,
            globalScaling=self.arm_scale,
            physicsClientId=self.physics_client
        )

        # 3. Discover Controllable Joints & End Effector
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

            if any(k in joint_name or k in link_name for k in ["finger", "gripper", "knuckle", "claw"]):
                self.gripper_joint_indices.append(i)

            if "ee_link" in link_name or "tool0" in link_name:
                self.end_effector_idx = i

        if self.end_effector_idx == -1:
            self.end_effector_idx = self.arm_joints[-1] if self.arm_joints else 6

        if not self.gripper_joint_indices:
            self.gripper_joint_indices = [j for j in self.movable_joint_indices if j not in self.arm_joints]

        # Home folded position
        self.home_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        for idx, joint_idx in enumerate(self.arm_joints):
            if idx < len(self.home_joint_positions):
                p.resetJointState(self.arm_id, joint_idx, self.home_joint_positions[idx], physicsClientId=self.physics_client)

        # 4. Spawn Active Scenario
        self.spawn_scenario(self.active_scenario)
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)
        print(f"[Simulator] Initialized. Active Scenario: '{self.active_scenario}', Arm Scale: {self.arm_scale}")

    def _clear_workspace_objects(self):
        """Removes all dynamic scenario objects, constraints, and custom planes."""
        if getattr(self, "held_constraint", None) is not None:
            try:
                p.removeConstraint(self.held_constraint, physicsClientId=self.physics_client)
            except Exception:
                pass
            self.held_constraint = None
            self.held_object_id = None

        for name, body_id in list(self.object_ids.items()):
            try:
                p.removeBody(body_id, physicsClientId=self.physics_client)
            except Exception:
                pass
        self.object_ids.clear()

        for acc_id in self.accessory_ids:
            try:
                p.removeBody(acc_id, physicsClientId=self.physics_client)
            except Exception:
                pass
        self.accessory_ids.clear()

        if self.plane_id is not None:
            try:
                p.removeBody(self.plane_id, physicsClientId=self.physics_client)
            except Exception:
                pass
            self.plane_id = None

    def _spawn_calibration_markers(self):
        """Spawns standard 4 corner calibration markers at (5,5), (45,5), (5,45), (45,45) cm."""
        marker_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.012)
        corner_positions = [
            [0.05, 0.05, 0.008],
            [0.45, 0.05, 0.008],
            [0.05, 0.45, 0.008],
            [0.45, 0.45, 0.008]
        ]
        for idx, pos in enumerate(corner_positions):
            marker_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[1.0, 0.95, 0.0, 1.0])
            m_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=marker_shape,
                baseVisualShapeIndex=marker_visual,
                basePosition=pos,
                physicsClientId=self.physics_client
            )
            self.object_ids[f"calibration_marker_{idx}"] = m_id

    def spawn_scenario(self, scenario_name: str):
        """Spawns an environment scenario showcasing general cognitive reasoning over VLAs:
        - 'sorting': Standard workspace with red soda can, blue cube, obstacle cylinder.
        - 'mars_rover': Martian terrain with target hematite mineral rock, olivine, basalt rock, sample carousel slots.
        - 'chemistry_lab': Lab bench with acid reagent, base reagent, reaction beaker, hazard neutralizer.
        - 'manufacturing_plant': Anti-static ESD mat with microcontroller IC chip, good capacitor, defective capacitor, PCB socket."""
        self._clear_workspace_objects()
        self.active_scenario = scenario_name

        if scenario_name == "mars_rover":
            # Martian red regolith ground
            plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01])
            plane_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01], rgbaColor=[0.75, 0.35, 0.20, 1.0])
            self.plane_id = p.createMultiBody(0, plane_shape, plane_visual, [0.25, 0.25, -0.01], physicsClientId=self.physics_client)

            # Target 1: Hematite Ore Sample (Dark Red-Bronze Specimen)
            hem_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.028, height=0.055)
            hem_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.028, length=0.055, rgbaColor=[0.6, 0.15, 0.15, 1.0])
            self.object_ids["hematite_target_sample"] = p.createMultiBody(
                0.2, hem_shape, hem_vis, [0.22, 0.22, 0.028], physicsClientId=self.physics_client
            )

            # Target 2: Olivine Crystalline Mineral (Olive Green Block)
            oli_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.022])
            oli_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.022, 0.022, 0.022], rgbaColor=[0.3, 0.65, 0.25, 1.0])
            self.object_ids["olivine_mineral"] = p.createMultiBody(
                0.15, oli_shape, oli_vis, [0.32, 0.16, 0.022], physicsClientId=self.physics_client
            )

            # Target 3: Basalt Waste Rock (Dark Gray Asteroid/Basalt Rock)
            bas_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.028, 0.028, 0.035])
            bas_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.028, 0.028, 0.035], rgbaColor=[0.25, 0.25, 0.28, 1.0])
            self.object_ids["basalt_waste_rock"] = p.createMultiBody(
                0.3, bas_shape, bas_vis, [0.16, 0.34, 0.035], physicsClientId=self.physics_client
            )

            # Rover Carousel Sample Container Bin 1 (Gold hermetic chamber)
            bin1_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.02])
            bin1_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.02], rgbaColor=[0.85, 0.70, 0.15, 1.0])
            self.object_ids["rover_carousel_bin_1"] = p.createMultiBody(
                0, bin1_shape, bin1_vis, [0.40, 0.38, 0.02], physicsClientId=self.physics_client
            )

            # Rover Carousel Sample Container Bin 2 (Cyan spectrometer slot)
            bin2_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.035, height=0.04)
            bin2_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.04, rgbaColor=[0.1, 0.6, 0.8, 1.0])
            self.object_ids["rover_spectrometer_slot"] = p.createMultiBody(
                0, bin2_shape, bin2_vis, [0.10, 0.38, 0.02], physicsClientId=self.physics_client
            )

        elif scenario_name == "chemistry_lab":
            # Sterile White Lab Bench
            plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01])
            plane_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01], rgbaColor=[0.95, 0.95, 0.97, 1.0])
            self.plane_id = p.createMultiBody(0, plane_shape, plane_visual, [0.25, 0.25, -0.01], physicsClientId=self.physics_client)

            # Reagent Vial A (Concentrated Acid - Amber Cylinder)
            acid_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.022, height=0.075)
            acid_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.075, rgbaColor=[0.85, 0.45, 0.1, 1.0])
            self.object_ids["acid_reagent_vial"] = p.createMultiBody(
                0.15, acid_shape, acid_vis, [0.18, 0.18, 0.038], physicsClientId=self.physics_client
            )

            # Reagent Vial B (Neutralizer Base - Royal Blue Cylinder)
            base_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.022, height=0.075)
            base_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.075, rgbaColor=[0.15, 0.35, 0.95, 1.0])
            self.object_ids["base_reagent_vial"] = p.createMultiBody(
                0.15, base_shape, base_vis, [0.32, 0.18, 0.038], physicsClientId=self.physics_client
            )

            # Reaction Titration Beaker (Large Transparent Cyan Chamber)
            bk_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.045, height=0.07)
            bk_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.045, length=0.07, rgbaColor=[0.2, 0.8, 0.8, 0.75])
            self.object_ids["titration_reaction_beaker"] = p.createMultiBody(
                0, bk_shape, bk_vis, [0.25, 0.33, 0.035], physicsClientId=self.physics_client
            )

            # Hazardous Waste Neutralizer Basin (Red Disposal Tray)
            tray_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.04, 0.015])
            tray_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.04, 0.015], rgbaColor=[0.85, 0.15, 0.15, 1.0])
            self.object_ids["hazardous_waste_basin"] = p.createMultiBody(
                0, tray_shape, tray_vis, [0.40, 0.36, 0.015], physicsClientId=self.physics_client
            )

        elif scenario_name == "manufacturing_plant":
            # ESD Anti-Static Dark Green Mat
            plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01])
            plane_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01], rgbaColor=[0.18, 0.32, 0.24, 1.0])
            self.plane_id = p.createMultiBody(0, plane_shape, plane_visual, [0.25, 0.25, -0.01], physicsClientId=self.physics_client)

            # Microcontroller IC Chip (QFN square flat box with gold pin-1 mark)
            ic_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.01])
            ic_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.01], rgbaColor=[0.12, 0.12, 0.12, 1.0])
            self.object_ids["microcontroller_ic_chip"] = p.createMultiBody(
                0.08, ic_shape, ic_vis, [0.20, 0.20, 0.01], physicsClientId=self.physics_client
            )

            # Valid Electrolytic Capacitor (Silver/Blue tall cylinder)
            cap_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.018, height=0.065)
            cap_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.018, length=0.065, rgbaColor=[0.2, 0.45, 0.85, 1.0])
            self.object_ids["valid_capacitor"] = p.createMultiBody(
                0.1, cap_shape, cap_vis, [0.32, 0.16, 0.033], physicsClientId=self.physics_client
            )

            # Defective Capacitor (Bulged / Defective Orange Component)
            def_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.022, height=0.065)
            def_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.022, length=0.065, rgbaColor=[0.9, 0.3, 0.1, 1.0])
            self.object_ids["defective_capacitor"] = p.createMultiBody(
                0.1, def_shape, def_vis, [0.18, 0.32, 0.033], physicsClientId=self.physics_client
            )

            # Motherboard PCB Socket Receptor
            sock_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.035, 0.035, 0.01])
            sock_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.035, 0.035, 0.01], rgbaColor=[0.1, 0.5, 0.3, 1.0])
            self.object_ids["pcb_target_socket"] = p.createMultiBody(
                0, sock_shape, sock_vis, [0.36, 0.34, 0.01], physicsClientId=self.physics_client
            )

            # QA Defect Discard Chute (Yellow container)
            chute_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.045, 0.04, 0.025])
            chute_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.045, 0.04, 0.025], rgbaColor=[0.95, 0.85, 0.1, 1.0])
            self.object_ids["qa_reject_chute"] = p.createMultiBody(
                0, chute_shape, chute_vis, [0.10, 0.16, 0.025], physicsClientId=self.physics_client
            )

        else:
            # Default "sorting" workspace
            plane_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01])
            plane_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.01], rgbaColor=[0.85, 0.85, 0.87, 1.0])
            self.plane_id = p.createMultiBody(0, plane_shape, plane_visual, [0.25, 0.25, -0.01], physicsClientId=self.physics_client)

            # 1. Red Soda Can
            can_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.08)
            can_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.08, rgbaColor=[0.9, 0.1, 0.1, 1.0])
            self.object_ids["red_soda_can"] = p.createMultiBody(
                0.2, can_shape, can_visual, [0.20, 0.20, 0.04], physicsClientId=self.physics_client
            )

            # 2. Blue Cube
            blue_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025])
            blue_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.025, 0.025, 0.025], rgbaColor=[0.1, 0.35, 0.9, 1.0])
            self.object_ids["blue_cube"] = p.createMultiBody(
                0.1, blue_shape, blue_visual, [0.32, 0.15, 0.025], physicsClientId=self.physics_client
            )

            # 3. Green Obstacle Cylinder
            green_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.07)
            green_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.07, rgbaColor=[0.1, 0.75, 0.2, 1.0])
            self.object_ids["green_obstacle"] = p.createMultiBody(
                0.15, green_shape, green_vis, [0.26, 0.30, 0.035], physicsClientId=self.physics_client
            )

        # Spawn Yellow Calibration Markers
        self._spawn_calibration_markers()

        # Settle simulation physics
        for _ in range(35):
            p.stepSimulation(physicsClientId=self.physics_client)

        print(f"[Simulator] Scenario '{scenario_name}' loaded successfully with {len(self.object_ids)} items.")

    def _get_camera_matrices(self, resolution=(640, 480)):
        """Calculates PyBullet view and projection matrices using configured frustum."""
        res_w, res_h = resolution
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_eye,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=cfg.get("CAMERA_UP_VECTOR", [0, 1, 0])
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=float(res_w) / float(res_h),
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        return view_matrix, proj_matrix

    def render_esp32_frame(self, resolution=None) -> bytes:
        """Renders top-down RGB frame from PyBullet camera."""
        res = resolution if resolution else cfg.get("CAMERA_RESOLUTION", (640, 480))
        res_w, res_h = res
        view_matrix, proj_matrix = self._get_camera_matrices(res)

        _, _, rgb_pixels, _, _ = p.getCameraImage(
            width=res_w,
            height=res_h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        frame_rgba = np.reshape(rgb_pixels, (res_h, res_w, 4)).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

        if cfg.get("ESP32_NOISE", False):
            scanlines = np.ones_like(frame_bgr, dtype=np.float32)
            scanlines[::4, :, :] = 0.90
            frame_bgr = (frame_bgr.astype(np.float32) * scanlines).clip(0, 255).astype(np.uint8)

        _, enc_img = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return enc_img.tobytes()

    def get_metric_depth_map(self, resolution=None) -> np.ndarray:
        """Calculates metric elevation above the table surface in centimeters for each pixel.
        Uses the OpenGL depth buffer and pinhole projection."""
        res = resolution if resolution else cfg.get("CAMERA_RESOLUTION", (640, 480))
        res_w, res_h = res
        view_matrix, proj_matrix = self._get_camera_matrices(res)

        _, _, _, depth_pixels, _ = p.getCameraImage(
            width=res_w,
            height=res_h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        depth_buf = np.reshape(depth_pixels, (res_h, res_w)).astype(np.float32)
        # Convert non-linear OpenGL NDC depth buffer [0, 1] to true line-of-sight distance (meters)
        near_val = self.camera_near
        far_val = self.camera_far
        depth_m = far_val * near_val / (far_val - (far_val - near_val) * depth_buf)

        # Camera is at height camera_eye[2] looking straight down at z=0 (table plane)
        eye_z = self.camera_eye[2]
        elevation_m = np.maximum(0.0, eye_z - depth_m)
        elevation_cm = elevation_m * 100.0
        return elevation_cm

    def render_depth_frame(self, resolution=None) -> bytes:
        """Renders a colorized heatmap visualization of the secondary depth sensor."""
        elevation_cm = self.get_metric_depth_map(resolution)
        
        # Max expected workspace height is ~15cm for visualization scaling
        norm_depth = np.clip((elevation_cm / 12.0) * 255.0, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_depth, cv2.COLORMAP_TURBO)

        # Overlay HUD metrics
        max_h = round(float(np.max(elevation_cm)), 1)
        mean_h = round(float(np.mean(elevation_cm[elevation_cm > 0.5])), 1) if np.any(elevation_cm > 0.5) else 0.0

        cv2.putText(heatmap, f"SECONDARY DEPTH SENSOR: ACTIVE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(heatmap, f"Peak Elevation: {max_h} cm | Avg Object: {mean_h} cm", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        _, enc_img = cv2.imencode(".jpg", heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return enc_img.tobytes()

    def execute_park_arm(self, pose: str = "survey") -> dict:
        """Parks the robotic arm out of the camera's field of view or into rest pose."""
        pose_str = str(pose).strip().lower()
        print(f"\n[Tool Call Received] Parking Arm -> Pose: '{pose_str.upper()}'")

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
            "reached_position_cm": [round(c * 100.0, 1) for c in current_pos]
        }

    def execute_move_arm(self, x_cm: float, y_cm: float, z_cm: float) -> dict:
        """Solves inverse kinematics to move the end-effector to target (X, Y, Z) in cm.
        Applies calibrated tool-offset scaling for the compact arm."""
        target_pos = [x_cm / 100.0, y_cm / 100.0, z_cm / 100.0]
        self.last_target_pos = target_pos
        print(f"\n[Tool Call Received] Moving End-Effector to (cm): X={x_cm}, Y={y_cm}, Z={z_cm}")

        downward_orientation = p.getQuaternionFromEuler([np.pi, 0, 0])

        # Scaled tool length offset: 12cm at scale 0.75 -> scaled proportionally
        calibrated_tool_offset = 0.12 * (self.arm_scale / 0.75)
        adjusted_target_pos = [
            target_pos[0] - calibrated_tool_offset,
            target_pos[1],
            target_pos[2]
        ]

        # Calculate joint limits for smooth IK
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
        for j_idx in self.movable_joint_indices:
            info = p.getJointInfo(self.arm_id, j_idx, physicsClientId=self.physics_client)
            ll, ul = info[8], info[9]
            if ll > ul:
                ll, ul = -2 * math.pi, 2 * math.pi
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(ul - ll)
            rest_poses.append((ll + ul) / 2.0)

        joint_poses = p.calculateInverseKinematics(
            self.arm_id,
            self.end_effector_idx,
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

        for _ in range(180):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1.0 / 240.0)

        current_pos = p.getLinkState(self.arm_id, self.end_effector_idx, physicsClientId=self.physics_client)[0]
        cur_cm = [round(c * 100.0, 1) for c in current_pos]
        print(f"[Simulator] Motion Complete. Current Tip Position: X={cur_cm[0]}cm, Y={cur_cm[1]}cm, Z={cur_cm[2]}cm\n")
        return {
            "status": "success",
            "reached_position_cm": cur_cm,
            "target_cm": [x_cm, y_cm, z_cm]
        }

    def execute_set_claw(self, state: str) -> dict:
        """Opens or closes the Robotiq 140 gripper with physics attachment."""
        state_str = str(state).strip().lower()
        is_closing = state_str in ["close", "closed"]
        print(f"\n[Tool Call Received] Setting Claw: '{state_str.upper()}'")

        # Color indicator: Red=Closed, Green=Open
        indicator_color = [1.0, 0.0, 0.0, 1.0] if is_closing else [0.0, 1.0, 0.0, 1.0]
        p.changeVisualShape(self.arm_id, self.end_effector_idx, rgbaColor=indicator_color, physicsClientId=self.physics_client)

        if not is_closing:
            if getattr(self, "held_constraint", None) is not None:
                try:
                    p.removeConstraint(self.held_constraint, physicsClientId=self.physics_client)
                except Exception:
                    pass
                self.held_constraint = None
                self.held_object_id = None
                print("[Simulator] Claw Opened: Constraint released.")

        target_finger_pos = 0.65 if is_closing else 0.0
        driver_gripper_joints = []
        for g_idx in self.gripper_joint_indices:
            info = p.getJointInfo(self.arm_id, g_idx, physicsClientId=self.physics_client)
            j_name = info[1].decode("utf-8").lower()
            if "knuckle" in j_name or "finger_joint" in j_name:
                driver_gripper_joints.append(g_idx)

        if not driver_gripper_joints:
            driver_gripper_joints = self.gripper_joint_indices[:2]

        for g_idx in driver_gripper_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_id,
                jointIndex=g_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_finger_pos,
                force=250,
                maxVelocity=1.5,
                physicsClientId=self.physics_client
            )

        for _ in range(120):
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1.0 / 240.0)

        attached_obj_name = None
        if is_closing and getattr(self, "held_constraint", None) is None:
            ee_state = p.getLinkState(self.arm_id, self.end_effector_idx, physicsClientId=self.physics_client)
            ee_pos = ee_state[0]

            closest_dist = 999.0
            closest_obj = None

            for obj_name, obj_id in self.object_ids.items():
                if "calibration" in obj_name or "basin" in obj_name or "socket" in obj_name or "slot" in obj_name or "chute" in obj_name or "bin" in obj_name:
                    continue
                obj_pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client)
                dist = sum((e - o) ** 2 for e, o in zip(ee_pos, obj_pos)) ** 0.5

                if dist < 0.16 and dist < closest_dist:
                    closest_dist = dist
                    closest_obj = (obj_name, obj_id)

            if closest_obj:
                obj_name, obj_id = closest_obj
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
                attached_obj_name = obj_name
                print(f"[Simulator] Claw Closed: Attached '{obj_name}' (distance: {round(closest_dist*100, 1)} cm).")

        return {
            "status": "success",
            "message": f"Claw set to {state_str}",
            "held_object": attached_obj_name or (self.held_object_id is not None)
        }

    def get_telemetry(self) -> dict:
        """Returns live system telemetry for dashboard HUD."""
        ee_pos = p.getLinkState(self.arm_id, self.end_effector_idx, physicsClientId=self.physics_client)[0]
        
        objects_telemetry = []
        for name, obj_id in self.object_ids.items():
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.physics_client)
            objects_telemetry.append({
                "name": name,
                "x_cm": round(pos[0] * 100.0, 1),
                "y_cm": round(pos[1] * 100.0, 1),
                "z_cm": round(pos[2] * 100.0, 1)
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
            "arm_scale": self.arm_scale,
            "camera_elevation_m": self.camera_eye[2],
            "objects": objects_telemetry
        }


# ==============================================================================
# 🌐 FASTAPI APPLICATION & UNIFIED WEB HOSTING
# ==============================================================================
sim_world = WorldSimulator()
app = FastAPI(title="Robotic Arm Autonomy Platform")

# Ensure web directory exists
web_dir = os.path.join(os.path.dirname(__file__), "web")
os.makedirs(web_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = os.path.join(web_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Robotic Arm Autonomy Platform: web/index.html not found yet.</h2>")

@app.get("/capture")
def capture_frame():
    jpeg_bytes = sim_world.render_esp32_frame()
    return Response(content=jpeg_bytes, media_type="image/jpeg")

@app.get("/capture_depth")
def capture_depth_frame():
    jpeg_bytes = sim_world.render_depth_frame()
    return Response(content=jpeg_bytes, media_type="image/jpeg")

@app.get("/telemetry")
def get_telemetry():
    return sim_world.get_telemetry()

@app.get("/scenarios")
def list_scenarios():
    return {
        "current": sim_world.active_scenario,
        "scenarios": [
            {
                "id": "sorting",
                "name": "Standard Workspace Sorting",
                "badge": "General",
                "desc": "Red soda can, blue cube, obstacle cylinder, yellow calibration markers."
            },
            {
                "id": "mars_rover",
                "name": "Mars Rover Sample Curation",
                "badge": "Astrobiology",
                "desc": "Martian regolith, hematite mineral targets, olivine, basalt rock, rover sample carousel."
            },
            {
                "id": "chemistry_lab",
                "name": "Chemical & Bio-Synthesis Lab",
                "badge": "Laboratory",
                "desc": "Sterile bench, acid reagent vial, base neutralizer, titration beaker, hazard basin."
            },
            {
                "id": "manufacturing_plant",
                "name": "High-Precision Electronics Assembly",
                "badge": "Industrial",
                "desc": "ESD mat, microcontroller IC chip, valid capacitor, defective bulged capacitor, PCB socket."
            }
        ]
    }

@app.post("/switch_scenario")
def switch_scenario(payload: dict):
    target = payload.get("scenario", "sorting")
    if target in cfg.get("SUPPORTED_SCENARIOS", ["sorting", "mars_rover", "chemistry_lab", "manufacturing_plant"]):
        sim_world.spawn_scenario(target)
        return {"status": "success", "active_scenario": target}
    return {"status": "error", "message": f"Unsupported scenario '{target}'"}

@app.post("/execute_tool")
def execute_tool_call(payload: dict):
    tool_name = payload.get("tool")
    args = payload.get("arguments", {})

    try:
        if tool_name == "move_arm":
            return sim_world.execute_move_arm(float(args["x_cm"]), float(args["y_cm"]), float(args["z_cm"]))
        elif tool_name == "set_claw":
            return sim_world.execute_set_claw(args["state"])
        elif tool_name == "park_arm":
            return sim_world.execute_park_arm(args.get("pose", "survey"))

        return {"status": "error", "message": f"Unknown tool name '{tool_name}'"}
    except KeyError as e:
        return {"status": "error", "message": f"Missing required argument: {e}"}

@app.post("/api/manual_move")
def manual_move(payload: dict):
    action = payload.get("action")
    if action == "move":
        x = float(payload.get("x", 25.0))
        y = float(payload.get("y", 25.0))
        z = float(payload.get("z", 10.0))
        return sim_world.execute_move_arm(x, y, z)
    elif action == "claw":
        state = payload.get("state", "open")
        return sim_world.execute_set_claw(state)
    elif action == "park":
        pose = payload.get("pose", "survey")
        return sim_world.execute_park_arm(pose)
    return {"status": "error", "message": f"Unknown manual action '{action}'"}

@app.post("/api/scan")
def scan_workspace():
    """Triggers 3D Perception & Depth Fusion on current frame."""
    from sam2_processor import SAM2Processor
    processor = SAM2Processor()
    
    # Save fresh capture
    raw_path = "handshake/raw_capture.jpg"
    os.makedirs("handshake", exist_ok=True)
    frame_bytes = sim_world.render_esp32_frame()
    with open(raw_path, "wb") as f:
        f.write(frame_bytes)
        
    depth_map = sim_world.get_metric_depth_map()
    _, objects = processor.process_image(image_input=raw_path, depth_buffer=depth_map)
    return {"status": "success", "objects": objects}

@app.post("/api/chat")
def autonomy_chat_step(payload: dict):
    """Executes a complete perception-reasoning-action cycle from user natural language command."""
    command = payload.get("command", "")
    from sam2_processor import SAM2Processor
    from agent_harness import AgentHarness
    
    processor = SAM2Processor()
    harness = AgentHarness()
    
    # 1. Capture & 3D Depth Fusion
    raw_path = "handshake/raw_capture.jpg"
    os.makedirs("handshake", exist_ok=True)
    frame_bytes = sim_world.render_esp32_frame()
    with open(raw_path, "wb") as f:
        f.write(frame_bytes)
        
    depth_map = sim_world.get_metric_depth_map()
    _, objects = processor.process_image(image_input=raw_path, depth_buffer=depth_map)
    
    # 2. Cognitive Agent Decision
    out = harness.decide_action(
        user_command=command,
        image_path=config.sam2_processor_CONFIG.get("OUTPUT_IMAGE", "handshake/annotated_output.jpg"),
        detected_objects=objects
    )
    
    # 3. Kinematic Dispatch
    tool = out.get("tool_call", {}).get("tool")
    args = out.get("tool_call", {}).get("arguments", {})
    exec_result = {}
    
    if tool == "move_arm":
        exec_result = sim_world.execute_move_arm(float(args["x_cm"]), float(args["y_cm"]), float(args["z_cm"]))
    elif tool == "set_claw":
        exec_result = sim_world.execute_set_claw(args["state"])
    elif tool == "park_arm":
        exec_result = sim_world.execute_park_arm(args.get("pose", "survey"))
        
    out["execution_result"] = exec_result
    out["detected_objects"] = objects
    return out

if __name__ == "__main__":
    print(f"\n[Simulator] Launching Robotic Arm Autonomy Server at http://{cfg['HOST']}:{cfg['PORT']}")
    uvicorn.run(app, host=cfg["HOST"], port=cfg["PORT"], log_level="warning")