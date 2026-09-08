# simulator.py
# PyBullet physics simulation + FastAPI HTTP server.
# Replaces the physical ESP32-CAM + robotic arm hardware backend.
# Endpoints: GET /capture (JPEG), POST /execute_tool (tool call dispatch)

import os
import time
import math
import numpy as np
import cv2
import subprocess
import pybullet as p
import pybullet_data

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

from fastapi import FastAPI, Response
import uvicorn
import config

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
        home_joint_positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        for idx, joint_idx in enumerate(self.arm_joints):
            if idx < len(home_joint_positions):
                p.resetJointState(self.arm_id, joint_idx, home_joint_positions[idx], physicsClientId=self.physics_client)

        print(f"[Simulator] UR5 ready. Arm Joints: {self.arm_joints} | Gripper Joints: {self.gripper_joint_indices}")

        # Spawn workspace items and calibration markers
        self._spawn_default_objects()
        p.setRealTimeSimulation(0, physicsClientId=self.physics_client)

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

    def render_esp32_frame(self, resolution=None) -> bytes:
        """Renders a top-down JPEG frame from the PyBullet camera.
        Optionally applies ESP32 scanline noise artifacts."""
        res_w, res_h = resolution if resolution else cfg.get("CAMERA_RESOLUTION", (640, 480))

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.25, 0.25, 0.80],
            cameraTargetPosition=[0.25, 0.25, 0.0],
            cameraUpVector=[0, 1, 0]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=cfg.get("CAMERA_FOV", 60.0),
            aspect=float(res_w) / res_h,
            nearVal=0.1,
            farVal=2.0
        )

        _, _, rgb_pixels, _, _ = p.getCameraImage(
            width=res_w,
            height=res_h,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        frame_rgba = np.reshape(rgb_pixels, (res_h, res_w, 4)).astype(np.uint8)
        frame_rgb = frame_rgba[:, :, :3]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if cfg.get("ESP32_NOISE", False):
            scanlines = np.ones_like(frame_bgr, dtype=np.float32)
            scanlines[::4, :, :] = 0.90
            frame_bgr = (frame_bgr.astype(np.float32) * scanlines).clip(0, 255).astype(np.uint8)

        _, enc_img = cv2.imencode('.jpg', frame_bgr)
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
        gripper_length_offset = 0.12  # 12 cm tool length offset
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


sim_world = WorldSimulator()

@app.get("/capture")
def capture_frame():
    jpeg_bytes = sim_world.render_esp32_frame()
    return Response(content=jpeg_bytes, media_type="image/jpeg")

@app.post("/execute_tool")
def execute_tool_call(payload: dict):
    tool_name = payload.get("tool")
    args = payload.get("arguments", {})

    try:
        if tool_name == "move_arm":
            result = sim_world.execute_move_arm(args["x_cm"], args["y_cm"], args["z_cm"])
            return result
        elif tool_name == "set_claw":
            result = sim_world.execute_set_claw(args["state"])
            return result

        return {"status": "error", "message": f"Unknown tool name '{tool_name}'"}
    except KeyError as e:
        return {"status": "error", "message": f"Missing required argument: {e}"}

if __name__ == "__main__":
    print(f"\n[Simulator] Starting Desktop Arm Simulator at http://{cfg['HOST']}:{cfg['PORT']}")
    uvicorn.run(app, host=cfg["HOST"], port=cfg["PORT"], log_level="warning")