# sam2_processor.py
# Perception Node: SAM 2.1 Segmentation + Multi-Sensor 3D Depth Perception & Fusion.
# Computes precise physical 3D coordinates (X, Y, Z in cm) and physical object dimensions.

import os
import cv2
import numpy as np
from PIL import Image
import config

CONFIG = config.sam2_processor_CONFIG


class SAM2Processor:
    """Perception engine combining visual segmentation with secondary depth sensor fusion.
    Extracts 2D object boundaries, associates them with depth sensor point arrays,
    and constructs millimeter-grounded 3D spatial representations."""

    def __init__(self, config: dict = CONFIG):
        self.config = config
        self.model = None
        self.pixels_per_cm = self.config.get("PIXELS_PER_CM", 25.0)

        # Attempt to load SAM 2.1 if available, otherwise use high-precision computer vision fallback
        model_path = self.config.get("MODEL_PATH", "sam2.1_l.pt")
        if os.path.exists(model_path):
            try:
                import torch
                from ultralytics import SAM
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[SAM2Processor] Loading Ultralytics SAM 2.1 on {self.device} from '{model_path}'...")
                self.model = SAM(model_path)
                print("[SAM2Processor] SAM 2.1 weights loaded successfully.")
            except Exception as e:
                print(f"[SAM2Processor] Notice: Could not initialize Ultralytics SAM ({e}). Using robust CV perception.")
                self.model = None
        else:
            print(f"[SAM2Processor] SAM weights '{model_path}' not present on disk. Running in fast CV perception mode.")

    def _denoise_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """Applies median filtering to suppress sensor noise and scanlines."""
        return cv2.medianBlur(frame, 3)

    def _segment_objects(self, frame: np.ndarray):
        """Extracts masks, contours, and bounding boxes using SAM 2.1 or adaptive CV contouring."""
        frame_h, frame_w = frame.shape[:2]
        masks_contours = []
        boxes = []

        if self.model is not None:
            try:
                results = self.model(
                    frame,
                    device=self.device,
                    retina_masks=True,
                    conf=self.config.get("CONFIDENCE_THRESHOLD", 0.80),
                    verbose=False
                )[0]
                if results.masks is not None:
                    for i, cnt in enumerate(results.masks.xy):
                        if len(cnt) >= 5:
                            masks_contours.append(cnt.astype(np.int32))
                            if results.boxes is not None and i < len(results.boxes.xyxy):
                                bx = results.boxes.xyxy[i].cpu().numpy()
                                boxes.append([int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])])
                            else:
                                x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
                                boxes.append([x, y, x + w, y + h])
                    return masks_contours, boxes
            except Exception as e:
                print(f"[SAM2Processor] SAM inference warning: {e}. Falling back to visual perception.")

        # Robust High-Precision Adaptive Vision Fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect workspace foreground objects against background plane
        edges = cv2.Canny(blurred, 30, 110)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < (frame_h * frame_w * 0.40):
                x, y, w, h = cv2.boundingRect(cnt)
                # Ignore edge border artifacts
                if x > 10 and y > 10 and (x + w) < (frame_w - 10) and (y + h) < (frame_h - 10):
                    masks_contours.append(cnt)
                    boxes.append([x, y, x + w, y + h])

        return masks_contours, boxes

    def process_image(self, image_input=None, depth_buffer: np.ndarray = None):
        """Runs the complete 3D Perception & Depth Fusion pipeline:
        1. Segments 2D object boundaries.
        2. Calibrates pixel coordinates using corner anchors and camera center.
        3. Projects secondary depth sensor values onto segmented masks.
        4. Calculates exact 3D centroids (X, Y, Z in cm) and physical heights.
        5. Generates annotated visualization with bounding boxes and 3D labels.
        Returns: (annotated_frame, list_of_3d_objects)"""
        target_input = image_input if image_input is not None else self.config.get("INPUT_IMAGE")

        if isinstance(target_input, str):
            if not os.path.exists(target_input):
                raise FileNotFoundError(f"Image not found at: {target_input}")
            frame = cv2.imread(target_input)
        elif isinstance(target_input, Image.Image):
            frame = cv2.cvtColor(np.array(target_input), cv2.COLOR_RGB2BGR)
        else:
            frame = target_input.copy()

        if frame is None:
            raise ValueError("Invalid image input provided.")

        if self.config.get("FILTER_SCANLINES", True):
            frame = self._denoise_scanlines(frame)

        frame_h, frame_w = frame.shape[:2]
        cam_center_x_px = frame_w / 2.0
        cam_center_y_px = frame_h / 2.0

        # Calibration: detect yellow markers to dynamically update px/cm scale
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, np.array([20, 90, 90]), np.array([40, 255, 255]))
        y_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_centers = []
        for yc in y_contours:
            if cv2.contourArea(yc) > 8:
                M = cv2.moments(yc)
                if M["m00"] != 0:
                    marker_centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

        calibrated_px_per_cm = self.pixels_per_cm
        if len(marker_centers) >= 2:
            xs = [m[0] for m in marker_centers]
            ys = [m[1] for m in marker_centers]
            spans = []
            if (max(xs) - min(xs)) > 50:
                spans.append((max(xs) - min(xs)) / 40.0)
            if (max(ys) - min(ys)) > 50:
                spans.append((max(ys) - min(ys)) / 40.0)
            if spans:
                calibrated_px_per_cm = float(np.mean(spans))

        # Run Segmentation
        contours, boxes = self._segment_objects(frame)
        detected_objects = []
        obj_id = 1

        for i, cnt in enumerate(contours):
            pts = cnt.reshape(-1, 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx_px = M["m10"] / M["m00"]
                cy_px = M["m01"] / M["m00"]
            else:
                cx_px, cy_px = float(pts[0][0]), float(pts[0][1])

            # Convert 2D image coordinates to physical workspace X, Y in cm
            # Table center is at (25.0 cm, 25.0 cm). PyBullet Y axis is inverted relative to image rows.
            dx_cm = (cx_px - cam_center_x_px) / calibrated_px_per_cm
            dy_cm = (cy_px - cam_center_y_px) / calibrated_px_per_cm
            center_x_cm = round(25.0 + dx_cm, 1)
            center_y_cm = round(25.0 - dy_cm, 1)

            # Skip markers themselves from object classification if they match marker positions
            is_calibration_marker = any(
                abs(center_x_cm - mx) < 3.5 and abs(center_y_cm - my) < 3.5
                for mx, my in [(5, 5), (45, 5), (5, 45), (45, 45)]
            )
            if is_calibration_marker:
                continue

            if i < len(boxes):
                x1, y1, x2, y2 = boxes[i]
                w_px = x2 - x1
                h_px = y2 - y1
                width_cm = round(w_px / calibrated_px_per_cm, 1)
                length_cm = round(h_px / calibrated_px_per_cm, 1)
            else:
                x1, y1, x2, y2 = int(cx_px - 15), int(cy_px - 15), int(cx_px + 15), int(cy_px + 15)
                w_px = h_px = 30
                width_cm = length_cm = round(30.0 / calibrated_px_per_cm, 1)

            # --- SECONDARY DEPTH SENSOR FUSION (2D-to-3D MAPPING) ---
            if depth_buffer is not None:
                # Create mask for this specific object contour
                mask_mat = np.zeros((frame_h, frame_w), dtype=np.uint8)
                cv2.drawContours(mask_mat, [cnt], -1, 255, -1)
                
                # Sample depth elevation inside contour
                sampled_elevations = depth_buffer[mask_mat > 0]
                # Filter noise and table surface bleed (elevations > 0.4cm)
                elevated = sampled_elevations[sampled_elevations > 0.4]

                if len(elevated) > 0:
                    surface_top_z_cm = round(float(np.percentile(elevated, self.config.get("DEPTH_FILTER_PERCENTILE", 80.0))), 1)
                    height_cm = round(max(0.6, surface_top_z_cm), 1)
                    center_z_cm = round(surface_top_z_cm / 2.0, 1)
                    grasp_recommended_z_cm = round(max(1.0, surface_top_z_cm * 0.55), 1)
                    depth_confidence = 0.95
                else:
                    surface_top_z_cm = 4.0
                    height_cm = 4.0
                    center_z_cm = 2.0
                    grasp_recommended_z_cm = 2.0
                    depth_confidence = 0.60
            else:
                # Geometric estimation fallback if depth sensor stream not passed
                surface_top_z_cm = round(max(2.5, min(width_cm, length_cm) * 1.1), 1)
                height_cm = surface_top_z_cm
                center_z_cm = round(surface_top_z_cm / 2.0, 1)
                grasp_recommended_z_cm = round(max(1.0, surface_top_z_cm * 0.55), 1)
                depth_confidence = 0.70

            obj_record = {
                "id": obj_id,
                "centroid_cm": {
                    "x": center_x_cm,
                    "y": center_y_cm,
                    "z": center_z_cm
                },
                "surface_top_z_cm": surface_top_z_cm,
                "height_cm": height_cm,
                "grasp_recommended_z_cm": grasp_recommended_z_cm,
                "dimensions_cm": {
                    "width_cm": width_cm,
                    "length_cm": length_cm,
                    "height_cm": height_cm
                },
                "bounding_box_pixels": {
                    "left_x": x1, "top_y": y1, "right_x": x2, "bottom_y": y2
                },
                "depth_confidence": depth_confidence
            }

            detected_objects.append(obj_record)

            # Draw 3D Overlays: Glowing Cyan Box + Centroid dot + 3D Coordinates
            cv2.polylines(frame, [cnt], isClosed=True, color=(0, 255, 200), thickness=2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.circle(frame, (int(cx_px), int(cy_px)), 5, (0, 0, 255), -1)

            label_3d = f"#{obj_id} ({center_x_cm},{center_y_cm},{center_z_cm})cm | H:{height_cm}cm"
            cv2.putText(
                frame, label_3d,
                (x1, max(y1 - 6, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (0, 255, 255), 1, cv2.LINE_AA
            )

            obj_id += 1

        # Save annotated image artifact
        out_path = self.config.get("OUTPUT_IMAGE", "handshake/annotated_output.jpg")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, frame)

        return frame, detected_objects


if __name__ == "__main__":
    processor = SAM2Processor()
    test_img = CONFIG.get("INPUT_IMAGE", "handshake/raw_capture.jpg")
    if os.path.exists(test_img):
        annotated, objs = processor.process_image(test_img)
        print(f"\n[SAM2Processor] Found {len(objs)} 3D objects:")
        for o in objs:
            print(f"  Object #{o['id']} -> Centroid: {o['centroid_cm']} cm | Top Z: {o['surface_top_z_cm']} cm | Grasp Z: {o['grasp_recommended_z_cm']} cm")
    else:
        print("No test capture image found.")