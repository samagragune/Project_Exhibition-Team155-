# sam2_processor.py
# SAM 2.1 segmentation module: loads Ultralytics SAM model, detects objects,
# computes centroids in cm using color-filtered calibration markers, applies
# noise filtering, and renders annotated output.

import os
import cv2
import numpy as np
import torch
from ultralytics import SAM
from PIL import Image

# ==============================================================================
# ⚙️ CONFIGURATION SECTION
# ==============================================================================
import config
CONFIG = config.sam2_processor_CONFIG
# ==============================================================================


class SAM2Processor:
    """SAM 2.1 segmentation pipeline: loads the Ultralytics SAM model, runs
    inference on input frames, applies noise filtering, calibrates object
    centroids to physical cm coordinates using yellow corner markers, and
    renders annotated output with bounding boxes and labels."""

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2Processor] Running inference on device: {self.device}")
        
        # Load Ultralytics SAM 2.1 model
        self.model = SAM(self.config["MODEL_PATH"])
        self.pixels_per_cm = self.config["PIXELS_PER_CM"]

    def _denoise_scanlines(self, frame: np.ndarray) -> np.ndarray:
        """Applies a 3x3 median blur to reduce horizontal scanline artifacts
        caused by ESP32-CAM power noise and compression."""
        return cv2.medianBlur(frame, 3)

    def process_image(self, image_input=None):
        """Runs the full SAM 2.1 pipeline:
        1. Load and optionally denoise the input frame
        2. Run SAM 2.1 segmentation inference
        3. Calibrate pixel centroids to cm using HSV-filtered yellow markers
        4. Filter small/noisy masks by area, circularity, and box size
        5. Annotate frame with bounding boxes, centroids, and labels
        Returns (annotated_frame, list_of_detected_objects)."""
        target_input = image_input if image_input is not None else self.config["INPUT_IMAGE"]

        if isinstance(target_input, str):
            if not os.path.exists(target_input):
                raise FileNotFoundError(f"Input image not found: {target_input}")
            frame = cv2.imread(target_input)
        elif isinstance(target_input, Image.Image):
            frame = cv2.cvtColor(np.array(target_input), cv2.COLOR_RGB2BGR)
        else:
            frame = target_input.copy()

        if frame is None:
            raise ValueError("Invalid image input provided.")

        if self.config["FILTER_SCANLINES"]:
            frame = self._denoise_scanlines(frame)

        results = self.model(
            frame, 
            device=self.device, 
            retina_masks=True, 
            conf=self.config["CONFIDENCE_THRESHOLD"], 
            verbose=False
        )[0]

        detected_objects = []

        if results.masks is not None:
            masks_xy = results.masks.xy
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            valid_indices = list(range(len(masks_xy)))

            # --- FIXED CALIBRATION: Color-Filtered Markers + Center Anchor ---
            calibrated_pixels_per_cm = self.pixels_per_cm
            
            # 1. Use HSV color filtering to strictly isolate ONLY the yellow markers
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv_frame, np.array([20, 100, 100]), np.array([40, 255, 255]))
            yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            marker_centroids = []
            for cnt in yellow_contours:
                if cv2.contourArea(cnt) > 5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        marker_centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))

            # 2. Calculate scale robustly (even if 1-2 dots are covered/missing)
            if len(marker_centroids) >= 2:
                min_x, max_x = min(c[0] for c in marker_centroids), max(c[0] for c in marker_centroids)
                min_y, max_y = min(c[1] for c in marker_centroids), max(c[1] for c in marker_centroids)
                
                scales = []
                if (max_x - min_x) > 50: scales.append((max_x - min_x) / 40.0) # 40cm horizontal span
                if (max_y - min_y) > 50: scales.append((max_y - min_y) / 40.0) # 40cm vertical span
                
                if scales:
                    calibrated_pixels_per_cm = sum(scales) / len(scales)
                    print(f"[Calibration] Detected {len(marker_centroids)} markers. Scale: {calibrated_pixels_per_cm:.2f} px/cm")

            # Build metadata and overlays
            obj_counter = 1
            frame_h, frame_w = frame.shape[0], frame.shape[1]
            total_frame_area = frame_h * frame_w
            min_mask_area_px = self.config.get("MIN_MASK_AREA_PX", 25)

            for i in valid_indices:
                contour = masks_xy[i]
                if len(contour) < 5:
                    continue

                pts_check = contour.astype(np.int32)
                mask_area = cv2.contourArea(pts_check)
                if mask_area <= min_mask_area_px:
                    continue  

                perimeter = cv2.arcLength(pts_check, closed=True)
                if perimeter <= 0:
                    continue

                circularity = (4 * np.pi * mask_area) / (perimeter ** 2)
                if circularity < self.config.get("MIN_MASK_CIRCULARITY", 0.02):
                    continue  

                if i < len(boxes):
                    bx1, by1, bx2, by2 = map(int, boxes[i])
                    box_area = (bx2 - bx1) * (by2 - by1)
                    if box_area > (total_frame_area * 0.50):
                        continue  

                pts = contour.astype(np.int32)
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    center_x_px = M["m10"] / M["m00"]
                    center_y_px = M["m01"] / M["m00"]
                else:
                    center_x_px, center_y_px = pts[0][0], pts[0][1]

                # Anchor origin to the physical camera center (25cm, 25cm)
                # This makes tracking IMMUNE to missing or covered corner markers.
                frame_h, frame_w = frame.shape[:2]
                cam_center_x_px = frame_w / 2.0
                cam_center_y_px = frame_h / 2.0
                
                # Calculate pixel deviation from screen center, convert to cm
                delta_x_cm = (center_x_px - cam_center_x_px) / calibrated_pixels_per_cm
                delta_y_cm = (center_y_px - cam_center_y_px) / calibrated_pixels_per_cm
                
                # Add to the known physical center of the table (25.0 cm, 25.0 cm)
                # INVERT the Y-axis because PyBullet's +Y is "forward/up" but Image +Y is "down"
                center_x_cm = round(25.0 + delta_x_cm, 2)
                center_y_cm = round(25.0 - delta_y_cm, 2)

                if i < len(boxes):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    width_px = x2 - x1
                    height_px = y2 - y1
                    width_cm = round(width_px / calibrated_pixels_per_cm, 2)
                    height_cm = round(height_px / calibrated_pixels_per_cm, 2)
                else:
                    x1 = y1 = x2 = y2 = width_px = height_px = width_cm = height_cm = 0

                obj_data = {
                    "id": obj_counter,
                    "centroid_cm": {"x": center_x_cm, "y": center_y_cm},
                    "centroid_px": {"x": int(center_x_px), "y": int(center_y_px)},
                    "bounding_box_pixels": {
                        "left_x": x1, "top_y": y1, "right_x": x2, "bottom_y": y2,
                        "width_px": width_px, "height_px": height_px
                    },
                    "estimated_size_cm": {
                        "width_cm": width_cm, "height_cm": height_cm
                    }
                }

                detected_objects.append(obj_data)

                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(frame, (int(center_x_px), int(center_y_px)), 4, (0, 0, 255), -1)

                box_label = f"ID:{obj_counter} | ({center_x_cm},{center_y_cm})cm"
                cv2.putText(frame, box_label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

                obj_counter += 1

        cv2.imwrite(self.config["OUTPUT_IMAGE"], frame)
        return frame, detected_objects


# --- Standalone Test Execution ---
if __name__ == "__main__":
    # Initialize using top-level CONFIG
    processor = SAM2Processor(config=CONFIG)
    
    # Fallback check for test script
    input_file = CONFIG["INPUT_IMAGE"]
    if not os.path.exists(input_file):
        # Default to test_image.jpg if configured image isn't available
        input_file = "test_image.jpg"
    
    if os.path.exists(input_file):
        print(f"\n[Testing] Processing image: {input_file}")
        _, objects = processor.process_image(image_input=input_file)
        
        print(f"\n--- SAM 2.1 RESULTS ({len(objects)} objects found) ---")
        for obj in objects:
            print(
                f"Object #{obj['id']} -> Centroid: {obj['centroid_cm']} cm | "
                f"BBox: [L:{obj['bounding_box_pixels']['left_x']}, T:{obj['bounding_box_pixels']['top_y']}, "
                f"R:{obj['bounding_box_pixels']['right_x']}, B:{obj['bounding_box_pixels']['bottom_y']}]"
            )
        
        print(f"\nAnnotated output saved to: '{CONFIG['OUTPUT_IMAGE']}'")
    else:
        print("No input image found to run test.")