# get_image.py
# HTTP client module: Fetches raw RGB camera frames and secondary depth sensor heatmaps
# from the simulator (or physical ESP32-CAM / depth module in hardware mode).

import os
import requests
import time
import config

CONFIG = config.get_image_CONFIG


def capture_and_save_frame(
    ip: str = CONFIG["ESP32_IP"],
    endpoint: str = CONFIG["ENDPOINT"],
    output_filename: str = CONFIG["OUTPUT_FILENAME"],
    save_dir: str = CONFIG["SAVE_DIR"],
    timeout: int = CONFIG["TIMEOUT_SECONDS"],
    retries: int = CONFIG.get("MAX_RETRIES", 3)
) -> str:
    """Fetches a raw RGB JPEG frame and saves it locally.
    Returns: full path of saved image."""
    url = f"http://{ip}{endpoint}"
    full_save_path = os.path.join(save_dir, output_filename)
    os.makedirs(save_dir, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            with open(full_save_path, "wb") as f:
                f.write(response.content)
            return full_save_path
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(0.5)
            else:
                raise ConnectionError(f"[CameraNode] Failed to reach camera at {url} after {retries} attempts: {e}")


def capture_depth_heatmap(
    ip: str = CONFIG["ESP32_IP"],
    endpoint: str = CONFIG.get("DEPTH_ENDPOINT", "/capture_depth"),
    output_filename: str = "depth_heatmap.jpg",
    save_dir: str = CONFIG["SAVE_DIR"],
    timeout: int = CONFIG["TIMEOUT_SECONDS"]
) -> str:
    """Fetches secondary depth sensor heatmap visualization and saves locally."""
    url = f"http://{ip}{endpoint}"
    full_save_path = os.path.join(save_dir, output_filename)
    os.makedirs(save_dir, exist_ok=True)

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        with open(full_save_path, "wb") as f:
            f.write(response.content)
        return full_save_path
    except Exception as e:
        print(f"[DepthNode] Warning: Failed to fetch depth heatmap: {e}")
        return ""


if __name__ == "__main__":
    print("--- TESTING CAMERA & DEPTH NODES ---")
    try:
        rgb_path = capture_and_save_frame()
        depth_path = capture_depth_heatmap()
        print(f"RGB Frame Saved   : {rgb_path}")
        print(f"Depth Heatmap Saved: {depth_path}")
    except Exception as err:
        print(f"Error: {err}")