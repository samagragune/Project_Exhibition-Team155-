# get_image.py
# HTTP client module that fetches a raw JPEG frame from the simulator
# (or ESP32-CAM in hardware mode) and saves it to the handshake directory.

import os
import requests
import time

# ==============================================================================
# ⚙️ CONFIGURATION SECTION
# ==============================================================================
import config
CONFIG = config.get_image_CONFIG
# ==============================================================================


def capture_and_save_frame(
    ip: str = CONFIG["ESP32_IP"],
    endpoint: str = CONFIG["ENDPOINT"],
    output_filename: str = CONFIG["OUTPUT_FILENAME"],
    save_dir: str = CONFIG["SAVE_DIR"],
    timeout: int = CONFIG["TIMEOUT_SECONDS"],
    retries: int = CONFIG.get("MAX_RETRIES", 3)
) -> str:
    """
    Fetches a raw JPEG frame directly from the ESP32-CAM and saves it locally.
    
    :return: Full string path of the successfully saved image file.
    :raises ConnectionError: If all retry attempts to fetch the image fail.
    """
    url = f"http://{ip}{endpoint}"
    full_save_path = os.path.join(save_dir, output_filename)
    
    for attempt in range(1, retries + 1):
        try:
            print(f"[CameraNode] Requesting image from {url} (Attempt {attempt}/{retries})...")
            response = requests.get(url, timeout=timeout)
            
            # Verify valid HTTP 200 response
            response.raise_for_status()
            
            # Ensure saved directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Write binary JPEG payload directly to disk
            with open(full_save_path, "wb") as f:
                f.write(response.content)
                
            print(f"[CameraNode] Frame saved successfully -> '{full_save_path}'")
            return full_save_path

        except requests.exceptions.RequestException as e:
            print(f"[CameraNode] Warning: Fetch failed on attempt {attempt} ({e})")
            if attempt < retries:
                time.sleep(1) # Brief pause before retrying
            else:
                raise ConnectionError(
                    f"[CameraNode] Failed to reach ESP32-CAM at {url} after {retries} attempts."
                )


# ==============================================================================
# 🚀 STANDALONE TEST EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("--- TESTING CAMERA NODE STANDALONE ---")
    try:
        saved_path = capture_and_save_frame()
        print(f"[Success] Image ready for processing at: {saved_path}")
    except Exception as err:
        print(f"[Error] {err}")