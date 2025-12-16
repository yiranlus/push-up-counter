import cv2
import time
import os

# --- Configuration ---
TARGET_FPS = 10.0  # Desired frames per second
CAPTURE_DURATION = 5.0  # Total duration to capture in seconds
CAMERA_INDEX = 0  # Usually 0 for the default webcam
OUTPUT_DIR = "captured_photos"

# --- Calculations ---
T_FRAME = 1.0 / TARGET_FPS  # Target time (in seconds) between two frames
NUM_FRAMES = int(TARGET_FPS * CAPTURE_DURATION)
print(f"Targeting {NUM_FRAMES} frames at {TARGET_FPS:.2f} FPS over {CAPTURE_DURATION:.1f} seconds.")

# --- Setup ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set initial properties (may not be respected by all cameras)
# cap.set(cv2.CAP_PROP_FPS, TARGET_FPS) # This often doesn't work for webcams

start_time = time.time()
frame_count = 0

# --- Capture Loop ---
while frame_count < NUM_FRAMES:
    frame_start_time = time.time()

    # 1. Capture the frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # 2. Save the image
    filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(filename, frame)

    frame_count += 1

    # 3. Calculate remaining time to wait (to maintain FPS)
    frame_process_time = time.time() - frame_start_time
    time_to_wait = T_FRAME - frame_process_time

    if time_to_wait > 0:
        time.sleep(time_to_wait)

    # Optional: Display elapsed time and actual FPS (for testing)
    elapsed_time = time.time() - start_time
    actual_fps = frame_count / elapsed_time
    print(f"Captured frame {frame_count}/{NUM_FRAMES} | Actual FPS: {actual_fps:.2f}", end='\r')

# --- Cleanup ---
cap.release()
print("\nCapture finished.")
print(f"Total time: {time.time() - start_time:.2f} seconds.")