import cv2
import numpy as np

def detect_hyper_or_hypo_reactivity(
    video_path,
    frame_skip=3,
    hyper_spike_threshold=3.0, # How sharp a motion spike must be to be a 'flinch'
    hypo_stillness_threshold=0.3, # Motion level below which is considered 'still'
    hypo_duration_seconds=5, # How many seconds of stillness to be considered 'hypo-reactive'
):
    """
    Detects signs of hyper- or hypo-reactivity in a video by analyzing motion patterns.

    - Hyper-reactivity is detected by sharp spikes in motion (startle/flinch).
    - Hypo-reactivity is detected by prolonged periods of extreme stillness.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int): Skips frames to speed up processing.
        hyper_spike_threshold (float): The minimum increase in motion score to be a spike.
        hypo_stillness_threshold (float): The motion score below which is considered still.
        hypo_duration_seconds (int): Consecutive seconds of stillness to trigger detection.

    Returns:
        dict: A dictionary like {'hyper': 1, 'hypo': 0} indicating detection.
    """
    # --- 1. Initialize Video and Parameters ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return {'hyper': 0, 'hypo': 0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default FPS if not available

    # Calculate the number of consecutive still frames needed to trigger hypo-reactivity
    hypo_streak_threshold = int(fps * hypo_duration_seconds / frame_skip)

    # --- 2. Initialize Counters and State Variables ---
    processed_frames = 0
    frame_idx = 0
    prev_gray = None
    prev_motion_score = 0.0
    low_motion_streak = 0
    
    hyper_detected = False
    hypo_detected = False

    # --- 3. Process Video Frame by Frame ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Optimization: Skip frames
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        processed_frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0

        if prev_gray is not None:
            # Calculate overall motion using Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.mean(magnitude)

            # --- 4a. Check for Hyper-Reactivity (Sudden Spikes) ---
            # Calculate the change in motion from the previous frame
            motion_delta = motion_score - prev_motion_score
            if motion_delta > hyper_spike_threshold:
                hyper_detected = True # A single, large spike is a significant event
                print(f"Frame {frame_idx}: Hyper-reactive spike detected! Motion increase: {motion_delta:.2f}")

            # --- 4b. Check for Hypo-Reactivity (Prolonged Stillness) ---
            if motion_score < hypo_stillness_threshold:
                low_motion_streak += 1 # Increment the stillness counter
            else:
                low_motion_streak = 0 # Reset if there is movement

            # If the stillness streak exceeds our duration threshold, flag it
            if low_motion_streak >= hypo_streak_threshold:
                hypo_detected = True
                print(f"Frame {frame_idx}: Hypo-reactive stillness detected! Streak of {low_motion_streak} frames.")
        
        prev_gray = gray
        prev_motion_score = motion_score
        frame_idx += 1
        
        # Optimization: If both are found, no need to process the rest of the video
        if hyper_detected and hypo_detected:
            break

    # --- 5. Final Decision ---
    cap.release()
    
    results = {
        'hyper': 1 if hyper_detected else 0,
        'hypo': 1 if hypo_detected else 0
    }
    
    print("\n--- Detection Summary ---")
    print(f"Hyper-reactivity Detected: {'Yes' if results['hyper'] else 'No'}")
    print(f"Hypo-reactivity Detected:  {'Yes' if results['hypo'] else 'No'}")
    
    return results

# --- Example Usage ---
# result = detect_hyper_or_hypo_reactivity("your_video.mp4")
# print("\nFinal Result:", result)