import cv2
import numpy as np
import mediapipe as mp
from collections import deque

def calculate_velocity(p1, p2, time_delta=1):
    """Calculates the Euclidean distance (velocity) between two 3D points."""
    if p1 is None or p2 is None:
        return 0
    # Use 3D coordinates for more accurate movement tracking
    return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2) / time_delta

def detect_upper_limb_stereotypies(
    video_path,
    frame_skip=2,
    velocity_threshold=0.15,      # Threshold for detecting rapid wrist movement
    proximity_threshold=0.2,      # How close hands must be to head/torso (normalized)
    time_window_seconds=2,        # Analyze movement patterns over a 2-second window
    persistence_ratio=0.2         # % of frames needing stereotypy to trigger detection
):
    """
    Detects upper limb stereotypies like hand-flapping in a video.

    This function analyzes the velocity, proximity to the body, and repetitive nature
    of wrist movements to identify stereotypy.

    Returns:
        int: 1 if stereotypy is detected, 0 otherwise.
    """
    # --- 1. Initialize MediaPipe Pose and Video ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default FPS if not available

    # --- 2. Initialize Buffers and Counters ---
    # A deque is a special list that automatically removes old items when new ones are added.
    # We use it to store recent wrist positions for pattern analysis.
    window_size = int(fps * time_window_seconds / frame_skip)
    left_wrist_history = deque(maxlen=window_size)
    right_wrist_history = deque(maxlen=window_size)

    stereotypy_frames = 0
    processed_frames = 0
    frame_idx = 0
    prev_landmarks = None

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
        h, w, _ = frame.shape
        
        # --- 4. Perform Pose Estimation ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        is_stereotypy_in_frame = False
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get key landmark positions
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # --- 5. Analyze Movement Characteristics ---
            if prev_landmarks:
                # 5a. Calculate Instantaneous Velocity
                left_v = calculate_velocity(prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST], left_wrist)
                right_v = calculate_velocity(prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], right_wrist)
                
                # 5b. Check Proximity to Head and Torso
                dist_left_to_nose = calculate_velocity(left_wrist, nose)
                dist_right_to_nose = calculate_velocity(right_wrist, nose)
                dist_left_to_shoulder = calculate_velocity(left_wrist, right_shoulder) # Check against opposite shoulder
                dist_right_to_shoulder = calculate_velocity(right_wrist, left_shoulder)

                is_fast_movement = left_v > velocity_threshold or right_v > velocity_threshold
                is_proximal = (dist_left_to_nose < proximity_threshold or
                               dist_right_to_nose < proximity_threshold or
                               dist_left_to_shoulder < proximity_threshold or
                               dist_right_to_shoulder < proximity_threshold)

                # --- 6. Analyze Repetitive Patterns in Time Window ---
                left_wrist_history.append(left_wrist.y) # Store vertical position
                right_wrist_history.append(right_wrist.y)

                if len(left_wrist_history) == window_size:
                    # A simple way to check for repetition is to see if the movement changes direction often.
                    # We count the number of "peaks" and "troughs" in the recent movement history.
                    # High frequency of direction change suggests an oscillatory, repetitive motion.
                    left_peaks = np.sum(np.diff(np.sign(np.diff(left_wrist_history))) != 0)
                    right_peaks = np.sum(np.diff(np.sign(np.diff(right_wrist_history))) != 0)
                    
                    # A repetitive motion will have many peaks (e.g., more than 1/4 of the window size)
                    is_repetitive = (left_peaks > window_size / 4) or (right_peaks > window_size / 4)

                    # FINAL CHECK: A frame shows stereotypy if movement is fast, close to the body, AND repetitive.
                    if is_fast_movement and is_proximal and is_repetitive:
                        is_stereotypy_in_frame = True

            prev_landmarks = landmarks
        else:
            prev_landmarks = None # Reset if person is not detected

        if is_stereotypy_in_frame:
            stereotypy_frames += 1

        frame_idx += 1

    # --- 7. Final Decision ---
    cap.release()
    pose.close()

    if processed_frames == 0:
        return 0

    final_ratio = stereotypy_frames / processed_frames
    print(f"Upper Limb Stereotypy Ratio: {final_ratio:.2f} ({stereotypy_frames} / {processed_frames} frames)")

    if final_ratio >= persistence_ratio:
        return 1
    return 0

# --- Example Usage ---
# result = detect_upper_limb_stereotypies("your_video.mp4")
# if result == 1:
#     print("Final Decision: Upper limb stereotypy detected.")
# else:
#     print("Final Decision: No significant upper limb stereotypy detected.")