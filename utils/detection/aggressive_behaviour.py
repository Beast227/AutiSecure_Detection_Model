import cv2
import mediapipe as mp
import numpy as np
from fer import FER

def calculate_velocity(p1, p2, time_delta=1):
    """Calculates the Euclidean distance (as a proxy for velocity) between two points."""
    if p1 is None or p2 is None:
        return 0
    return np.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2) / time_delta

def detect_aggressive_behavior_improved(
    video_path,
    motion_threshold=1.0,       # Avg. pixel motion magnitude from optical flow
    anger_threshold=0.6,        # Confidence threshold for 'angry' emotion
    wrist_velocity_threshold=0.08, # Threshold for normalized wrist movement speed
    aggression_score_threshold=2.5, # Min score in a frame to be 'aggressive'
    aggression_frame_ratio=0.05, # % of aggressive frames to classify the video
    frame_skip=3                # Process every Nth frame to improve performance
):
    """
    Detects aggressive behavior in a video using a combination of optical flow,
    pose velocity, and facial emotion recognition.

    Returns:
        1 if aggressive behavior is detected, 0 otherwise.
    """
    mp_pose = mp.solutions.pose
    detector = FER(mtcnn=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    aggressive_frame_count = 0
    processed_frame_count = 0
    frame_idx = 0

    prev_gray = None
    prev_landmarks = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- 1. Frame Skipping for Performance ---
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            processed_frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            aggression_score = 0
            
            # --- 2. Motion Detection (Optical Flow) ---
            motion_score = 0
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = np.mean(magnitude)
                if motion_score > motion_threshold:
                    aggression_score += 1.0  # Weight for general motion

            # --- 3. Pose Analysis (Wrist Velocity) ---
            wrist_velocity = 0
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks and prev_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Calculate velocity for left and right wrists
                left_wrist_v = calculate_velocity(prev_landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
                right_wrist_v = calculate_velocity(prev_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
                
                # Use the max velocity of either wrist
                wrist_velocity = max(left_wrist_v, right_wrist_v)

                if wrist_velocity > wrist_velocity_threshold:
                    aggression_score += 2.0 # Higher weight for specific actions

            if results.pose_landmarks:
                prev_landmarks = results.pose_landmarks.landmark
            else:
                prev_landmarks = None # Reset if person leaves the frame

            # --- 4. Facial Emotion Recognition (Conditional) ---
            # Only run the heavy FER model if there's already a motion/pose cue
            max_anger_score = 0
            if aggression_score > 0:
                emotions = detector.detect_emotions(frame)
                for face in emotions:
                    anger = face["emotions"].get("angry", 0)
                    if anger > max_anger_score:
                        max_anger_score = anger
                
                if max_anger_score > anger_threshold:
                    aggression_score += 1.5 # Weight for strong emotion

            # --- 5. Final Decision for the Frame ---
            if aggression_score >= aggression_score_threshold:
                aggressive_frame_count += 1

            prev_gray = gray
            frame_idx += 1

    cap.release()

    # --- 6. Final Decision for the Video ---
    if processed_frame_count == 0:
        return 0 # No frames were processed

    final_ratio = aggressive_frame_count / processed_frame_count
    print(f"Aggression Ratio: {final_ratio:.2f} ({aggressive_frame_count} / {processed_frame_count} frames)")

    if final_ratio >= aggression_frame_ratio:
        return 1  # Aggressive
    return 0  # Not aggressive

# --- Example Usage ---
# video_file = "path/to/your/video.mp4"
result = detect_aggressive_behavior_improved("1.mp4")
if result == 1:
    print("Aggressive behavior was detected in the video.")
else:
    print("No significant aggressive behavior was detected.")