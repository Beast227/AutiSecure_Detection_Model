import cv2
import numpy as np
import os

def detect_non_responsiveness(
    video_path,
    frame_skip=5,
    stillness_threshold=0.35, # Motion level below which is considered 'still'
    duration_seconds=8,       # How many consecutive seconds of non-responsiveness to trigger
):
    """
    Detects prolonged non-responsiveness in a video.

    This is identified by a sustained period of both physical stillness and a
    lack of detectable face engagement.

    Args:
        video_path (str): Path to the video file.
        frame_skip (int): Skips frames to speed up processing.
        stillness_threshold (float): The motion score below which is considered still.
        duration_seconds (int): Consecutive seconds of non-responsiveness to trigger detection.

    Returns:
        int: 1 if non-responsiveness is detected, 0 otherwise.
    """
    # --- 1. Initialize Video, Models, and Parameters ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return 0

    # Load OpenCV's pre-trained face detector (lightweight and effective)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 # Default FPS if not available

    # Calculate the number of consecutive frames needed to trigger the detection
    non_responsive_streak_threshold = int(fps * duration_seconds / frame_skip)

    # --- 2. Initialize Counters and State Variables ---
    processed_frames = 0
    frame_idx = 0
    prev_gray = None
    non_responsive_streak = 0
    
    detection_triggered = False

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
            # --- 4a. Analyze Motion ---
            # Calculate overall motion using Farneback Optical Flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.mean(magnitude)
            is_still = motion_score < stillness_threshold

            # --- 4b. Analyze Face Engagement ---
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            is_face_detected = len(faces) > 0

            # --- 5. Combine Clues and Track Streak ---
            # The key condition: Is the scene still AND is the face not engaged?
            if is_still and not is_face_detected:
                non_responsive_streak += 1
            else:
                # If there's movement or a face is seen, the streak is broken
                non_responsive_streak = 0
            
            # If the streak of non-responsiveness exceeds our duration threshold, trigger detection
            if non_responsive_streak >= non_responsive_streak_threshold:
                detection_triggered = True
                print(f"Frame {frame_idx}: Non-responsiveness detected! Streak of {non_responsive_streak} frames.")
                break # Exit early once the behavior is confirmed

        prev_gray = gray
        frame_idx += 1

    # --- 6. Final Decision ---
    cap.release()
    
    if detection_triggered:
        print("\nFinal Decision: Prolonged non-responsiveness was detected.")
        return 1
    else:
        print("\nFinal Decision: No significant non-responsiveness was detected.")
        return 0

# --- Example Usage ---
# result = detect_non_responsiveness("your_video.mp4")
# print("\nFinal Result:", result)