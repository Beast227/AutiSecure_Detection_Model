import cv2
import mediapipe as mp
import numpy as np

class SelfHitDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)

    def detect_self_hitting(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Example: detect hand near head (simple heuristic)
            left_hand = np.array([lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x,
                                  lm[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y])
            right_hand = np.array([lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x,
                                   lm[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y])
            head = np.array([lm[mp.solutions.pose.PoseLandmark.NOSE].x,
                             lm[mp.solutions.pose.PoseLandmark.NOSE].y])

            # Distance check
            left_dist = np.linalg.norm(left_hand - head)
            right_dist = np.linalg.norm(right_hand - head)

            # Simple probability logic
            prob = 1.0 if left_dist < 0.1 or right_dist < 0.1 else 0.0
            print(f"Frame probability: {prob}")
            return prob
        else:
            print("Frame probability: 0.0")
            return 0.0

    def process_video(self, video_path, hit_threshold_frames=10):
        cap = cv2.VideoCapture(video_path)
        consecutive_hits = 0
        final_result = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            prob = self.detect_self_hitting(frame)

            if prob == 1.0:
                consecutive_hits += 1
                if consecutive_hits >= hit_threshold_frames:
                    final_result = 1
                    break
            else:
                consecutive_hits = 0

        cap.release()
        return final_result
