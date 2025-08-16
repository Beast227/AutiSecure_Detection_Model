import cv2
import mediapipe as mp
import math

class SpinningDetector:
    def __init__(self, angle_threshold=25, sequence_length=5, motion_threshold=0.7):
        self.previous_angle = None
        self.angle_changes = []
        self.angle_threshold = angle_threshold  # degrees/frame considered spinning
        self.sequence_length = sequence_length  # how many recent frames to check
        self.motion_threshold = motion_threshold  # proportion of frames with fast enough rotation
    
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False)
    
    def calculate_torso_angle(self, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        mid_hip_x = (left_hip.x + right_hip.x) / 2
        mid_hip_y = (left_hip.y + right_hip.y) / 2
    
        dx = mid_shoulder_x - mid_hip_x
        dy = mid_shoulder_y - mid_hip_y
        angle = math.degrees(math.atan2(dy, dx))
        return angle

    def process_frame(self, frame):
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return False
        landmarks = results.pose_landmarks.landmark
        current_angle = self.calculate_torso_angle(landmarks)
        detected = False
    
        if self.previous_angle is not None:
            angle_diff = current_angle - self.previous_angle
            angle_diff = (angle_diff + 180) % 360 - 180
            self.angle_changes.append(abs(angle_diff))
            if len(self.angle_changes) > self.sequence_length:
                self.angle_changes.pop(0)
    
            high_motion_frames = sum(
                1 for delta in self.angle_changes if delta > self.angle_threshold
            )
            if (
                len(self.angle_changes) == self.sequence_length and
                high_motion_frames / self.sequence_length > self.motion_threshold
            ):
                detected = True
    
        self.previous_angle = current_angle
        return detected

def analyze_video_for_spinning(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = SpinningDetector()
    spinning_detected = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if detector.process_frame(frame):
            spinning_detected = True
            break
    cap.release()
    return spinning_detected
