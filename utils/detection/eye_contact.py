import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def detect_eye_contact(video_path, gaze_threshold = 0.15, min_frames_ratio = 0.7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
       raise ValueError(f"Cannot open video: {video_path}")

    with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as face_mesh :
        total_frames = 0
        avoidance_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
              break

            total_frames += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)


            if results.multi_face_landmarks:
               for face_landmarks in results.multi_face_landmarks:
                  # Left eye landmarks (example indices from mediapipe FaceMesh)
                  left_eye_indices = [33, 133]
                  right_eye_indices = [362, 263]

                  # Get average X position of each type
                  left_eye_x = sum([face_landmarks.landmark[i].x for i in left_eye_indices]) / 2
                  right_eye_x = sum([face_landmarks.landmark[i].x for i in right_eye_indices]) / 2

                  # Eye center vs. face center
                  face_center_x = face_landmarks[i].x # nose tip
                  gaze_offset = abs(((left_eye_x + right_eye_x) / 2) - face_center_x)

                  # If gaze offset > threshold -> avoidance
                  if gaze_offset > gaze_threshold:
                     avoidance_frames += 1
        
        cap.release()

        # Ratio of avoidance frames to total frames
        if total_frames == 0:
           return 0
        
        avoidance_ratio = avoidance_frames / total_frames
        return 1 if avoidance_ratio > min_frames_ratio else 0
