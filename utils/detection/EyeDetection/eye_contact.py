import cv2
import mediapipe as mp
import numpy as np

# --- Settings ---
GAZE_THRESHOLD = 0.15
SMOOTHING_FACTOR = 0.6
FRAMES_REQUIRED = 3
FACE_SCALE_THRESHOLD = 0.15  # reduced threshold so real faces pass

# --- New Global Setting ---
EYE_CONTACT_PERCENTAGE_THRESHOLD = 0.1  # 10% of the video frames with eye contact

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh

# --- Landmark Indices ---
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]
NOSE_TIP_IDX = [1]


def average_landmarks(landmarks, indices):
    """Averages the coordinates of a set of landmarks."""
    points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
    return np.mean(points, axis=0)


def detect_eye_contact_final(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    eye_contact_frames = 0
    smoothed_offset = 0
    consecutive_contact = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                valid_faces = []
                for lm in results.multi_face_landmarks:
                    # Height of face in normalized coordinates
                    face_height = abs(lm.landmark[10].y - lm.landmark[152].y)
                    if face_height > FACE_SCALE_THRESHOLD:
                        valid_faces.append(lm)
                        print(f"[DEBUG] Face height: {face_height:.3f} -> accepted")
                    else:
                        print(f"[DEBUG] Face height: {face_height:.3f} -> skipped")

                if valid_faces:
                    # Choose largest face
                    main_face = max(valid_faces, key=lambda lm: abs(lm.landmark[10].y - lm.landmark[152].y))
                else:
                    continue

                left_eye_center = average_landmarks(main_face.landmark, LEFT_EYE_IDX)
                right_eye_center = average_landmarks(main_face.landmark, RIGHT_EYE_IDX)
                avg_eye_center = (left_eye_center + right_eye_center) / 2

                nose_tip = np.array([main_face.landmark[1].x, main_face.landmark[1].y])
                gaze_vector = avg_eye_center - nose_tip
                face_width = main_face.landmark[362].x - main_face.landmark[133].x
                normalized_gaze_x = gaze_vector[0] / face_width

                smoothed_offset = (
                    SMOOTHING_FACTOR * normalized_gaze_x
                    + (1 - SMOOTHING_FACTOR) * smoothed_offset
                )

                print(f"[DEBUG] Frame {total_frames} | Gaze offset: {smoothed_offset:.3f}")

                if abs(smoothed_offset) < GAZE_THRESHOLD:
                    consecutive_contact += 1
                else:
                    consecutive_contact = 0

                if consecutive_contact >= FRAMES_REQUIRED:
                    eye_contact_frames += 1
                    print(f"[DEBUG] Eye contact detected! Total so far: {eye_contact_frames}")

    cap.release()

    # Final calculation and return value
    if total_frames > 0:
        eye_contact_percentage = eye_contact_frames / total_frames
        print(f"[RESULT] Eye contact percentage: {eye_contact_percentage:.2%}")
        return 0 if eye_contact_percentage >= EYE_CONTACT_PERCENTAGE_THRESHOLD else 1
    else:
        print("[ERROR] No frames read from video.")
        return 0


# Example usage:
output = detect_eye_contact_final("11.mp4")
print(f"Final output: {output}")
