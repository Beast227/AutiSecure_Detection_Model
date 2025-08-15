import cv2
import numpy as np
import math
import os

def detect_object_lining_up(video_path, frame_skip=5, min_objects=3, alignment_tolerance=20, spacing_tolerance=0.15, persistence_ratio=0.3):
    """
    Detects 'lining up objects' behavior in a video.
    Returns: 1 if lining-up pattern detected, else 0.
    """

    base_path = os.path.dirname(__file__)  # folder where this script is located
    proto = os.path.join(base_path, "mobilenet_ssd.prototxt")
    model = os.path.join(base_path, "mobilenet_ssd.caffemodel")

    net = cv2.dnn.readNetFromCaffe(proto, model)                      # Pretrained model file


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lined_up_frames = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames for speed
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % frame_skip != 0:
            continue

        processed_frames += 1

        # Prepare input for object detector
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Extract center points of detected objects
        centers = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                center_x = (startX + endX) / 2
                center_y = (startY + endY) / 2
                centers.append((center_x, center_y))

        # Skip if too few objects detected
        if len(centers) < min_objects:
            continue

        # Check alignment: Fit a line and calculate distances
        centers_np = np.array(centers)
        [vx, vy, cx, cy] = cv2.fitLine(centers_np, cv2.DIST_L2, 0, 0.01, 0.01)
        distances = []
        for (px, py) in centers:
            dist = abs(vy * px - vx * py + (cx * vy - cy * vx))
            distances.append(dist)

        # Check if all objects are close to the fitted line
        aligned = all(d < alignment_tolerance for d in distances)

        # Check if spacing is roughly equal
        sorted_centers = sorted(centers, key=lambda c: (c[0], c[1]))
        spacings = [math.dist(sorted_centers[i], sorted_centers[i+1]) for i in range(len(sorted_centers)-1)]
        avg_spacing = np.mean(spacings)
        spacing_ok = all(abs(s - avg_spacing) / avg_spacing < spacing_tolerance for s in spacings)

        if aligned and spacing_ok:
            lined_up_frames += 1

    cap.release()

    # Decide if behavior is present
    if processed_frames > 0 and (lined_up_frames / processed_frames) >= persistence_ratio:
        return 1  # Lining-up detected
    return 0  # Not detected


results = detect_object_lining_up("3.mp4")
if results == 1:
    print("Lining-up behavior detected.")
else:
    print("No lining-up behavior detected.")