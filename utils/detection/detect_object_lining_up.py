import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

def detect_object_lining_up(video_path):
    import cv2
    import numpy as np
    import os
    from sklearn.cluster import DBSCAN
    from sklearn.linear_model import LinearRegression

    proto = os.path.join(os.path.dirname(__file__), "mobilenet_ssd.prototxt")
    model = os.path.join(os.path.dirname(__file__), "mobilenet_ssd.caffemodel")
    net = cv2.dnn.readNetFromCaffe(proto, model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return 0

    frame_skip = 2
    confidence_threshold = 0.2
    rmse_tolerance = 30.0  # more lenient
    spacing_variation = 1.0  # allow more variation

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=15, detectShadows=False)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_skip != 0:
            continue

        h, w = frame.shape[:2]
        candidate_objects = []

        # 1️⃣ Object detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                width, height = endX - startX, endY - startY
                if 10 < width < 250 and 10 < height < 250:
                    candidate_objects.append(((startX + endX) / 2, (startY + endY) / 2))

        # 2️⃣ Motion-based detection fallback
        if len(candidate_objects) < 3:
            fg_mask = back_sub.apply(frame)
            fg_mask = cv2.medianBlur(fg_mask, 5)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if 10 < cw < 250 and 10 < ch < 250:
                    candidate_objects.append((x + cw / 2, y + ch / 2))

        # 3️⃣ Alignment check
        if len(candidate_objects) >= 3:
            clustering = DBSCAN(eps=w * 0.15, min_samples=3).fit(candidate_objects)
            labels = clustering.labels_

            for label in set(labels):
                if label == -1:
                    continue
                cluster_points = np.array([candidate_objects[i] for i in range(len(candidate_objects)) if labels[i] == label])
                if len(cluster_points) >= 3:
                    X = cluster_points[:, 0].reshape(-1, 1)
                    y_vals = cluster_points[:, 1]
                    model_lr = LinearRegression().fit(X, y_vals)
                    predictions = model_lr.predict(X)
                    rmse = np.sqrt(np.mean((y_vals - predictions) ** 2))

                    dists = [np.linalg.norm(cluster_points[i] - cluster_points[i - 1]) for i in range(1, len(cluster_points))]
                    if len(dists) > 1:
                        std_dev = np.std(dists)
                        mean_dist = np.mean(dists)

                        if rmse < rmse_tolerance and (std_dev / mean_dist) < spacing_variation:
                            cap.release()
                            return 1  # Found alignment immediately

    cap.release()
    return 0


# Example usage
result = detect_object_lining_up("1.mp4")
print(f"Lining up objects detected: {result}")