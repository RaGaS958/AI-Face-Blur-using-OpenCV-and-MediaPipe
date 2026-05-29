import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Face outline indices
oval_indices = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # RGB conversion
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh
    results = face_mesh.process(rgb)

    # Frames for visualization
    landmark_frame = frame.copy()
    contour_frame = frame.copy()

    # Empty mask
    mask = np.zeros((h, w), dtype=np.uint8)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            face_points = []

            # -----------------------------------
            # Draw landmarks
            # -----------------------------------

            for landmark in face_landmarks.landmark:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    landmark_frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

            # -----------------------------------
            # Get face contour
            # -----------------------------------

            for idx in oval_indices:

                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)

                face_points.append((x, y))

            face_points = np.array(face_points, dtype=np.int32)

            # Draw contour
            cv2.polylines(
                contour_frame,
                [face_points],
                True,
                (0, 255, 255),
                3
            )

            # Fill face mask
            cv2.fillConvexPoly(mask, face_points, 255)

    # -----------------------------------
    # Expand mask area
    # -----------------------------------

    kernel = np.ones((75, 75), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)

    # Smooth edges
    smooth_mask = cv2.GaussianBlur(
        expanded_mask,
        (71, 71),
        40
    )

    # -----------------------------------
    # Blur frame
    # -----------------------------------

    blurred = cv2.GaussianBlur(
        frame,
        (151, 151),
        60
    )

    # Normalize mask
    normalized_mask = smooth_mask.astype(float) / 255.0
    normalized_mask = np.expand_dims(normalized_mask, axis=2)

    # Final output
    result = (
        blurred * normalized_mask +
        frame * (1 - normalized_mask)
    ).astype(np.uint8)

    # Convert masks to color
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    smooth_display = cv2.cvtColor(
        smooth_mask,
        cv2.COLOR_GRAY2BGR
    )

    # -----------------------------------
    # SHOW 6 IMPORTANT WINDOWS
    # -----------------------------------

    cv2.imshow("1 - Original Webcam", frame)

    cv2.imshow("2 - Face Landmarks", landmark_frame)

    cv2.imshow("3 - Face Contour Detection", contour_frame)

    cv2.imshow("4 - Face Mask", mask_display)

    cv2.imshow("5 - Smooth Expanded Mask", smooth_display)

    cv2.imshow("6 - Final Face Blur", result)

    # Exit with ESC
    key = cv2.waitKey(1)

    if key == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()