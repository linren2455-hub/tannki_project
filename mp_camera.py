# mediapipe用
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            lm = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            shoulder = [lm[11].x * w, lm[11].y * h]
            elbow    = [lm[13].x * w, lm[13].y * h]
            wrist    = [lm[15].x * w, lm[15].y * h]

            print(f"肩: ({shoulder[0]:.1f}, {shoulder[1]:.1f})")
            print(f"肘: ({elbow[0]:.1f}, {elbow[1]:.1f})")
            print(f"手首: ({wrist[0]:.1f}, {wrist[1]:.1f})")
            print("---")

        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()