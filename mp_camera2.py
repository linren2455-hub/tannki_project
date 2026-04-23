import cv2
import mediapipe as mp
import os

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def get_keypoints(frame, pose):
    """1フレームから関節座標を取得"""
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks is None:
        return None
    
    lm = results.pose_landmarks.landmark
    h, w, _ = frame.shape
    
    # OpenPoseの2,3,4に対応（右肩,右肘,右手首）
    keypoints = {
        "right_shoulder": [lm[12].x * w, lm[12].y * h, lm[12].visibility],
        "right_elbow":    [lm[14].x * w, lm[14].y * h, lm[14].visibility],
        "right_wrist":    [lm[16].x * w, lm[16].y * h, lm[16].visibility],
    }
    return keypoints, results

def print_keypoints(keypoints):
    """座標をクリアして表示"""
    os.system('clear')
    print("=== 関節座標 ===")
    for name, val in keypoints.items():
        print(f"{name}: x={val[0]:.1f}, y={val[1]:.1f}, conf={val[2]:.2f}")

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = get_keypoints(frame, pose)
            if result:
                keypoints, results = result
                print_keypoints(keypoints)
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("MediaPipe Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()