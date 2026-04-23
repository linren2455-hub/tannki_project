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
    
    keypoints = {
        "right_shoulder": [lm[12].x * w, lm[12].y * h, lm[12].visibility],
        "right_elbow":    [lm[14].x * w, lm[14].y * h, lm[14].visibility],
        "right_wrist":    [lm[16].x * w, lm[16].y * h, lm[16].visibility],
    }
    return keypoints, results

def print_keypoints(keypoints):
    os.system('clear')
    print("=== 関節座標 ===")
    for name, val in keypoints.items():
        print(f"{name}: x={val[0]:.1f}, y={val[1]:.1f}, conf={val[2]:.2f}")

def get_averaged_keypoints(cap, pose, n=5, conf_threshold=0.5):
    """
    n フレームの平均座標を返す（外れ値除去あり）
    外れ値：各座標の平均±2σ の範囲外を除外
    """
    import numpy as np

    samples = []
    while len(samples) < n:
        ret, frame = cap.read()
        if not ret:
            break
        result = get_keypoints(frame, pose)
        if result is None:
            continue
        keypoints, _ = result
        # 全関節がconf_threshold以上のフレームのみ使う
        if all(v[2] >= conf_threshold for v in keypoints.values()):
            samples.append(keypoints)

    if not samples:
        return None

    # 関節ごとに平均計算
    averaged = {}
    for joint in samples[0].keys():
        xs = np.array([s[joint][0] for s in samples])
        ys = np.array([s[joint][1] for s in samples])

        # 外れ値除去（±2σ）
        for arr in [xs, ys]:
            mean, std = arr.mean(), arr.std()
            mask = np.abs(arr - mean) < 2 * std
            xs = xs[mask]
            ys = ys[mask]

        averaged[joint] = [xs.mean(), ys.mean()]

    return averaged

def to_shoulder_origin(keypoints):
    """右肩を原点とした相対座標に変換"""
    origin = keypoints["right_shoulder"]
    converted = {}
    for name, val in keypoints.items():
        converted[name] = [val[0] - origin[0], val[1] - origin[1]]
    return converted

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # 5フレーム平均取得
            avg = get_averaged_keypoints(cap, pose, n=5)
            if avg is None:
                continue

            # 右肩原点に変換
            relative = to_shoulder_origin(avg)

            os.system('clear')
            print("=== 平均座標（右肩原点） ===")
            for name, val in relative.items():
                print(f"{name}: x={val[0]:.1f}, y={val[1]:.1f}")

            # 表示用に1フレーム取る
            ret, frame = cap.read()
            if ret:
                result = get_keypoints(frame, pose)
                if result:
                    _, results = result
                    mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                cv2.imshow("MediaPipe Pose", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()