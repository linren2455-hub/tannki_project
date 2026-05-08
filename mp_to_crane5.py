#!/usr/bin/env python3
# mp_to_crane3.py
# カメラ→MediaPipe→IK→CRANE-X7 メインスクリプト
import rclpy
import cv2
import mediapipe as mp
import threading
import time
import sys

sys.path.append('/home/iwamoto/openpose/openpose_python')
from pose_utils import get_rel, camera_to_robot, get_averaged_pos, draw_info
from crane_controller import CraneDriver

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

class CraneController(CraneDriver):
    def __init__(self):
        super().__init__()

        # カメラ
        self.pose = mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)
        self.cam_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # 状態管理
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_moving = False
        self.last_pos = None
        self.pos_history = []
        self.N_AVERAGE = 3

        # キャリブレーション
        self.phase = "calib_x"
        self.rel_x_max = 200.0
        self.rel_z_max = 200.0
        self.calib_samples = []
        self.calib_start_time = time.time()

        # カメラスレッド起動
        self.camera_thread = threading.Thread(
            target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        self.get_logger().info("=" * 40)
        self.get_logger().info("【キャリブレーション】")
        self.get_logger().info("① 右腕を横に最大に伸ばしてください（5秒間）")
        self.get_logger().info("=" * 40)

        self.timer = self.create_timer(0.1, self.timer_callback)

    def camera_loop(self):
        """カメラ取得・表示を30FPSで維持（制御とは独立）"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.pose.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)

                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                rel_x, rel_z, _ = get_rel(lm, h, w)

                draw_info(frame, self.phase,
                          rel_x, rel_z,
                          self.rel_x_max, self.rel_z_max,
                          self.last_pos)

            with self.frame_lock:
                self.latest_frame = frame.copy()

            cv2.imshow("MediaPipe", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def move_task(self, rx, ry, rz):
        """IK計算→trajectory送信（上書き方式）"""
        joint_positions = self.solve_ik(rx, ry, rz)
        if joint_positions:
            # move_timeを長めに設定（次の指令が来たら上書きされる）
            self.send_trajectory(joint_positions[:7], move_time=2.0)
        else:
            self.get_logger().warn("IK解なし")
        self.is_moving = False  # すぐにFalseにして次の指令を受け付ける

    def timer_callback(self):
        """0.1秒ごとに呼ばれる制御ループ"""
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        results = self.pose.process(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return

        lm = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        rel_x, rel_z, visibility = get_rel(lm, h, w)

        if visibility < 0.3:
            return

        elapsed = time.time() - self.calib_start_time

        # ① キャリブレーション：横方向（5秒間）
        if self.phase == "calib_x":
            if rel_x > 0:
                self.calib_samples.append(rel_x)
            if elapsed > 5.0:
                self.rel_x_max = max(self.calib_samples) if self.calib_samples else 200.0
                self.get_logger().info(f"✅ rel_x_max = {self.rel_x_max:.0f}")
                self.get_logger().info("② 右腕を上に最大に上げてください（5秒間）")
                self.calib_samples = []
                self.calib_start_time = time.time()
                self.phase = "calib_z"
            return

        # ② キャリブレーション：上方向（5秒間）
        if self.phase == "calib_z":
            if rel_z > 0:
                self.calib_samples.append(rel_z)
            if elapsed > 5.0:
                self.rel_z_max = max(self.calib_samples) if self.calib_samples else 200.0
                self.get_logger().info(f"✅ rel_z_max = {self.rel_z_max:.0f}")
                self.get_logger().info("キャリブレーション完了！制御開始します")
                self.pos_history = []
                self.calib_start_time = time.time()
                self.phase = "control"
            return

        # ③ 制御フェーズ（MediaPipeの結果をそのまま使う）
        if self.phase == "control":
            if self.is_moving:
                return

            rx, ry, rz = camera_to_robot(
                rel_x, rel_z, self.rel_x_max, self.rel_z_max)
            self.pos_history, averaged = get_averaged_pos(
                self.pos_history, rx, ry, rz, n=self.N_AVERAGE)

            if averaged is None:
                return
            rx, ry, rz = averaged

            import numpy as np
            current_pos = np.array([rx, ry, rz])
            if self.last_pos is not None:
                if np.linalg.norm(current_pos - np.array(self.last_pos)) < 0.02:
                    return

            self.last_pos = (rx, ry, rz)
            self.is_moving = True
            self.get_logger().info(
                f"rel=({rel_x:.0f},{rel_z:.0f}) → robot x={rx:.3f} z={rz:.3f}")

            threading.Thread(
                target=self.move_task,
                args=(rx, ry, rz),
                daemon=True
            ).start()

def main():
    rclpy.init()
    node = CraneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()