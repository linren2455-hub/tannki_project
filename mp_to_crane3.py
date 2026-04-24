#!/usr/bin/env python3
#ikをMoveitからikpyに置き換え
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
import ikpy.chain
import cv2
import mediapipe as mp
import numpy as np
import threading
import time

JOINT_NAMES = [
    'crane_x7_shoulder_fixed_part_pan_joint',
    'crane_x7_shoulder_revolute_part_tilt_joint',
    'crane_x7_upper_arm_revolute_part_twist_joint',
    'crane_x7_upper_arm_revolute_part_rotate_joint',
    'crane_x7_lower_arm_fixed_part_joint',
    'crane_x7_lower_arm_revolute_part_joint',
    'crane_x7_wrist_joint',
]

class CraneController(Node):
    def __init__(self):
        super().__init__('crane_controller')

        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/crane_x7_arm_controller/joint_trajectory',
            10)

        self.last_positions = [0.0] * 7
        self.last_velocities = [0.0] * 7
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10)

        # ikpyチェーン読み込み
        self.chain = ikpy.chain.Chain.from_urdf_file(
            '/home/iwamoto/crane_x7_simple.urdf',
            base_elements=['base_link'],
            base_element_type='link',
            active_links_mask=[False, False, False, True, True, True, True, True, True, True]
        )
        self.get_logger().info("✅ ikpyチェーン読み込み完了")

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)
        self.cam_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_moving = False
        self.last_pos = None
        self.pos_history = []
        self.N_AVERAGE = 3  # 5→3に

        # キャリブレーション用
        self.phase = "calib_x"  # calib_x → calib_z → control
        self.rel_x_max = 200.0  # デフォルト値
        self.rel_z_max = 200.0  # デフォルト値
        self.calib_samples = []
        self.calib_start_time = None

        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        self.get_logger().info("✅ 準備完了")
        self.get_logger().info("=" * 40)
        self.get_logger().info("【キャリブレーション】")
        self.get_logger().info("① 右腕を横に最大に伸ばしてください（5秒間）")
        self.get_logger().info("=" * 40)

        self.calib_start_time = time.time()
        self.timer = self.create_timer(0.1, self.timer_callback)

    def joint_state_callback(self, msg):
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                if len(msg.position) > idx:
                    self.last_positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.last_velocities[i] = msg.velocity[idx]

    def send_trajectory(self, target_positions, move_time=1.0):
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = JOINT_NAMES

        start = JointTrajectoryPoint()
        start.positions = self.last_positions.copy()
        start.velocities = [0.0] * 7
        start.time_from_start = Duration(sec=0, nanosec=0)

        end = JointTrajectoryPoint()
        end.positions = target_positions
        end.velocities = [0.0] * 7
        sec = int(move_time)
        nanosec = int((move_time - sec) * 1e9)
        end.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj.points = [start, end]
        self.traj_pub.publish(traj)

    def solve_ik(self, x, y, z):
        """ikpyでIK計算（高速・安定）"""
        try:
            # 現在の関節角度を初期値として使う
            initial = [0.0, 0.0, 0.0] + self.last_positions[:7]
            angles = self.chain.inverse_kinematics(
                [x, y, z],
                initial_position=initial
            )
            # 有効関節角度（index 3〜9）を返す
            return list(angles[3:10])
        except Exception as e:
            self.get_logger().warn(f"IK失敗: {e}")
            return None

    def get_rel(self, lm, h, w):
        """肩原点の相対座標を取得（左がrel_x正、上がrel_z正）"""
        shoulder = lm[12]
        wrist = lm[16]
        sx = shoulder.x * w
        sy = shoulder.y * h
        wx = wrist.x * w
        wy = wrist.y * h

        rel_x = sx - wx  # 左が正
        rel_z = sy - wy  # 上が正（画像は上がy小）
        return rel_x, rel_z, wrist.visibility

    def camera_to_robot(self, rel_x, rel_z):
        """rel座標 → ロボット座標"""
        # 正規化（0〜1）
        nx = np.clip(rel_x / self.rel_x_max, 0.0, 1.0)
        nz = np.clip(rel_z / self.rel_z_max, 0.0, 1.0)

        # ロボット座標にマッピング
        rx = 0.10 + nx * (0.45 - 0.10)  # 0.10〜0.45
        rz = 0.20 + nz * (0.50 - 0.20)  # 0.20〜0.50
        ry = 0.00  # 固定

        return rx, ry, rz

    def get_averaged_pos(self, rx, ry, rz):
        self.pos_history.append([rx, ry, rz])
        if len(self.pos_history) > self.N_AVERAGE:
            self.pos_history.pop(0)
        if len(self.pos_history) < self.N_AVERAGE:
            return None

        arr = np.array(self.pos_history)
        averaged = []
        for i in range(3):
            col = arr[:, i]
            mean, std = col.mean(), col.std()
            mask = np.abs(col - mean) < 2 * std
            averaged.append(col[mask].mean() if mask.sum() > 0 else mean)
        return averaged

    def camera_loop(self):
        mp_draw = mp.solutions.drawing_utils
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS)

                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                rel_x, rel_z, _ = self.get_rel(lm, h, w)

                # フェーズ表示
                phase_msg = {
                    "calib_x": "① 右腕を横に最大に伸ばしてください",
                    "calib_z": "② 右腕を上に最大に上げてください",
                    "control": "✅ 制御中",
                }.get(self.phase, "")

                cv2.putText(frame, phase_msg,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"rel_x={rel_x:.0f} rel_z={rel_z:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"x_max={self.rel_x_max:.0f} z_max={self.rel_z_max:.0f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                if self.last_pos:
                    rx, ry, rz = self.last_pos
                    cv2.putText(frame, f"robot: x={rx:.2f} z={rz:.2f}",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            with self.frame_lock:
                self.latest_frame = frame.copy()

            cv2.imshow("MediaPipe", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def move_task(self, rx, ry, rz):
        joint_positions = self.solve_ik(rx, ry, rz)
        if joint_positions:
            self.send_trajectory(joint_positions[:7], move_time=0.5)
            time.sleep(0.5)  # move_timeと合わせる
        else:
            self.get_logger().warn("IK解なし")
        self.is_moving = False

    def timer_callback(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return

        lm = results.pose_landmarks.landmark
        h, w, _ = frame.shape
        rel_x, rel_z, visibility = self.get_rel(lm, h, w)

        if visibility < 0.3:
            return

        elapsed = time.time() - self.calib_start_time

        # ① キャリブレーション：横方向（5秒間）
        if self.phase == "calib_x":
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
            self.calib_samples.append(rel_z)
            if elapsed > 5.0:
                self.rel_z_max = max(self.calib_samples) if self.calib_samples else 200.0
                self.get_logger().info(f"✅ rel_z_max = {self.rel_z_max:.0f}")
                self.get_logger().info("=" * 40)
                self.get_logger().info("キャリブレーション完了！制御開始します")
                self.get_logger().info("=" * 40)
                self.pos_history = []
                self.calib_start_time = time.time()
                self.phase = "control"
            return

        # ③ 制御フェーズ
        if self.phase == "control":
            if self.is_moving:
                return

            rx, ry, rz = self.camera_to_robot(rel_x, rel_z)
            averaged = self.get_averaged_pos(rx, ry, rz)
            if averaged is None:
                return
            rx, ry, rz = averaged

            current_pos = np.array([rx, ry, rz])
            if self.last_pos is not None:
                if np.linalg.norm(current_pos - np.array(self.last_pos)) < 0.01:
                    return

            self.last_pos = (rx, ry, rz)
            self.get_logger().info(
                f"rel=({rel_x:.0f},{rel_z:.0f}) → robot x={rx:.3f} z={rz:.3f}")
            self.is_moving = True
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