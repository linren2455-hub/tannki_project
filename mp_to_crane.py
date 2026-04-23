#!/usr/bin/env python3
# craneにつなげtrajectory_controller直接送信
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, BoundingVolume
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from rclpy.action import ActionClient
import cv2
import mediapipe as mp
import numpy as np
import threading

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

        # ① trajectory_controllerに直接送信
        self.traj_pub = self.create_publisher(
            JointTrajectory,
            '/crane_x7_arm_controller/joint_trajectory',
            10)

        # ② 現在の関節角度を購読
        self.last_positions = [0.0] * 7
        self.last_velocities = [0.0] * 7
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)

        # ③ IK用MoveGroupアクション
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self.get_logger().info("MoveGroupサーバー待機中...")
        self._action_client.wait_for_server(timeout_sec=10.0)

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)
        self.cam_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_moving = False
        self.last_pos = None

        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        self.timer = self.create_timer(2.0, self.timer_callback)  # 0.1→2.0秒に
        self.rel_wrist = (0.0, 0.0)
        self.pos_history = []  # 座標履歴（平均化用）
        self.N_AVERAGE = 5     # 平均フレーム数
        self.get_logger().info("✅ 準備完了")

    def joint_state_callback(self, msg):
        """現在の関節角度を常に更新"""
        for i, name in enumerate(JOINT_NAMES):
            if name in msg.name:
                idx = msg.name.index(name)
                if len(msg.position) > idx:
                    self.last_positions[i] = msg.position[idx]
                if len(msg.velocity) > idx:
                    self.last_velocities[i] = msg.velocity[idx]

    def send_trajectory(self, target_positions, move_time=0.5):
        """現在位置→目標位置をtrajectory_controllerに直接送信"""
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = JOINT_NAMES

        # 現在位置（開始点）
        start = JointTrajectoryPoint()
        start.positions = self.last_positions.copy()
        start.velocities = [0.0] * 7
        start.time_from_start = Duration(sec=0, nanosec=0)

        # 目標位置（終了点）
        end = JointTrajectoryPoint()
        end.positions = target_positions
        end.velocities = [0.0] * 7
        sec = int(move_time)
        nanosec = int((move_time - sec) * 1e9)
        end.time_from_start = Duration(sec=sec, nanosec=nanosec)

        traj.points = [start, end]
        self.traj_pub.publish(traj)
        self.get_logger().info(f"✅ trajectory送信完了 ({move_time}秒)")

    def solve_ik(self, x, y, z):
        """MoveItのIKで目標関節角度を計算"""
        goal_msg = MoveGroup.Goal()
        request = goal_msg.request
        request.group_name = "arm"
        request.allowed_planning_time = 2.0
        request.num_planning_attempts = 5

        constraints = Constraints()
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "crane_x7_gripper_base_link"
        volume = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.05]
        volume.primitives.append(primitive)
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        volume.primitive_poses.append(pose)
        pos_constraint.constraint_region = volume
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)
        request.goal_constraints.append(constraints)

        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        result = result_future.result().result

        if result.error_code.val == 1:
            traj = result.planned_trajectory.joint_trajectory
            if traj.points:
                return list(traj.points[-1].positions)
        return None

    def camera_to_robot(self, cx, cy, sx, sy):
        """肩基準の相対座標でマッピング"""
        rel_x = cx - sx  # 正=右、負=左
        rel_y = cy - sy  # 正=下、負=上
        scale = 0.001    # 1ピクセル = 1mm（要調整）

        rx = 0.25 + (-rel_y * scale)  # 上に伸ばす→前に
        ry = -rel_x * scale            # 右に伸ばす→右に
        rz = 0.20                      # 高さ固定

        rx = np.clip(rx, 0.10, 0.40)
        ry = np.clip(ry, -0.25, 0.25)
        return rx, ry, rz

    def get_averaged_pos(self, rx, ry, rz):
        """5フレーム平均 + 外れ値除去（±2σ）"""
        self.pos_history.append([rx, ry, rz])
        if len(self.pos_history) > self.N_AVERAGE:
            self.pos_history.pop(0)
        if len(self.pos_history) < self.N_AVERAGE:
            return None  # まだデータ不足

        arr = np.array(self.pos_history)
        averaged = []
        for i in range(3):
            col = arr[:, i]
            mean, std = col.mean(), col.std()
            mask = np.abs(col - mean) < 2 * std
            if mask.sum() == 0:
                averaged.append(mean)
            else:
                averaged.append(col[mask].mean())
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
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape

                wrist = lm[16]
                shoulder = lm[12]  # 右肩

                # 肩基準の手先座標（ピクセル）
                sx = shoulder.x * w
                sy = shoulder.y * h
                wx = wrist.x * w
                wy = wrist.y * h
                rel_x = wx - sx
                rel_y = wy - sy

                self.rel_wrist = (rel_x, rel_y)  # 共有用

                cx, cy = int(wx), int(wy)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                cv2.circle(frame, (int(sx), int(sy)), 10, (255, 0, 0), -1)

                # 画面に表示
                cv2.putText(frame, f"cam rel: ({rel_x:.0f},{rel_y:.0f})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                if self.last_pos:
                    rx, ry, rz = self.last_pos
                    cv2.putText(frame, f"robot: ({rx:.3f},{ry:.3f},{rz:.3f})",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                status = "移動中..." if self.is_moving else "待機中"
                cv2.putText(frame, status,
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            with self.frame_lock:
                self.latest_frame = frame.copy()
            cv2.imshow("MediaPipe", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def move_task(self, rx, ry, rz):
        rel_x, rel_y = self.rel_wrist
        self.get_logger().info(
            f"cam_rel=({rel_x:.0f},{rel_y:.0f}) → robot=({rx:.3f},{ry:.3f},{rz:.3f})")
        joint_positions = self.solve_ik(rx, ry, rz)
        if joint_positions:
            self.send_trajectory(joint_positions[:7], move_time=2.0)  # 0.5→2.0秒に
        else:
            self.get_logger().warn("IK解なし")
        self.is_moving = False

    def timer_callback(self):
        if self.is_moving:
            return

        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return

        lm = results.pose_landmarks.landmark
        h, w, _ = frame.shape

        right_wrist = lm[16]
        left_wrist = lm[15]
        wrist = right_wrist if right_wrist.visibility >= left_wrist.visibility else left_wrist
        shoulder = lm[12]  # 右肩

        if wrist.visibility < 0.3:
            return

        cx = wrist.x * w
        cy = wrist.y * h
        sx = shoulder.x * w
        sy = shoulder.y * h

        rx, ry, rz = self.camera_to_robot(cx, cy, sx, sy)

        # 5フレーム平均・外れ値除去
        averaged = self.get_averaged_pos(rx, ry, rz)
        if averaged is None:
            return
        rx, ry, rz = averaged

        current_pos = np.array([rx, ry, rz])
        if self.last_pos is not None:
            if np.linalg.norm(current_pos - np.array(self.last_pos)) < 0.01:
                return

        self.last_pos = (rx, ry, rz)
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