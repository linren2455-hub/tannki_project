# mp_to_moveit.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
import cv2
import mediapipe as mp
import numpy as np
import threading

class CranePoseController(Node):
    def __init__(self):
        super().__init__('crane_pose_controller')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroupサーバーが見つかりません")
            return

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)
        self.cam_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.last_pos = None
        self.is_moving = False  # MoveIt処理中フラグ

        # カメラ表示用の最新フレームを共有
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # カメラスレッド（30FPSで常に更新）
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        # 2.0 → 0.1に変更
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("✅ 準備完了")

    def camera_loop(self):
        """カメラ取得・表示を独立スレッドで30FPS維持"""
        mp_draw = mp.solutions.drawing_utils
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                h, w, _ = frame.shape
                wrist = lm[16]
                cx = wrist.x * w
                cy = wrist.y * h

                mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.circle(frame, (int(cx), int(cy)), 10, (0, 0, 255), -1)

                if self.last_pos is not None:
                    rx, ry, rz = self.last_pos
                    cv2.putText(frame,
                        f"robot: ({rx:.2f},{ry:.2f},{rz:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                status = "移動中..." if self.is_moving else "待機中"
                cv2.putText(frame, status,
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            with self.frame_lock:
                self.latest_frame = frame.copy()

            cv2.imshow("MediaPipe", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def camera_to_robot(self, cx, cy):
        nx = (cx / self.cam_w) - 0.5
        ny = (cy / self.cam_h) - 0.5
        rx = 0.20 + (-ny * 0.20)
        ry = -nx * 0.20
        rz = 0.20
        rx = np.clip(rx, 0.10, 0.35)
        ry = np.clip(ry, -0.20, 0.20)
        rz = np.clip(rz, 0.10, 0.40)
        return rx, ry, rz

    def move_to_pose(self, x, y, z):
        self.is_moving = True
        goal_msg = MoveGroup.Goal()
        request = goal_msg.request
        request.group_name = "arm"
        request.allowed_planning_time = 5.0
        request.num_planning_attempts = 10
        request.max_velocity_scaling_factor = 0.3
        request.max_acceleration_scaling_factor = 0.3

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

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = "base_link"
        ori_constraint.link_name = "crane_x7_gripper_base_link"
        ori_constraint.orientation.w = 1.0
        ori_constraint.absolute_x_axis_tolerance = 0.8
        ori_constraint.absolute_y_axis_tolerance = 0.8
        ori_constraint.absolute_z_axis_tolerance = 0.8
        ori_constraint.weight = 0.1
        constraints.orientation_constraints.append(ori_constraint)

        request.goal_constraints.append(constraints)

        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        goal_handle = future.result()
        if goal_handle and goal_handle.accepted:
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)

        self.is_moving = False

    def timer_callback(self):
        if self.is_moving:  # フラグチェック
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

        if wrist.visibility < 0.3:
            return

        cx = wrist.x * w
        cy = wrist.y * h
        rx, ry, rz = self.camera_to_robot(cx, cy)
        current_pos = np.array([rx, ry, rz])

        if self.last_pos is not None:
            if np.linalg.norm(current_pos - np.array(self.last_pos)) < 0.01:
                return

        self.last_pos = (rx, ry, rz)
        self.is_moving = True  # ← スレッド起動前にフラグを立てる

        self.get_logger().info(f"🎯 move_to_pose 呼ぶ: robot=({rx:.3f},{ry:.3f},{rz:.3f})")
        threading.Thread(
            target=self.move_to_pose,
            args=(rx, ry, rz),
            daemon=True
        ).start()

def main():
    rclpy.init()
    node = CranePoseController()
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