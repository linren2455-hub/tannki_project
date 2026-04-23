import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import cv2
import mediapipe as mp
import numpy as np
import math

def calc_angle_rad(a, b, c):
    """b が頂点の角度をラジアンで返す"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1 = a - b
    v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return math.radians(deg)

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)

        self.elbow_angle = 0.0  # 肘角度（ラジアン）

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # 右腕
            shoulder = [lm[12].x * w, lm[12].y * h]
            elbow    = [lm[14].x * w, lm[14].y * h]
            wrist    = [lm[16].x * w, lm[16].y * h]

            raw_angle = calc_angle_rad(shoulder, elbow, wrist)
            # 肘を伸ばす=π、曲げる=0 に変換
            self.elbow_angle = math.pi - raw_angle

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS)

            # 画面に角度表示
            deg = math.degrees(self.elbow_angle)
            cv2.putText(frame, f"elbow: {deg:.1f}deg",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 2)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [
            'crane_x7_shoulder_fixed_part_pan_joint',
            'crane_x7_shoulder_revolute_part_tilt_joint',
            'crane_x7_upper_arm_revolute_part_twist_joint',
            'crane_x7_upper_arm_revolute_part_rotate_joint',  # 肘
            'crane_x7_lower_arm_fixed_part_joint',
            'crane_x7_lower_arm_revolute_part_joint',
            'crane_x7_wrist_joint',
            'crane_x7_gripper_finger_a_joint',
            'crane_x7_gripper_finger_b_joint',
        ]
        msg.position = [
            0.0,                  # 肩 左右
            0.0,                  # 肩 上下
            0.0,                  # 上腕ひねり
            self.elbow_angle,     # 肘 ← ここだけ動かす
            0.0,                  # 前腕
            0.0,                  # 前腕ひねり
            0.0,                  # 手首
            0.0,                  # グリッパーa
            0.0,                  # グリッパーb
        ]
        self.pub.publish(msg)

        cv2.imshow("Pose", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = PosePublisher()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()