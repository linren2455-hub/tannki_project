#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
import time

class CranePoseController(Node):
    def __init__(self):
        super().__init__('crane_pose_controller')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("MoveGroupアクションサーバーが見つかりません！")
            return

        self.get_logger().info("✅ MoveGroupアクション準備完了（制約緩め版）")

    def move_to_pose(self, x: float, y: float, z: float):
        goal_msg = MoveGroup.Goal()
        request = goal_msg.request

        request.group_name = "arm"
        request.allowed_planning_time = 10.0
        request.num_planning_attempts = 20
        request.max_velocity_scaling_factor = 0.25
        request.max_acceleration_scaling_factor = 0.25

        constraints = Constraints()

        # 位置制約（到達範囲0.4m以内に抑える）
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "crane_x7_gripper_base_link"

        volume = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.SPHERE
        primitive.dimensions = [0.04]          # 許容範囲を少し広め
        volume.primitives.append(primitive)

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        volume.primitive_poses.append(pose)

        pos_constraint.constraint_region = volume
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        # 姿勢制約を大幅に緩める（手の向きをほぼ無視）
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

        self.get_logger().info(f"🎯 目標送信 → x={x:.3f}, y={y:.3f}, z={z:.3f}")

        # アクション実行
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("目標が拒否されました")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info("✅ 移動成功！")
        else:
            self.get_logger().error(f"❌ 失敗 (error_code: {result.error_code.val})")

def main():
    rclpy.init()
    node = CranePoseController()

    try:
        # 到達範囲0.4m以内の安全な位置（これで動く可能性が高い）
        node.move_to_pose(0.18, 0.00, 0.25)
        time.sleep(6)

        node.move_to_pose(0.15, 0.10, 0.22)
        time.sleep(6)

        node.move_to_pose(0.20, -0.08, 0.24)
        time.sleep(6)

        node.move_to_pose(0.12, 0.05, 0.18)

    except KeyboardInterrupt:
        node.get_logger().info("中断されました")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()