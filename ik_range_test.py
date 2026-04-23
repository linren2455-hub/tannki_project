#!/usr/bin/env python3
# ロボット手先座標の範囲確認用テストコード
import rclpy
import time
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint, BoundingVolume
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from rclpy.action import ActionClient

class IKRangeTest(Node):
    def __init__(self):
        super().__init__('ik_range_test')
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        self.get_logger().info("MoveGroupサーバー待機中...")
        self._action_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info("✅ 準備完了")

    def solve_ik(self, x, y, z):
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
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)
        result = result_future.result().result
        return result.error_code.val == 1

    def run_test(self):
        # ⚠️ 安全範囲内のみ（少しずつ広げる）
        test_points = [
            # まず中央に戻す
            (0.25, 0.00, 0.20, "中央（基準）"),

            # rx（前後）x=0.10〜0.45で確認
            (0.15, 0.00, 0.20, "rx=0.15"),
            (0.25, 0.00, 0.20, "rx=0.25"),
            (0.35, 0.00, 0.20, "rx=0.35"),
            (0.40, 0.00, 0.20, "rx=0.40"),
            (0.45, 0.00, 0.20, "rx=0.45"),

            # 中央に戻す
            (0.25, 0.00, 0.20, "中央（基準）"),

            # rz（高さ）z=0.10〜0.40で確認
            (0.25, 0.00, 0.10, "rz=0.10"),
            (0.25, 0.00, 0.20, "rz=0.20"),
            (0.25, 0.00, 0.30, "rz=0.30"),
            (0.25, 0.00, 0.40, "rz=0.40"),
            (0.10, 0.00, 0.50, "rz=0.50（注意）"),

            # 最後に中央に戻す
            (0.25, 0.00, 0.20, "中央（終了）"),
        ]

        self.get_logger().info("=" * 50)
        self.get_logger().info("IK到達範囲テスト開始")
        self.get_logger().info("⚠️  CRANEの周りに障害物がないことを確認してください")
        self.get_logger().info("=" * 50)
        time.sleep(3.0)  # 確認のための待機

        for x, y, z, label in test_points:
            ok = self.solve_ik(x, y, z)
            result_str = "✅ IK成功" if ok else "❌ IK失敗"
            self.get_logger().info(
                f"{label:15s} x={x:.2f} y={y:+.2f} z={z:.2f} → {result_str}")
            time.sleep(4.0)  # 動作完了を待つ

        self.get_logger().info("=" * 50)
        self.get_logger().info("テスト完了")

def main():
    rclpy.init()
    node = IKRangeTest()
    node.run_test()
    rclpy.shutdown()

if __name__ == '__main__':
    main()