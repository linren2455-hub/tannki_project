#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from geometry_msgs.msg import PoseStamped
import time

class CraneController(Node):
    def __init__(self):
        super().__init__('crane_moveit_controller')
        self.moveit_py = MoveItPy(node_name="moveit_py")
        self.planning_component = self.moveit_py.get_planning_component("arm")

        self.get_logger().info("✅ MoveItPy 準備完了")
        self.get_logger().info("   使用する手先リンク: crane_x7_gripper_base_link")

    def move_to(self, x: float, y: float, z: float):
        """手先位置を指定して計画・実行"""
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose.position.x = x
        pose_goal.pose.position.y = y
        pose_goal.pose.position.z = z
        
        # 姿勢は一旦「真っ直ぐ下向き」に近いクォータニオン（後で調整可）
        pose_goal.pose.orientation.x = 0.0
        pose_goal.pose.orientation.y = 0.7071  # ≈ -90度回転（手先が下を向く方向）
        pose_goal.pose.orientation.z = 0.0
        pose_goal.pose.orientation.w = 0.7071

        self.get_logger().info(f"🎯 目標位置を設定: ({x:.3f}, {y:.3f}, {z:.3f})")

        # 手先リンクを明示的に指定
        success = self.planning_component.set_goal_state(
            pose_stamped_msg=pose_goal, 
            pose_link="crane_x7_gripper_base_link"
        )

        if not success:
            self.get_logger().error("目標状態の設定に失敗")
            return False

        plan_result = self.planning_component.plan()
        if plan_result:
            self.get_logger().info("✅ 計画成功！ 実行します...")
            self.planning_component.execute()
            return True
        else:
            self.get_logger().error("❌ 計画失敗（到達不能、または姿勢が厳しすぎる可能性）")
            return False

def main():
    rclpy.init()
    controller = CraneController()

    try:
        # 安全な初期位置（CRANE-X7の実寸に合わせて調整）
        # まずは高めからスタートして徐々に近づける
        controller.move_to(0.30, 0.00, 0.45)
        time.sleep(4)

        # 少し手前・低めに
        controller.move_to(0.25, 0.05, 0.35)
        time.sleep(4)

        # もう少し動かしてみる例
        controller.move_to(0.20, -0.05, 0.40)

    except KeyboardInterrupt:
        controller.get_logger().info("ユーザーにより中断されました")
    except Exception as e:
        controller.get_logger().error(f"エラー発生: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()