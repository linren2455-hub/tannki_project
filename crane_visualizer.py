#!/usr/bin/env python3
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque

# ===== 書き換え箇所の説明 =====
# 元コード: rclpy / ROS2 / MoveGroup / JointTrajectory すべて削除
# 書き換え: move_task → plot_position() でmatplotlibにリアルタイム描画
# その他のロジック（キャリブレーション・座標変換・平均化）は一切変更なし
# ================================

class CraneVisualizer:
    def __init__(self):
        # --- キャリブレーション ---
        self.phase = "calib_x"
        self.rel_x_max = 200.0
        self.rel_z_max = 200.0
        self.calib_samples = []
        self.calib_start_time = time.time()

        # --- 姿勢推定 ---
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.cap = cv2.VideoCapture(0)
        self.cam_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.cam_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # --- 状態管理 ---
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_moving = False
        self.last_pos = None
        self.pos_history = []
        self.N_AVERAGE = 5

        # --- 軌跡バッファ（最新100点） ---
        self.trail_x = deque(maxlen=100)
        self.trail_z = deque(maxlen=100)
        self.plot_lock = threading.Lock()

        # --- matplotlib セットアップ ---
        self._setup_plot()

        # --- カメラスレッド起動 ---
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        print("✅ 準備完了")
        print("=" * 40)
        print("【キャリブレーション】")
        print("① 右腕を横に最大に伸ばしてください（5秒間）")
        print("=" * 40)

    # ------------------------------------------------------------------
    # matplotlibセットアップ（元コードのROSパブリッシャ初期化に相当）
    # ------------------------------------------------------------------
    def _setup_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.fig.canvas.manager.set_window_title("crane_x7 手先位置 (XZ平面)")

        self.ax.set_xlim(0.05, 0.50)
        self.ax.set_ylim(0.15, 0.55)
        self.ax.set_xlabel("X [m]  (前方)")
        self.ax.set_ylabel("Z [m]  (高さ)")
        self.ax.set_title("手先位置リアルタイム表示")
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.ax.set_aspect('equal')

        # 可動域の枠
        rect = mpatches.FancyBboxPatch(
            (0.10, 0.20), 0.35, 0.30,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor='steelblue', facecolor='aliceblue', alpha=0.4,
            label="可動域 (0.10–0.45, 0.20–0.50)"
        )
        self.ax.add_patch(rect)

        # 軌跡ライン
        self.trail_line, = self.ax.plot([], [], '-', color='steelblue', alpha=0.4, linewidth=1.5)

        # 現在位置マーカー
        self.current_dot, = self.ax.plot([], [], 'o', color='tomato', markersize=10, zorder=5)

        # 凡例
        self.ax.legend(loc='upper right', fontsize=8)

        # フェーズ表示テキスト
        self.phase_text = self.ax.text(
            0.05, 0.97, "フェーズ: calib_x",
            transform=self.ax.transAxes,
            fontsize=9, verticalalignment='top', color='gray'
        )
        self.coord_text = self.ax.text(
            0.05, 0.91, "",
            transform=self.ax.transAxes,
            fontsize=9, verticalalignment='top', color='dimgray'
        )

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # ★ 書き換え箇所: send_trajectory + solve_ik → plot_position
    # 元コード: MoveGroupでIK解いてJointTrajectoryをパブリッシュ
    # 新コード: ロボット座標をmatplotlibにプロット
    # ------------------------------------------------------------------
    def plot_position(self, rx, ry, rz):
        with self.plot_lock:
            self.trail_x.append(rx)
            self.trail_z.append(rz)

            self.trail_line.set_data(list(self.trail_x), list(self.trail_z))
            self.current_dot.set_data([rx], [rz])

            self.coord_text.set_text(f"x={rx:.3f} m  z={rz:.3f} m  y={ry:.3f} m")
            self.phase_text.set_text(f"フェーズ: {self.phase}")

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        self.is_moving = False

    # ------------------------------------------------------------------
    # ★ 書き換え箇所: move_task
    # 元コード: solve_ik → send_trajectory → is_moving=False
    # 新コード: plot_position → is_moving=False
    # ------------------------------------------------------------------
    def move_task(self, rx, ry, rz):
        self.plot_position(rx, ry, rz)

    # ------------------------------------------------------------------
    # 以下は元コードから変更なし
    # ------------------------------------------------------------------
    def get_rel(self, lm, h, w):
        shoulder = lm[12]
        wrist    = lm[16]
        sx = shoulder.x * w;  sy = shoulder.y * h
        wx = wrist.x    * w;  wy = wrist.y    * h
        rel_x = sx - wx
        rel_z = sy - wy
        return rel_x, rel_z, wrist.visibility

    def camera_to_robot(self, rel_x, rel_z):
        nx = np.clip(rel_x / self.rel_x_max, 0.0, 1.0)
        nz = np.clip(rel_z / self.rel_z_max, 0.0, 1.0)
        rx = 0.10 + nx * (0.45 - 0.10)
        rz = 0.20 + nz * (0.50 - 0.20)
        ry = 0.00
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
            col  = arr[:, i]
            mean = col.mean()
            std  = col.std()
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

    def timer_tick(self):
        """元コードのtimer_callbackに相当。ROSタイマーの代わりにwhileループから呼ぶ"""
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

        if self.phase == "calib_x":
            self.calib_samples.append(rel_x)
            if elapsed > 5.0:
                self.rel_x_max = max(self.calib_samples) if self.calib_samples else 200.0
                print(f"✅ rel_x_max = {self.rel_x_max:.0f}")
                print("② 右腕を上に最大に上げてください（5秒間）")
                self.calib_samples = []
                self.calib_start_time = time.time()
                self.phase = "calib_z"
            return

        if self.phase == "calib_z":
            self.calib_samples.append(rel_z)
            if elapsed > 5.0:
                self.rel_z_max = max(self.calib_samples) if self.calib_samples else 200.0
                print(f"✅ rel_z_max = {self.rel_z_max:.0f}")
                print("=" * 40)
                print("キャリブレーション完了！制御開始します")
                print("=" * 40)
                self.pos_history = []
                self.calib_start_time = time.time()
                self.phase = "control"
            return

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
            print(f"rel=({rel_x:.0f},{rel_z:.0f}) → robot x={rx:.3f} z={rz:.3f}")
            self.is_moving = True
            threading.Thread(
                target=self.move_task,
                args=(rx, ry, rz),
                daemon=True
            ).start()

    def run(self):
        try:
            while True:
                self.timer_tick()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            plt.close('all')
            print("終了しました")


def main():
    node = CraneVisualizer()
    node.run()


if __name__ == '__main__':
    main()
