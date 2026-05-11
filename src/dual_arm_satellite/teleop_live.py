"""
CLI teleop: terminal keyboard (raw mode, no echo) + two live figures: ``base_link`` pose vs
time, and a 3D ``base_link``-frame view with base XYZ axes, arm link skeletons, thruster
positions, and thrust directions.

On POSIX TTYs, keys are read from stdin in raw mode so they control the sim instead of the
shell line editor. On Windows without a TTY, keys are read from the matplotlib window when
it has focus.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import mpl_toolkits.mplot3d  # noqa: F401  # registers 3d projection

from physics_sim import BaseLinkPose, PhysicsSimEngine, Thruster
from physics_sim.plotting import plot_base_link_pose
from physics_sim.thrusters_3d import redraw_thrusters_in_base_frame


POS_KEYS = ("q", "w", "e", "r", "t", "y", "u", "i")
NEG_KEYS = ("a", "s", "d", "f", "g", "h", "j", "k")


def _print_teleop_help(*, input_mode: str) -> None:
    if input_mode == "stdin_raw":
        focus = "Keys: this terminal (TTY raw mode — no shell line echo)."
    elif input_mode == "win32":
        focus = "Keys: this console (Windows). Type here while the plot is open."
    else:
        focus = "Keys: click the plot window (stdin is not a TTY; no raw terminal mode)."
    lines = [
        "",
        "=== Teleop ===",
        focus,
        "",
        "  Joints 1–8  +Δ:  " + "  ".join(f"{k}→J{i}" for i, k in enumerate(POS_KEYS, start=1)),
        "  Joints 1–8  −Δ:  " + "  ".join(f"{k}→J{i}" for i, k in enumerate(NEG_KEYS, start=1)),
        "  Left thruster (single impulse):  [",
        "  Right thruster (single impulse): ]",
        "  Reset simulation:                0 (zero)",
        "  Quit:                            close the plot window",
        "",
    ]
    print("\n".join(lines))


def _joint_limits_rad(engine: PhysicsSimEngine, index: int) -> Tuple[float, float]:
    if index < 0 or index > 7:
        raise IndexError("joint index must be 0..7")
    if index < 4:
        j = engine.left_arm.joints[index]
    else:
        j = engine.right_arm.joints[index - 4]
    return j.limit_lower_rad, j.limit_upper_rad


def _bump_joint_clamped(engine: PhysicsSimEngine, index: int, delta_rad: float) -> None:
    q = engine.joint_positions_rad.copy()
    lo, hi = _joint_limits_rad(engine, index)
    q[index] = float(np.clip(q[index] + delta_rad, lo, hi))
    engine.set_joint_positions_rad(q)


def _apply_teleop_char(st: "_TeleopState", ch: str) -> None:
    if not ch:
        return
    if len(ch) != 1:
        return
    k = ch.lower()

    if k in POS_KEYS:
        idx = POS_KEYS.index(k)
        _bump_joint_clamped(st.engine, idx, st.joint_step_rad)
        return
    if k in NEG_KEYS:
        idx = NEG_KEYS.index(k)
        _bump_joint_clamped(st.engine, idx, -st.joint_step_rad)
        return
    if ch == "[":
        st.engine.left_thruster.fire(
            st.engine,
            st.impulse_throttle,
            impulse_window_s=st.dt_s,
        )
        return
    if ch == "]":
        st.engine.right_thruster.fire(
            st.engine,
            st.impulse_throttle,
            impulse_window_s=st.dt_s,
        )
        return
    if k == "0":
        st.reset()
        return


def _os_read_available(fd: int) -> bytes:
    return os.read(fd, 4096)


def _stdin_raw_reader(
    char_queue: "queue.Queue[str]",
    stop: threading.Event,
) -> None:
    """POSIX: read single keys without echo; push one-char strings to ``char_queue``."""
    if sys.platform == "win32" or not sys.stdin.isatty():
        return
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except (termios.error, AttributeError):
        return
    try:
        tty.setraw(fd, termios.TCSANOW)
        while not stop.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            data = _os_read_available(fd)
            if not data:
                continue
            try:
                char_queue.put(data.decode("utf-8", errors="ignore"))
            except Exception:
                pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (termios.error, AttributeError):
            pass


def _win32_reader(char_queue: "queue.Queue[str]", stop: threading.Event) -> None:
    try:
        import msvcrt
    except ImportError:
        return
    while not stop.is_set():
        if msvcrt.kbhit():
            chb = msvcrt.getch()
            if chb in (b"\x03",):
                break
            try:
                char_queue.put(chb.decode("utf-8", errors="ignore"))
            except Exception:
                pass
        else:
            stop.wait(0.05)


def _start_keyboard_thread(
    char_queue: "queue.Queue[str]", stop: threading.Event
) -> tuple[Optional[threading.Thread], str]:
    """
    Start a thread that fills ``char_queue`` with teleop key characters.

    Returns ``(thread or None, input_mode)`` where ``input_mode`` is
    ``"stdin_raw"`` | ``"win32"`` | ``"mpl"``.
    """
    if sys.platform == "win32":
        t = threading.Thread(target=_win32_reader, args=(char_queue, stop), daemon=True)
        t.start()
        return t, "win32"
    if sys.stdin.isatty():
        t = threading.Thread(
            target=_stdin_raw_reader, args=(char_queue, stop), daemon=True
        )
        t.start()
        return t, "stdin_raw"
    return None, "mpl"


@dataclass(eq=False)
class _TeleopState:
    engine: PhysicsSimEngine
    joint_step_rad: float
    sim_time_s: float = 0.0
    dt_s: float = 0.02
    impulse_throttle: float = 1.0
    max_points: int = 6000
    times: List[float] = field(default_factory=list)
    pos: List[np.ndarray] = field(default_factory=list)
    quat: List[np.ndarray] = field(default_factory=list)
    initial_pose: BaseLinkPose = field(init=False)
    initial_joints: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.initial_pose = self.engine.base_pose.copy()
        self.initial_joints = self.engine.joint_positions_rad.copy()

    def reset(self) -> None:
        self.engine.reset(
            initial_base_pose=self.initial_pose,
            initial_joint_positions_rad=self.initial_joints,
            clear_pose_log=True,
        )
        self.engine.set_linear_velocity_world_m_s(np.zeros(3, dtype=float))
        self.sim_time_s = 0.0
        self.times.clear()
        self.pos.clear()
        self.quat.clear()
        self.engine.record_base_pose(0.0)
        self.append_sample()

    def append_sample(self) -> None:
        pose = self.engine.base_pose
        self.times.append(self.sim_time_s)
        self.pos.append(pose.position_m.copy())
        self.quat.append(pose.quaternion_xyzw.copy())
        if len(self.times) > self.max_points:
            self.times.pop(0)
            self.pos.pop(0)
            self.quat.pop(0)

    def on_mpl_key(self, event) -> None:
        """Used only in ``mpl`` input mode: figure window must have focus."""
        if event.key is None:
            return
        if event.key in ("[", "bracketleft"):
            _apply_teleop_char(self, "[")
            return
        if event.key in ("]", "bracketright"):
            _apply_teleop_char(self, "]")
            return
        k = event.key
        if isinstance(k, str) and len(k) == 1:
            _apply_teleop_char(self, k)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teleop from terminal (raw, no echo) + live base_link pose plot."
    )
    p.add_argument(
        "--joint-step",
        type=float,
        default=np.deg2rad(2.0),
        help="radians per joint key press (default: 2°)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="simulation timestep (s); also used as thruster impulse window for [ ]",
    )
    p.add_argument(
        "--interval-ms",
        type=int,
        default=20,
        help="animation frame interval in ms (default: 20)",
    )
    p.add_argument(
        "--max-plot-points",
        type=int,
        default=6000,
        help="rolling history length for the live plot",
    )
    p.add_argument(
        "--left-max-thrust-n",
        type=float,
        default=20.0,
        help="left thruster peak force (N)",
    )
    p.add_argument(
        "--right-max-thrust-n",
        type=float,
        default=20.0,
        help="right thruster peak force (N)",
    )
    p.add_argument(
        "--thruster-throttle",
        type=float,
        default=1.0,
        help="throttle 0..1 for [ and ] impulses (default: 1)",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="after closing the plot, save base_link pose vs time to this file (e.g. teleop_pose.png)",
    )
    p.add_argument(
        "--thruster-arrow-m",
        type=float,
        default=0.22,
        help="length of thrust direction arrows in the 3D base-frame view (m)",
    )
    p.add_argument(
        "--base-axes-m",
        type=float,
        default=0.12,
        help="length of +x/+y/+z axis segments from base origin in the 3D view (m)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    throttle = float(np.clip(args.thruster_throttle, 0.0, 1.0))

    engine = PhysicsSimEngine.from_urdf(
        left_thruster=Thruster(
            side="left",
            max_thrust_n=args.left_max_thrust_n,
            impulse_window_s=args.dt,
        ),
        right_thruster=Thruster(
            side="right",
            max_thrust_n=args.right_max_thrust_n,
            impulse_window_s=args.dt,
        ),
    )

    st = _TeleopState(
        engine=engine,
        joint_step_rad=args.joint_step,
        dt_s=args.dt,
        impulse_throttle=throttle,
        max_points=args.max_plot_points,
    )

    char_queue: "queue.Queue[str]" = queue.Queue()
    stop_reader = threading.Event()
    reader_thread, input_mode = _start_keyboard_thread(char_queue, stop_reader)

    _print_teleop_help(input_mode=input_mode)

    fig, (ax_p, ax_q) = plt.subplots(2, 1, sharex=True, figsize=(10, 7), constrained_layout=True)
    try:
        fig.canvas.manager.set_window_title("Teleop — base_link pose")
    except Exception:
        pass
    fig.suptitle("base_link pose (teleop keys in this terminal when raw mode is active)")
    (lx,) = ax_p.plot([], [], label="x (m)")
    (ly,) = ax_p.plot([], [], label="y (m)")
    (lz,) = ax_p.plot([], [], label="z (m)")
    ax_p.set_ylabel("position (m)")
    ax_p.legend(loc="upper right")
    ax_p.grid(True, alpha=0.3)

    (lqx,) = ax_q.plot([], [], label="qx")
    (lqy,) = ax_q.plot([], [], label="qy")
    (lqz,) = ax_q.plot([], [], label="qz")
    (lqw,) = ax_q.plot([], [], label="qw")
    ax_q.set_xlabel("time (s)")
    ax_q.set_ylabel("quaternion (xyzw)")
    ax_q.legend(loc="upper right")
    ax_q.grid(True, alpha=0.3)

    if input_mode == "mpl":
        fig.canvas.mpl_connect("key_press_event", st.on_mpl_key)

    fig_3d = plt.figure(figsize=(7, 6))
    try:
        fig_3d.canvas.manager.set_window_title("Teleop — thrusters (base frame)")
    except Exception:
        pass
    ax_3d = fig_3d.add_subplot(111, projection="3d")

    st.engine.record_base_pose(0.0)
    st.append_sample()
    redraw_thrusters_in_base_frame(
        ax_3d,
        st.engine,
        arrow_length_m=args.thruster_arrow_m,
        base_axes_length_m=args.base_axes_m,
    )

    def _drain_keyboard_queue() -> None:
        while True:
            try:
                block = char_queue.get_nowait()
            except queue.Empty:
                break
            for ch in block:
                _apply_teleop_char(st, ch)

    def _frame(_i: int):
        _drain_keyboard_queue()
        st.engine.step(st.dt_s)
        st.sim_time_s += st.dt_s
        st.engine.record_base_pose(st.sim_time_s)
        st.append_sample()
        if not st.times:
            return (lx, ly, lz, lqx, lqy, lqz, lqw)
        t = np.asarray(st.times, dtype=float)
        P = np.stack(st.pos, axis=0)
        Q = np.stack(st.quat, axis=0)
        lx.set_data(t, P[:, 0])
        ly.set_data(t, P[:, 1])
        lz.set_data(t, P[:, 2])
        lqx.set_data(t, Q[:, 0])
        lqy.set_data(t, Q[:, 1])
        lqz.set_data(t, Q[:, 2])
        lqw.set_data(t, Q[:, 3])
        ax_p.relim()
        ax_p.autoscale_view()
        ax_q.relim()
        ax_q.autoscale_view()
        redraw_thrusters_in_base_frame(
            ax_3d,
            st.engine,
            arrow_length_m=args.thruster_arrow_m,
            base_axes_length_m=args.base_axes_m,
        )
        fig_3d.canvas.draw_idle()
        return (lx, ly, lz, lqx, lqy, lqz, lqw)

    def _on_figure_close(_evt) -> None:
        stop_reader.set()

    fig.canvas.mpl_connect("close_event", _on_figure_close)
    fig_3d.canvas.mpl_connect("close_event", _on_figure_close)

    anim = FuncAnimation(
        fig,
        _frame,
        interval=args.interval_ms,
        cache_frame_data=False,
    )
    try:
        plt.show()
    finally:
        stop_reader.set()
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)
        if args.save is not None and st.times:
            t = np.asarray(st.times, dtype=float)
            P = np.stack(st.pos, axis=0)
            Q = np.stack(st.quat, axis=0)
            plot_base_link_pose(
                t,
                P,
                Q,
                title="base_link pose (teleop session)",
                save_path=args.save,
                show=False,
            )
            print(f"Saved plot to {args.save.resolve()}")
        elif args.save is not None:
            print("No pose samples recorded; skipping --save.")


if __name__ == "__main__":
    main()
