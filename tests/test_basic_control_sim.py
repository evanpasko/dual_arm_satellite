"""Smoke tests for the bang-bang control demo's headless path."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from dual_arm_satellite.basic_control_sim import main, run_headless


def test_run_headless_moves_toward_target() -> None:
    target = np.array([0.5, 0.0, 0.0])
    times, positions, quats = run_headless(
        target_position_world_m=target,
        duration_s=1.0,
        dt_s=0.02,
        max_thrust_n=50.0,
    )
    assert times.shape[0] == positions.shape[0] == quats.shape[0]
    assert times.shape[0] > 1
    initial_err = float(np.linalg.norm(positions[0] - target))
    final_err = float(np.linalg.norm(positions[-1] - target))
    assert final_err < initial_err


def test_run_headless_saves_plot(tmp_path) -> None:
    out = tmp_path / "bang_bang.png"
    run_headless(
        target_position_world_m=np.array([0.2, 0.0, 0.0]),
        duration_s=0.2,
        dt_s=0.02,
        max_thrust_n=20.0,
        save_plot_path=out,
    )
    assert out.is_file()


def test_main_no_show_requires_duration() -> None:
    with pytest.raises(SystemExit):
        main(["--target", "1.0", "0.0", "0.0", "--no-show"])


def test_main_no_show_runs(capsys) -> None:
    main(
        [
            "--target", "0.3", "0.0", "0.0",
            "--no-show",
            "--duration", "0.4",
            "--dt", "0.02",
            "--max-thrust-n", "40.0",
        ]
    )
    captured = capsys.readouterr()
    assert "Final |position error|" in captured.out
