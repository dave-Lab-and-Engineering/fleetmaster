from __future__ import annotations

import numpy as np

from fleetmaster.core.settings import MeshConfig, SimulationSettings


def _make_settings(**overrides):
    base = {
        "stl_files": [MeshConfig(file="dummy.stl")],
        "wave_periods": [1.0],
        "wave_directions": [0.0],
    }
    base.update(overrides)
    return SimulationSettings(**base)


def test_water_depth_accepts_dot_inf_scalar() -> None:
    settings = _make_settings(water_depth=".inf")
    assert settings.water_depth == np.inf


def test_water_depth_accepts_inf_scalar() -> None:
    settings = _make_settings(water_depth="inf")
    assert settings.water_depth == np.inf


def test_water_depth_accepts_dot_inf_in_list() -> None:
    settings = _make_settings(water_depth=[1.8, ".inf"])

    assert isinstance(settings.water_depth, list)
    assert settings.water_depth[0] == 1.8
    assert settings.water_depth[1] == np.inf


def test_water_depth_accepts_inf_in_list() -> None:
    settings = _make_settings(water_depth=[1.8, "inf"])

    assert isinstance(settings.water_depth, list)
    assert settings.water_depth[0] == 1.8
    assert settings.water_depth[1] == np.inf


def test_water_level_accepts_dot_inf_scalar() -> None:
    settings = _make_settings(water_level=".inf")
    assert settings.water_level == np.inf


def test_water_level_accepts_inf_scalar() -> None:
    settings = _make_settings(water_level="inf")
    assert settings.water_level == np.inf


def test_heading_symmetry_without_grid_symmetry_logs_warning(caplog) -> None:
    with caplog.at_level("WARNING"):
        settings = _make_settings(heading_symmetry=True, grid_symmetry=False)

    assert settings.heading_symmetry is True
    assert "heading_symmetry is enabled while grid_symmetry is disabled" in caplog.text


def test_heading_symmetry_with_grid_symmetry_no_warning(caplog) -> None:
    with caplog.at_level("WARNING"):
        settings = _make_settings(heading_symmetry=True, grid_symmetry=True)

    assert settings.heading_symmetry is True
    assert "heading_symmetry is enabled while grid_symmetry is disabled" not in caplog.text


def test_wave_periods_accepts_compound_range_string() -> None:
    settings = _make_settings(wave_periods="1:11.1:1,12:30:2")

    assert settings.wave_periods == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        14.0,
        16.0,
        18.0,
        20.0,
        22.0,
        24.0,
        26.0,
        28.0,
    ]


def test_wave_directions_accepts_compound_range_string() -> None:
    settings = _make_settings(wave_directions="0:91:45,180:271:45")

    assert settings.wave_directions == [0.0, 45.0, 90.0, 180.0, 225.0, 270.0]
