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
