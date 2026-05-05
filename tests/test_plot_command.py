from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from click import UsageError

from fleetmaster.commands.plot import (
    _default_plot_output_path,
    _extract_grid_coverage,
    _resolve_plot_case_name,
)


def test_extract_grid_coverage_from_excitation_force() -> None:
    dataset = xr.Dataset(
        data_vars={
            "excitation_force": (
                ("omega", "wave_direction", "influenced_dof"),
                np.ones((2, 3, 1), dtype=complex),
            )
        },
        coords={
            "omega": [np.pi, 2 * np.pi],
            "period": ("omega", [2.0, 1.0]),
            "wave_direction": [0.0, np.pi / 2, np.pi],
            "influenced_dof": ["Heave"],
        },
    )

    periods, directions_deg, coverage = _extract_grid_coverage(dataset)

    np.testing.assert_allclose(periods, [2.0, 1.0])
    np.testing.assert_allclose(directions_deg, [0.0, 90.0, 180.0])
    assert coverage.shape == (2, 3)
    assert np.all(coverage)


def test_default_plot_output_path(tmp_path: Path) -> None:
    hdf5_file = tmp_path / "results.hdf5"
    assert _default_plot_output_path(hdf5_file, "case_a") == tmp_path / "case_a_grid.png"


def test_resolve_plot_case_name_single_case(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "db.hdf5"
    dataset = xr.Dataset(coords={"omega": [1.0], "wave_direction": [0.0]})
    dataset.to_netcdf(hdf5_path, mode="w", group="case_a", engine="h5netcdf")

    assert _resolve_plot_case_name(hdf5_path, None) == "case_a"


def test_resolve_plot_case_name_requires_case_when_multiple(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "db.hdf5"
    dataset = xr.Dataset(coords={"omega": [1.0], "wave_direction": [0.0]})
    dataset.to_netcdf(hdf5_path, mode="w", group="case_a", engine="h5netcdf")
    dataset.to_netcdf(hdf5_path, mode="a", group="case_b", engine="h5netcdf")

    with pytest.raises(UsageError, match="Please specify a case name"):
        _resolve_plot_case_name(hdf5_path, None)
