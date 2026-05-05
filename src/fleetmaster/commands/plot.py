"""CLI command for plotting a theta/period grid for one simulation case."""

from __future__ import annotations

from pathlib import Path

import click
import h5py
import numpy as np
import xarray as xr


def _resolve_plot_case_name(hdf5_file: Path, case_name: str | None) -> str:
    """Resolve the case to plot, defaulting only when exactly one case exists."""
    with h5py.File(hdf5_file, "r") as stream:
        available_cases = sorted(name for name in stream if name != "meshes")

    if case_name is not None:
        if case_name not in available_cases:
            msg = f"Case '{case_name}' not found in '{hdf5_file}'. Available cases: {available_cases}"
            raise click.UsageError(msg)
        return case_name

    if len(available_cases) == 1:
        return str(available_cases[0])

    msg = f"Please specify a case name. Available cases: {available_cases}"
    raise click.UsageError(msg)


def _extract_grid_coverage(dataset: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract period, direction and grid coverage from a simulation dataset."""
    if "wave_direction" not in dataset.coords:
        msg = "Dataset does not contain 'wave_direction' coordinates."
        raise ValueError(msg)
    if "period" in dataset.coords:
        periods = np.asarray(dataset["period"].values, dtype=float)
    elif "omega" in dataset.coords:
        periods = 2 * np.pi / np.asarray(dataset["omega"].values, dtype=float)
    else:
        msg = "Dataset does not contain 'period' or 'omega' coordinates."
        raise ValueError(msg)

    directions_deg = np.rad2deg(np.asarray(dataset["wave_direction"].values, dtype=float))

    candidate_names = ("excitation_force", "diffraction_force", "Froude_Krylov_force")
    data_array = next((dataset[name] for name in candidate_names if name in dataset.data_vars), None)
    if data_array is None:
        msg = f"Dataset does not contain any supported force variable: {candidate_names}."
        raise ValueError(msg)

    coverage = xr.apply_ufunc(np.isfinite, np.abs(data_array))
    for dim in tuple(coverage.dims):
        if dim not in {"omega", "wave_direction"}:
            coverage = coverage.any(dim=dim)

    coverage = coverage.transpose("omega", "wave_direction")
    return periods, directions_deg, np.asarray(coverage.values, dtype=bool)


def _default_plot_output_path(hdf5_file: Path, case_name: str) -> Path:
    """Return the default output path for a grid plot."""
    return hdf5_file.with_name(f"{case_name}_grid.png")


def _plot_theta_period_grid(
    periods: np.ndarray, directions_deg: np.ndarray, coverage: np.ndarray, case_name: str, output_path: Path
) -> Path:
    """Create and save a theta/period coverage plot."""
    import matplotlib.pyplot as plt

    fig_width = max(6.0, len(directions_deg) * 0.45)
    fig_height = max(4.0, len(periods) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(coverage.astype(int), origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(directions_deg)))
    ax.set_xticklabels([f"{value:.0f}" for value in directions_deg])
    ax.set_yticks(np.arange(len(periods)))
    ax.set_yticklabels([f"{value:.2f}" for value in periods])
    ax.set_xlabel("Wave direction [deg]")
    ax.set_ylabel("Period [s]")
    ax.set_title(f"Theta/Period Grid: {case_name}")

    ax.set_xticks(np.arange(-0.5, len(directions_deg), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(periods), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    colorbar = fig.colorbar(image, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["missing", "available"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


@click.command(name="plot", help="Plot a quick theta/period grid for one simulation case in an HDF5 database.")
@click.argument("hdf5_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("case_name", required=False)
@click.option(
    "--output",
    "output_file",
    default=None,
    type=click.Path(dir_okay=False, resolve_path=True),
    help="Output PNG path. Defaults to <case_name>_grid.png next to the HDF5 file.",
)
@click.option("--show", is_flag=True, help="Also open the plot interactively after saving it.")
def plot(hdf5_file: str, case_name: str | None, output_file: str | None, show: bool) -> None:
    """Create a quick theta/period coverage plot for one simulation case."""
    db_path = Path(hdf5_file)
    selected_case = _resolve_plot_case_name(db_path, case_name)
    output_path = _default_plot_output_path(db_path, selected_case) if output_file is None else Path(output_file)

    with xr.open_dataset(db_path, group=selected_case, engine="h5netcdf") as dataset:
        periods, directions_deg, coverage = _extract_grid_coverage(dataset)

    saved_path = _plot_theta_period_grid(periods, directions_deg, coverage, selected_case, output_path)
    click.echo(f"✅ Saved grid plot to '{saved_path}'.")

    if show:
        import matplotlib.pyplot as plt

        image = plt.imread(saved_path)
        _fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(saved_path.name)
        plt.show()
