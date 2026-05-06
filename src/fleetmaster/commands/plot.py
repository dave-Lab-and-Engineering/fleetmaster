"""CLI command for plotting a theta/period grid for one simulation case."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
import xarray as xr

from fleetmaster.core.engine import create_hyd_from_capytaine_data


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


def _default_hyd_plot_output_dir(hdf5_file: Path) -> Path:
    """Return the default output directory for mafredo hydrodynamic plots."""
    return hdf5_file.parent / "hyd_plots"


def _collect_hyd_plot_figures(figures: Any, existing_figure_numbers: set[int]) -> list[Any]:
    """Collect hydrodynamic plot figures from return values or matplotlib state."""
    if figures is not None:
        figure_items = list(figures) if isinstance(figures, (list, tuple)) else [figures]
        valid_figures = [figure for figure in figure_items if hasattr(figure, "savefig")]
        if valid_figures:
            return valid_figures

    import matplotlib.pyplot as plt

    new_numbers = [num for num in plt.get_fignums() if num not in existing_figure_numbers]
    return [plt.figure(num) for num in new_numbers]


def _save_hyd_plot_figures(figures: list[Any], output_dir: Path, case_name: str) -> list[Path]:
    """Save matplotlib figures returned by mafredo Hyddb1.plot."""
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for index, figure in enumerate(figures, start=1):
        out_path = output_dir / f"{case_name}_hyd_{index:02d}.png"
        figure.savefig(out_path, dpi=180)
        saved_paths.append(out_path)

    return saved_paths


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
@click.option("--hyd-plot/--no-hyd-plot", default=True, help="Also create mafredo hydrodynamic plots.")
@click.option(
    "--save-hyd-plots",
    is_flag=True,
    help="Save the mafredo hydrodynamic plots as PNG files.",
)
@click.option(
    "--hyd-output-dir",
    default=None,
    type=click.Path(file_okay=False, resolve_path=True),
    help="Output directory for saved mafredo hydrodynamic plots.",
)
def plot(
    hdf5_file: str,
    case_name: str | None,
    output_file: str | None,
    show: bool,
    hyd_plot: bool,
    save_hyd_plots: bool,
    hyd_output_dir: str | None,
) -> None:
    """Create a quick theta/period coverage plot for one simulation case."""
    db_path = Path(hdf5_file)
    selected_case = _resolve_plot_case_name(db_path, case_name)
    output_path = _default_plot_output_path(db_path, selected_case) if output_file is None else Path(output_file)

    with xr.open_dataset(db_path, group=selected_case, engine="h5netcdf") as dataset:
        loaded_dataset = dataset.load()
        periods, directions_deg, coverage = _extract_grid_coverage(dataset)

    saved_path = _plot_theta_period_grid(periods, directions_deg, coverage, selected_case, output_path)
    click.echo(f"✅ Saved grid plot to '{saved_path}'.")

    if hyd_plot:
        import matplotlib.pyplot as plt

        hyd = create_hyd_from_capytaine_data(loaded_dataset)
        existing_figure_numbers = set(plt.get_fignums())
        figures = hyd.plot(do_show=False)
        hyd_figures = _collect_hyd_plot_figures(figures, existing_figure_numbers)

        if save_hyd_plots:
            output_dir = _default_hyd_plot_output_dir(db_path) if hyd_output_dir is None else Path(hyd_output_dir)
            saved_hyd_paths = _save_hyd_plot_figures(hyd_figures, output_dir, selected_case)
            if saved_hyd_paths:
                click.echo(f"✅ Saved {len(saved_hyd_paths)} hydrodynamic plot(s) to '{output_dir}'.")
                for saved_hyd_path in saved_hyd_paths:
                    click.echo(f"   - {saved_hyd_path}")
            else:
                click.echo("⚠️ No hydrodynamic plot figures were available to save.")

    if show:
        import matplotlib.pyplot as plt

        image = plt.imread(saved_path)
        _fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(saved_path.name)
        plt.show()
