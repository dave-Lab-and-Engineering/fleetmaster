"""CLI command for exporting a Fleetmaster HDF5 case group to a standalone NetCDF file."""

from pathlib import Path

import click

from fleetmaster.core.engine import export_hdf5_case_to_netcdf, list_case_groups_in_hdf5


def _print_cases(cases: list[str], *, err: bool = False) -> None:
    """Prints available case groups."""
    if err:
        click.echo("Available case groups:", err=True)
        for case in cases:
            click.echo(f"- {case}", err=True)
        return

    if cases:
        click.echo("Available case groups:")
        for case in cases:
            click.echo(f"- {case}")
    else:
        click.echo("No case groups found in this file.")


def _resolve_selected_case(case_name: str | None, available_cases: list[str]) -> str:
    """Resolves a selected case name or aborts with a useful message."""
    if case_name is not None:
        return case_name

    if len(available_cases) == 1:
        return available_cases[0]

    click.echo("❌ Please specify --case. Multiple case groups were found:", err=True)
    _print_cases(available_cases, err=True)
    raise click.Abort()


def _validate_case_exists(selected_case: str, available_cases: list[str], db_path: Path) -> None:
    """Validates that the selected case exists in the database."""
    if selected_case in available_cases:
        return

    click.echo(f"❌ Case group '{selected_case}' was not found in '{db_path}'.", err=True)
    if available_cases:
        _print_cases(available_cases, err=True)
    raise click.Abort()


@click.command(
    name="export-nc",
    help=(
        "Export one simulation case group from a Fleetmaster HDF5 database to a standalone NetCDF file "
        "without rerunning Capytaine."
    ),
)
@click.argument("hdf5_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("--case", "case_name", default=None, help="Name of the case group to export.")
@click.option(
    "--output",
    "output_file",
    default=None,
    type=click.Path(dir_okay=False, resolve_path=True),
    help="Output NetCDF file path. Defaults to <hdf5_stem>_<case_name>.nc.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite the output file if it already exists.")
@click.option("--list-cases", is_flag=True, help="List available case groups and exit.")
def export_nc(
    hdf5_file: str, case_name: str | None, output_file: str | None, overwrite: bool, list_cases: bool
) -> None:
    """Export one case from a Fleetmaster HDF5 database to a standalone NetCDF file."""
    db_path = Path(hdf5_file)
    available_cases = list_case_groups_in_hdf5(db_path)

    if list_cases:
        _print_cases(available_cases)
        return

    selected_case = _resolve_selected_case(case_name, available_cases)
    _validate_case_exists(selected_case, available_cases, db_path)

    output_path = db_path.with_name(f"{db_path.stem}_{selected_case}.nc") if output_file is None else Path(output_file)

    try:
        exported_path = export_hdf5_case_to_netcdf(
            hdf5_file=db_path,
            case_group=selected_case,
            output_netcdf_file=output_path,
            overwrite=overwrite,
        )
        click.echo(f"✅ Exported case '{selected_case}' to '{exported_path}'.")
    except Exception as e:
        click.echo(f"❌ Failed to export NetCDF: {e}", err=True)
        raise click.Abort() from e
