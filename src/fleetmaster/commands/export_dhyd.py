"""CLI command for exporting Fleetmaster HDF5 case groups to standalone .dhyd files."""

from pathlib import Path

import click

from fleetmaster.core.engine import export_hdf5_cases_to_dhyd, list_case_groups_in_hdf5


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


def _resolve_selected_cases(case_name: str | None, available_cases: list[str]) -> list[str]:
    """Resolves selected case names or returns all cases when none is specified."""
    if case_name is not None:
        return [case_name]

    return available_cases


def _validate_cases_exist(selected_cases: list[str], available_cases: list[str], db_path: Path) -> None:
    """Validates that the selected cases exist in the database."""
    missing_cases = [case for case in selected_cases if case not in available_cases]
    if not missing_cases:
        return

    click.echo(f"❌ Case group(s) {missing_cases} were not found in '{db_path}'.", err=True)
    if available_cases:
        _print_cases(available_cases, err=True)
    raise click.Abort()


@click.command(
    name="export-dhyd",
    help=(
        "Export one or more simulation case groups from a Fleetmaster HDF5 database to standalone .dhyd files "
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
    help="Output .dhyd file path. For multiple cases, the case name is appended to the provided base filename.",
)
@click.option("--overwrite", is_flag=True, help="Overwrite the output file if it already exists.")
@click.option("--list-cases", is_flag=True, help="List available case groups and exit.")
def export_dhyd(
    hdf5_file: str, case_name: str | None, output_file: str | None, overwrite: bool, list_cases: bool
) -> None:
    """Export one or more cases from a Fleetmaster HDF5 database to standalone .dhyd files."""
    db_path = Path(hdf5_file)
    available_cases = list_case_groups_in_hdf5(db_path)

    if list_cases:
        _print_cases(available_cases)
        return

    selected_cases = _resolve_selected_cases(case_name, available_cases)
    _validate_cases_exist(selected_cases, available_cases, db_path)

    try:
        exported_paths = export_hdf5_cases_to_dhyd(
            hdf5_file=db_path,
            case_groups=selected_cases,
            output_dhyd_file=output_file,
            overwrite=overwrite,
        )
        if len(exported_paths) == 1:
            click.echo(f"✅ Exported case '{selected_cases[0]}' to '{exported_paths[0]}'.")
        else:
            click.echo(f"✅ Exported {len(exported_paths)} cases to standalone .dhyd files.")
            for exported_path in exported_paths:
                click.echo(f"- {exported_path}")
    except Exception as e:
        click.echo(f"❌ Failed to export .dhyd: {e}", err=True)
        raise click.Abort() from e
