"""Command-line interface (CLI) for Fleetmaster.

This module provides the main entrypoint for the Fleetmaster CLI, a tool for
running hydrodynamic simulations with Capytaine. It uses the `click` library
to define the main command group and handles global options like version and
verbosity.

Functions:
    cli(): The main CLI entrypoint that registers subcommands and sets up logging.

Commands:
    run: Runs a batch of Capytaine simulations based on a settings file or CLI options.
    gui: Launches the Fleetmaster graphical user interface (GUI).
"""

import logging

import click
from rich.logging import RichHandler

from . import __version__
from .commands import export_dhyd, gui, list_command, run, view
from .logging_setup import setup_general_logger

logger = setup_general_logger()


@click.group(
    context_settings={"ignore_unknown_options": False},
    help="A CLI for running hydrodynamic simulations with Fleetmaster.",
    invoke_without_command=True,
)
@click.version_option(
    __version__,
    "--version",
    message="Version: %(version)s",
    help="Show the version and exit.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity level. Use -v for info, -vv for debug.",
)
def cli(verbose: int) -> None:
    """
    The main entrypoint for the Fleetmaster CLI.

    This function configures the application's logging level based on the
    --verbose flag and registers all available subcommands.
    """
    # Get the root logger of the package and set its level
    package_logger = logging.getLogger("fleetmaster")

    # Clear existing handlers to avoid duplicates
    if package_logger.hasHandlers():
        package_logger.handlers.clear()

    # Add RichHandler for beautiful console output
    handler = RichHandler(rich_tracebacks=True)
    package_logger.addHandler(handler)
    package_logger.propagate = False  # Prevent duplicate logging to the root logger

    # Define verbosity levels
    VERBOSITY_DEBUG = 2
    VERBOSITY_INFO = 1

    if verbose >= VERBOSITY_DEBUG:
        log_level = logging.DEBUG
    elif verbose == VERBOSITY_INFO:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    package_logger.setLevel(log_level)
    for h in package_logger.handlers:
        h.setLevel(log_level)

    # If no subcommand is invoked, show the help message.
    ctx = click.get_current_context()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return

    if log_level <= logging.INFO:
        logger.info(
            "🚀 Fleetmaster CLI — ready to start your capytaine simulations.",
        )


cli.add_command(run, name="run")
cli.add_command(export_dhyd, name="export-dhyd")
cli.add_command(gui, name="gui")
cli.add_command(list_command, name="list")
cli.add_command(view, name="view")


if __name__ == "__main__":
    cli()
