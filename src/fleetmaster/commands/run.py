import glob
import logging
import re
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin

import click
import numpy as np
import yaml
from pydantic import BaseModel, ValidationError

from fleetmaster.core.engine import run_simulation_batch
from fleetmaster.core.settings import MeshConfig, SimulationSettings

logger = logging.getLogger(__name__)

# Define a TypeVar for decorator typing
F = TypeVar("F", bound=Callable[..., Any])

# Set logging levels for external libraries
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("capytaine").setLevel(logging.WARNING)


def _expand_stl_files(stl_files: tuple[str, ...]) -> list[str]:
    """Expand glob patterns for STL files and return a list of paths."""
    if not stl_files:
        return []

    expanded_files = [path for pattern in stl_files for path in glob.glob(pattern)]

    if not expanded_files:
        err_msg = "No files found matching the provided STL patterns: " + ", ".join(stl_files)
        raise click.UsageError(err_msg)
    return expanded_files


def _convert_to_numeric(value_str: str, target_type: type) -> Any:
    """
    Converts a string to a numeric type, with special handling for
    'inf' and 'nan' strings.
    """
    val_lower = value_str.lower().strip()
    if val_lower in ("inf", "infinity", "np.inf"):
        return np.inf
    if val_lower in ("nan", "np.nan"):
        return np.nan
    # Fallback to the original type conversion (e.g., float(value_str))
    return target_type(value_str)


def _parse_range_string(range_str: str) -> list[float] | None:
    """Parses a string like 'start:stop:step' into a list of numbers."""
    if ":" not in range_str:
        return None

    parts = [p.strip() for p in range_str.split(":")]
    if len(parts) > 3:
        return None  # Invalid format

    try:
        # Convert parts to float, providing defaults for missing parts
        start = _convert_to_numeric(parts[0], float) if parts[0] else 0.0
        stop = _convert_to_numeric(parts[1], float) if len(parts) > 1 and parts[1] else None
        step = _convert_to_numeric(parts[2], float) if len(parts) > 2 and parts[2] else 1.0

        if stop is None:
            return None  # Stop is mandatory for a range

        return [float(x) for x in np.arange(start, stop, step)]

    except (ValueError, TypeError):
        return None


def _get_list_item_type(field_annotation: Any) -> type:
    """Extracts the inner type from a list annotation (e.g., float from list[float])."""
    raw_origin = get_origin(field_annotation)
    if raw_origin in (types.UnionType, Union):
        # Find the list type in the union to get its args
        for arg in get_args(field_annotation):
            if get_origin(arg) is list:
                return get_args(arg)[0] if get_args(arg) else str
    elif raw_origin is list:
        return get_args(field_annotation)[0] if get_args(field_annotation) else str
    return str


def _parse_cli_value_string(value_str: str, item_type: type, key: str) -> list[Any]:
    """
    Parses a single string from the CLI, handling ranges, delimited lists,
    and single values.
    """
    # 1. Try to parse as a range
    range_vals = _parse_range_string(value_str)
    if range_vals is not None:
        return range_vals

    # 2. Try to parse as a delimited string
    if re.search(r"[,;\s]", value_str):
        split_values = [v for v in re.split(r"[,;\s]+", value_str.strip()) if v]
        try:
            return [_convert_to_numeric(i, item_type) for i in split_values]
        except (ValueError, TypeError):
            logger.warning(f"Could not convert all delimited values in '{value_str}' for '{key}'.")
            return split_values  # Return as strings on failure

    # 3. Treat as a single value
    try:
        return [_convert_to_numeric(value_str, item_type)]
    except (ValueError, TypeError):
        logger.warning(f"Could not convert single value '{value_str}' for '{key}'.")
        return [value_str]


def _process_cli_args(kwargs: dict[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    """Process CLI arguments, filtering None values and parsing complex list inputs."""
    cli_args = {k: v for k, v in kwargs.items() if v is not None and v != () and v != ""}

    for key, value_tuple in cli_args.items():
        field = model.model_fields.get(key)
        if not field:
            continue

        is_list, _, _ = _get_option_type_info(field.annotation)
        if not (is_list and isinstance(value_tuple, tuple)):
            continue

        list_item_type = _get_list_item_type(field.annotation)

        final_values: list[Any] = []
        for value_str in value_tuple:
            parsed_values = _parse_cli_value_string(value_str, list_item_type, key)
            final_values.extend(parsed_values)

        # De-duplicate values while preserving order
        cli_args[key] = list(dict.fromkeys(final_values))

    return cli_args


def _validate_input_combinations(
    settings_file: str | None, expanded_stl_files: list[str], cli_args: dict[str, Any]
) -> None:
    """Validates the combination of provided command-line arguments."""
    has_settings_file = bool(settings_file)
    has_stl_files = bool(expanded_stl_files)
    has_drafts = "drafts" in cli_args

    if has_settings_file and (has_stl_files or has_drafts):
        err_msg = "A settings file cannot be combined with --stl-files or --drafts."
        raise click.UsageError(err_msg)

    if has_drafts and not has_stl_files:
        err_msg = "--drafts requires a single base STL file to be provided."
        raise click.UsageError(err_msg)

    if has_drafts and len(expanded_stl_files) > 1:
        err_msg = f"--drafts can only be used with a single base STL file, but {len(expanded_stl_files)} were found."
        raise click.UsageError(err_msg)

    if not has_settings_file and not has_stl_files:
        err_msg = "Either a settings file or at least one STL file must be provided."
        raise click.UsageError(err_msg)


def _parse_yaml_ranges(config: dict[str, Any]) -> None:
    """Parses range strings (e.g., '0:10:1') in config for specific keys."""
    for key in ["wave_periods", "wave_directions", "drafts"]:
        if key in config and isinstance(config[key], str):
            parsed_range = _parse_range_string(config[key])
            if parsed_range is not None:
                config[key] = parsed_range


def _resolve_stl_files_paths(stl_items: list[Any], settings_dir: Path) -> list[Any]:
    """Resolves relative STL paths to absolute paths for all supported stl_files item formats."""
    resolved_files: list[Any] = []
    for item in stl_items:
        if isinstance(item, str):
            resolved_files.append(str((settings_dir / item).resolve()) if not Path(item).is_absolute() else item)
        elif isinstance(item, dict) and "file" in item:
            if not Path(item["file"]).is_absolute():
                item["file"] = str((settings_dir / item["file"]).resolve())
            resolved_files.append(item)
        elif isinstance(item, MeshConfig):
            if not Path(item.file).is_absolute():
                item.file = str((settings_dir / item.file).resolve())
            resolved_files.append(item)
        else:
            resolved_files.append(item)
    return resolved_files


def _resolve_paths_in_config(config: dict[str, Any], settings_dir: Path) -> None:
    """Resolves relative paths for 'base_mesh' and 'stl_files' in the config."""
    # Resolve base_mesh path
    if config.get("base_mesh") and not Path(config["base_mesh"]).is_absolute():
        config["base_mesh"] = str((settings_dir / config["base_mesh"]).resolve())

    # Resolve optional standalone export paths
    if config.get("output_dhyd_file") and not Path(config["output_dhyd_file"]).is_absolute():
        config["output_dhyd_file"] = str((settings_dir / config["output_dhyd_file"]).resolve())
    if (
        config.get("output_transformed_stl_directory")
        and not Path(config["output_transformed_stl_directory"]).is_absolute()
    ):
        config["output_transformed_stl_directory"] = str(
            (settings_dir / config["output_transformed_stl_directory"]).resolve()
        )

    # Resolve stl_files paths
    if "stl_files" not in config:
        return

    config["stl_files"] = _resolve_stl_files_paths(config["stl_files"], settings_dir)


def _load_config(settings_file: str | None, cli_args: dict[str, Any]) -> dict[str, Any]:
    """Loads configuration from a YAML file or CLI arguments."""
    if not settings_file:
        # Handle case where settings are purely from CLI
        config: dict[str, Any] = {}
        if cli_args.get("stl_files"):
            config["stl_files"] = cli_args.pop("stl_files")
        return config

    # Handle case with a settings file
    if "base_mesh" in cli_args:
        err_msg = "--base-mesh cannot be used with --settings-file."
        raise click.UsageError(err_msg)

    with open(settings_file) as f:
        config = yaml.safe_load(f) or {}

    _parse_yaml_ranges(config)
    _resolve_paths_in_config(config, Path(settings_file).parent)

    return config


def _load_and_validate_settings(
    settings_file: str | None,
    stl_files: tuple[str, ...],
    kwargs: dict[str, Any],
) -> SimulationSettings:
    """Load settings from file or CLI, merge them, and validate."""
    expanded_stl_files = _expand_stl_files(stl_files)
    cli_args = _process_cli_args(kwargs, SimulationSettings)

    if expanded_stl_files:
        cli_args["stl_files"] = expanded_stl_files

    _validate_input_combinations(settings_file, expanded_stl_files, cli_args)
    config = _load_config(settings_file, cli_args)

    # Ensure base_mesh is an absolute path to an STL file
    if config.get("base_mesh"):
        base_mesh_path_obj = Path(config["base_mesh"])
        if not base_mesh_path_obj.is_absolute():
            # This case should ideally be handled by _resolve_paths_in_config,
            # but adding a safeguard here.
            settings_dir = Path(settings_file).parent if settings_file else Path.cwd()
            config["base_mesh"] = str((settings_dir / base_mesh_path_obj).resolve())

        # Additional check to ensure it's an STL file
        if not config["base_mesh"].lower().endswith(".stl"):
            err_msg = f"The base_mesh '{config['base_mesh']}' is not an STL file. Please provide a valid STL file."
            raise click.UsageError(err_msg)
    elif config.get("stl_files"):
        # If base_mesh is not explicitly set, use the first STL file from stl_files
        # This path should already be resolved by _resolve_paths_in_config
        first_stl_file = config["stl_files"][0]
        if isinstance(first_stl_file, dict):
            config["base_mesh"] = first_stl_file["file"]
        elif isinstance(first_stl_file, MeshConfig):
            config["base_mesh"] = first_stl_file.file
        else:  # it's a string
            config["base_mesh"] = first_stl_file

        logger.info(f"No --base-mesh provided. Using the first STL file as the base mesh: {config['base_mesh']}")

    # The original logic for setting base_mesh if not explicitly provided
    # and using the first STL file is now integrated above.
    # Remove the redundant block.
    # base_mesh_path = config.get("base_mesh")
    # if "stl_files" in config:
    #     all_files_in_config = [item if isinstance(item, str) else item["file"] for item in config["stl_files"]]
    #
    #     if not base_mesh_path and all_files_in_config:
    #         base_mesh_path = all_files_in_config[0]
    #         logger.info(f"No --base-mesh provided. Using the first STL file as the base mesh: {base_mesh_path}")
    #         config["base_mesh"] = base_mesh_path

    config.update(cli_args)

    try:
        settings = SimulationSettings(**config)
    except ValidationError as e:
        click.echo("❌ Error: Invalid settings provided.", err=True)
        click.echo(e, err=True)
        raise click.Abort() from e

    logger.info("Successfully validated simulation settings.")
    logger.debug(f"Running with settings: {settings.model_dump_json(indent=2)}")
    return settings


def _get_option_type_info(raw_option_type: Any) -> tuple[bool, bool, Any]:
    """Inspects a type annotation and returns its CLI characteristics."""
    is_list = False
    is_bool = False
    final_type = None

    origin = get_origin(raw_option_type)
    if origin in (types.UnionType, Union):
        args = get_args(raw_option_type)
        non_none_types = [t for t in args if t is not type(None)]

        for arg in non_none_types:
            if get_origin(arg) is list:
                is_list = True
                # For list types, we don't need to determine final_type, as it will be str
                return is_list, is_bool, None

        # If no list, find the first primitive type
        primitive_args = [t for t in non_none_types if get_origin(t) is not list]
        if primitive_args:
            final_type = primitive_args[0]
    else:
        if get_origin(raw_option_type) is list:
            is_list = True
        else:
            final_type = raw_option_type

    if final_type is bool:
        is_bool = True

    return is_list, is_bool, final_type


def create_cli_options(model: type[BaseModel]) -> Callable[[F], F]:
    """Dynamically create click options from a Pydantic model."""

    def decorator(f: F) -> F:
        # Decorators are applied bottom-up, so reverse the order of fields
        for name, field in reversed(model.model_fields.items()):
            # Skip stl_files as it's handled as a direct argument. base_mesh is handled separately.
            if name == "stl_files":
                continue

            is_list, is_bool, final_type = _get_option_type_info(field.annotation)

            option_name_base = name.replace("_", "-")
            help_text = field.description or f"Set the {name}."

            if is_bool:
                option_name = f"--{option_name_base}/--no-{option_name_base}"
                f = click.option(option_name, default=None, help=help_text)(f)
            elif is_list:
                option_name = f"--{option_name_base}"
                help_text += (
                    " Can be specified multiple times. Accepts single values, comma/space-separated strings, or "
                    "Python-like ranges ('start:stop:step')."
                )
                f = click.option(option_name, type=str, default=None, help=help_text, multiple=True)(f)
            else:
                option_name = f"--{option_name_base}"
                f = click.option(option_name, type=final_type, default=None, help=help_text)(f)
        return f

    return decorator


@click.command(context_settings={"ignore_unknown_options": False})
@click.argument("stl_files", required=False, nargs=-1)
@click.option("--settings-file", type=click.Path(exists=True), help="Path to a YAML settings file.")
@create_cli_options(SimulationSettings)
def run(stl_files: tuple[str, ...], settings_file: str | None, **kwargs: Any) -> None:
    """Runs a set of capytaine simulations based on provided settings."""
    try:
        settings = _load_and_validate_settings(settings_file, stl_files, kwargs)
        run_simulation_batch(settings)
        click.echo("✅ Run completed successfully!")
    except (click.UsageError, click.Abort):
        raise  # Re-raise to let click handle the error and exit
    except Exception as e:
        logger.exception("An unexpected error occurred")
        click.echo(f"❌ An unexpected error occurred: {e}", err=True)
