from __future__ import annotations

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import capytaine as cpt
import h5py
import numpy as np
import numpy.typing as npt
import trimesh
import xarray as xr
from mafredo import Hyddb1, Rao
from mafredo.helpers import MotionMode, MotionModeToStr, dof_names_to_numbers

from .exceptions import LidAndSymmetryEnabledError
from .io import load_meshes_from_hdf5
from .settings import MESH_GROUP_NAME, MeshConfig, SimulationSettings

logger = logging.getLogger(__name__)


@dataclass
class EngineMesh:
    """Represents a mesh object with its configuration."""

    name: str
    mesh: trimesh.Trimesh
    config: MeshConfig

    def copy(self) -> EngineMesh:
        """Creates a deep copy of the EngineMesh instance."""
        return EngineMesh(
            name=self.name,
            mesh=self.mesh.copy(),
            config=self.config.model_copy(deep=True),
        )


def make_database(
    body: Any,
    omegas: list | npt.NDArray[np.float64],
    wave_directions: list | npt.NDArray[np.float64],
    water_depth: float,
    water_level: float,
    forward_speed: float,
) -> Any:
    """Create a dataset of BEM results for a given body and conditions."""
    bem_solver = cpt.BEMSolver()
    problems: list[Any] = []
    logger.debug(f"Solving for water_depth={water_depth} water_level={water_level} forward_speed={forward_speed}")
    for omega in omegas:
        logger.debug(f"RadiationProblem and DiffractionProblem for omega {omega}")
        problems.extend(
            cpt.RadiationProblem(
                omega=omega,
                body=body,
                radiating_dof=dof,
                water_depth=water_depth,
                free_surface=water_level,
                forward_speed=forward_speed,
            )
            for dof in body.dofs
        )
        for wave_direction in wave_directions:
            logger.debug(f"DiffractionProblem for wave_direction {wave_direction} ")
            problems.append(
                cpt.DiffractionProblem(
                    omega=omega,
                    body=body,
                    wave_direction=wave_direction,
                    water_depth=water_depth,
                    free_surface=water_level,
                    forward_speed=forward_speed,
                )
            )

    results = [bem_solver.solve(problem) for problem in problems]

    database = cpt.assemble_dataset(results)

    # Rename phony dimensions that might be created by capytaine.
    # Based on user feedback, we expect phony_dim_0, 1, and 2.
    rename_map = {
        "phony_dim_0": "i",  # Likely a 3x3 matrix row
        "phony_dim_1": "j",  # Likely a 3x3 matrix column
        "phony_dim_2": "mesh_nodes",  # Likely a mesh-related dimension
    }
    # Filter for dims that actually exist in the dataset to avoid errors
    dims_to_rename = {k: v for k, v in rename_map.items() if k in database.dims}
    if dims_to_rename:
        logger.info(f"Renaming phony dimensions: {dims_to_rename}")
        database = database.rename_dims(dims_to_rename)

    for coord_name, coord_data in database.coords.items():
        if hasattr(coord_data.dtype, "categories"):  # Check for categorical dtype without pandas
            logger.debug(f"Converting coordinate '{coord_name}' from Categorical to unicode dtype.")
            database[coord_name] = database[coord_name].astype("U")

    return database


def _setup_output_file(settings: SimulationSettings) -> Path:
    """
    Determine the output directory and prepare the HDF5 file.
    Deletes the file if it already exists.

    If the output_directory is given in de settings file, the hd5 file is store in this directory.
    If no output_directory is, the hdf5 is stored next to the settings file itself.

    Returns:
        The full path to the HDF5 output file.
    """
    if not settings.stl_files:
        msg = "No STL files provided to process."
        raise ValueError(msg)

    first_stl_entry = settings.stl_files[0]
    if isinstance(first_stl_entry, dict):
        first_stl_path = first_stl_entry["file"]
    elif isinstance(first_stl_entry, str):
        first_stl_path = first_stl_entry
    else:  # Is an object with a .file attribute
        first_stl_path = first_stl_entry.file

    output_dir = Path(settings.output_directory) if settings.output_directory else Path(first_stl_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / settings.output_hdf5_file
    if output_file.exists() and settings.overwrite_meshes:
        logger.warning(f"Output file {output_file} already exists and will be overwritten as overwrite_meshes is True.")
        output_file.unlink()
    return output_file


def _prepare_trimesh_geometry(stl_file: str, mesh_config: MeshConfig | None = None) -> trimesh.Trimesh:
    """
    Loads an STL file and applies the specified translation and rotation.

    The rotation (roll, pitch, yaw) is performed around the center of gravity (cog)
    if specified in the mesh_config. If no cog is specified, the mesh's geometric
    center of mass is used as the rotation point. If no configuration is given,
    the untransformed loaded mesh is returned.

    Returns:
        A trimesh.Trimesh object representing the transformed geometry.
    """
    mesh = trimesh.load_mesh(stl_file)

    if mesh_config is None:
        return mesh

    return _apply_mesh_translation_and_rotation(
        mesh=mesh,
        translation_vector=mesh_config.translation,
        rotation_vector_deg=mesh_config.rotation,
        cog=mesh_config.cog,
    )


def _apply_mesh_translation_and_rotation(
    mesh: trimesh.Trimesh,
    translation_vector: npt.NDArray[np.float64] | list | None = None,
    rotation_vector_deg: npt.NDArray[np.float64] | list | None = None,
    cog: npt.NDArray[np.float64] | list | None = None,
) -> trimesh.Trimesh:
    """Apply a translation and rotation to a mesh object."""
    translation_vector = np.asarray(translation_vector) if translation_vector is not None else np.zeros(3)
    rotation_vector_deg = np.asarray(rotation_vector_deg) if rotation_vector_deg is not None else np.zeros(3)

    has_translation = np.any(translation_vector != 0)
    has_rotation = np.any(rotation_vector_deg != 0)

    if not has_translation and not has_rotation:
        return mesh

    # Start with an identity matrix (no transformation)
    # The affine matrix is defined as:
    # [ R R R T ]
    # [ R R R T ]
    # [ R R R T ]
    # [ 0 0 0 S ]
    # In our case the scaling factor always S = 1.
    transform_matrix = np.identity(4)

    # Apply rotation around the COG if specified
    if has_rotation:
        # Determine the point of rotation
        if cog is not None:
            rotation_point = np.asarray(cog)
            logger.debug(f"Using specified COG {rotation_point} as rotation point.")
        else:
            rotation_point = mesh.center_mass
            logger.debug(f"Using geometric center of mass {rotation_point} as rotation point.")

        # Create rotation matrix for rotation around the specified point
        rotation_vector_rad = np.deg2rad(rotation_vector_deg)
        rotation_matrix = trimesh.transformations.euler_matrix(
            rotation_vector_rad[0], rotation_vector_rad[1], rotation_vector_rad[2], "sxyz"
        )
        # The full rotation transform is: Translate to origin, Rotate, Translate back
        # note that C = A @ B is identical to C = np.matmul(A, B)
        rotation_transform = (
            trimesh.transformations.translation_matrix(rotation_point)
            @ rotation_matrix
            @ trimesh.transformations.translation_matrix(-rotation_point)
        )
        transform_matrix = rotation_transform @ transform_matrix

    # Apply the final translation if specified
    if has_translation:
        translation_matrix = trimesh.transformations.translation_matrix(translation_vector)
        transform_matrix = translation_matrix @ transform_matrix

    logger.debug(f"Applying transformation matrix:\n{transform_matrix}")
    mesh.apply_transform(transform_matrix)

    return mesh


def _prepare_capytaine_body(
    engine_mesh: EngineMesh,
    lid: bool,
    grid_symmetry: bool,  # Added from SimulationSettings
    water_level: float = 0.0,
) -> tuple[Any, trimesh.Trimesh | None]:
    """
    Configures a Capytaine FloatingBody from a pre-prepared trimesh object.

    The `center_of_mass` for Capytaine is determined by `mesh_config.cog`,
    falling back to the mesh's geometric center of mass. If no cog is given in
    the settings file, the geometric center of mass is used.
    """
    cog = None

    if engine_mesh.config.cog:
        cog = np.array(engine_mesh.config.cog)
        logger.debug(f"Using specified COG {cog} as the center of mass for Capytaine.")
    else:
        # If no local_origin is specified, use the center of mass of the (already translated) source_mesh.
        cog = engine_mesh.mesh.center_mass
        logger.debug(f"Using geometric center of mass {cog} of the translated mesh for Capytaine.")

    # Save the transformed mesh to a temporary file and load it with Capytaine.
    # This is more robust than creating a cpt.Mesh from vertices/faces directly.
    # We use NamedTemporaryFile to handle creation and cleanup automatically.
    temp_path = None
    try:
        # Write to the temporary file.
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            engine_mesh.mesh.export(temp_file, file_type="stl")
            logger.debug(f"Exported transformed mesh to temporary file: {temp_path}")

        # Read from the now-closed temporary file. This avoids race conditions.
        hull_mesh = cpt.load_mesh(str(temp_path), name=engine_mesh.name)

    finally:
        # Ensure the temporary file is always deleted, even if an error occurs.
        if temp_path and temp_path.exists():
            logger.debug(f"Deleting temporary file: {temp_path}")
            temp_path.unlink()

    # Configure the Capytaine FloatingBody
    lid_mesh = hull_mesh.generate_lid(z=-0.01) if lid else None
    if grid_symmetry:
        logger.debug("Applying grid symmetery")
        hull_mesh = cpt.ReflectionSymmetricMesh(hull_mesh, plane=cpt.xOz_Plane)

    boat = cpt.FloatingBody(mesh=hull_mesh, lid_mesh=lid_mesh, center_of_mass=cog)
    boat.keep_immersed_part(free_surface=water_level)

    # Check for empty mesh after keep_immersed_part
    # Use np.asarray(...).size to ensure we have a numpy ndarray (satisfies the type checker)
    if np.asarray(boat.mesh.vertices).size == 0 or np.asarray(boat.mesh.faces).size == 0:
        logger.warning("Resulting mesh is empty after keep_immersed_part. Check if water_level is above the mesh.")

    # Important: do this step after keep_immersed_part in order to keep the body constent with the cut mesh
    boat.add_all_rigid_body_dofs()

    # Extract the final mesh that Capytaine will use for the database. After keep_immersed_part,
    # boat.mesh contains the correct vertices and faces for both regular and symmetric meshes.
    final_mesh_trimesh = trimesh.Trimesh(vertices=boat.mesh.vertices, faces=boat.mesh.faces)

    return boat, final_mesh_trimesh


def _get_mesh_hash(mesh_to_add: trimesh.Trimesh) -> tuple[bytes, str]:
    """Exports mesh and computes its SHA256 hash."""
    new_stl_content = mesh_to_add.export(file_type="stl")
    if isinstance(new_stl_content, str):
        new_stl_content = new_stl_content.encode()
    elif not isinstance(new_stl_content, bytes):
        msg = f"Unsupported type from trimesh export: {type(new_stl_content)}"
        raise TypeError(msg)
    new_hash = hashlib.sha256(new_stl_content).hexdigest()
    return new_stl_content, new_hash


def _handle_existing_mesh(f: Any, mesh_group_path: str, new_hash: str, overwrite: bool, mesh_name: str) -> bool:
    """
    Checks for existing mesh in the HDF5 file and decides whether to skip or overwrite.

    Returns:
        True if the operation should be skipped, False otherwise.
    """
    if mesh_group_path in f:
        existing_group = f[mesh_group_path]
        stored_hash = existing_group.attrs.get("sha256")

        if stored_hash == new_hash:
            logger.info(f"Mesh '{mesh_name}' has the same SHA256 hash. Skipping.")
            return True  # Skip

        if not overwrite:
            logger.warning(
                f"Mesh '{mesh_name}' is different from the one in the database (SHA256 mismatch). "
                "Use --overwrite-meshes to overwrite."
            )
            return True  # Skip

        logger.warning(f"Overwriting existing mesh '{mesh_name}' as --overwrite-meshes is specified.")
        del f[mesh_group_path]
    return False  # Don't skip


def _write_mesh_to_group(
    group: Any,
    mesh_to_add: trimesh.Trimesh,
    mesh_config: MeshConfig | None,
    new_hash: str,
    new_stl_content: bytes,
) -> None:
    """Writes mesh properties, metadata, and content to an HDF5 group."""
    # Calculate geometric properties from the new mesh content
    fingerprint_attrs = {
        "volume": mesh_to_add.volume,
        "cog_x": mesh_to_add.center_mass[0],
        "cog_y": mesh_to_add.center_mass[1],
        "cog_z": mesh_to_add.center_mass[2],
        "bbox_lx": mesh_to_add.bounding_box.extents[0],
        "bbox_ly": mesh_to_add.bounding_box.extents[1],
        "bbox_lz": mesh_to_add.bounding_box.extents[2],
    }
    for key, value in fingerprint_attrs.items():
        group.attrs[key] = value
    logger.debug(f"  - Wrote {len(fingerprint_attrs)} fingerprint attributes.")

    # Add hash and original file name as attributes
    group.attrs["sha256"] = new_hash

    if mesh_config:
        if mesh_config.translation:
            group.attrs["translation"] = mesh_config.translation
        if mesh_config.rotation:
            group.attrs["rotation"] = mesh_config.rotation
        if mesh_config.cog:
            group.attrs["cog"] = mesh_config.cog

    group.create_dataset("inertia_tensor", data=mesh_to_add.moment_inertia)
    logger.debug("  - Wrote dataset: inertia_tensor")

    # Store the binary content of the final, transformed STL
    group.create_dataset("stl_content", data=np.void(new_stl_content))
    logger.debug("  - Wrote dataset: stl_content")


def add_mesh_to_database(
    output_file: Path,
    mesh_to_add: trimesh.Trimesh,
    mesh_name: str,
    overwrite: bool = False,
    mesh_config: MeshConfig | None = None,
) -> None:
    """
    Adds a mesh and its geometric properties to the HDF5 database under the MESH_GROUP_NAME.

    Checks if the mesh already exists by comparing SHA256 hashes.
    If the data is different, it will either raise a warning or overwrite if `overwrite` is True.

    Args:
        mesh_to_add: The trimesh object of the mesh to be added.
    """
    if not isinstance(mesh_to_add, trimesh.Trimesh) or mesh_to_add.is_empty:
        logger.warning(f"Attempted to add an empty or invalid mesh named '{mesh_name}' to the database. Skipping.")
        return

    mesh_group_path = f"{MESH_GROUP_NAME}/{mesh_name}"
    new_stl_content, new_hash = _get_mesh_hash(mesh_to_add)

    with h5py.File(output_file, "a") as f:
        if _handle_existing_mesh(f, mesh_group_path, new_hash, overwrite, mesh_name):
            return

        logger.debug(f"Adding mesh '{mesh_name}' to group '{MESH_GROUP_NAME}'...")
        group = f.create_group(mesh_group_path)
        _write_mesh_to_group(group, mesh_to_add, mesh_config, new_hash, new_stl_content)


def _format_value_for_name(value: float) -> str:
    """Formats a float for use in a group name."""
    if value == np.inf:
        return "inf"
    if value == int(value):
        return str(int(value))
    return f"{value:.1f}"


def _generate_case_group_name(mesh_name: str, water_depth: float, water_level: float, forward_speed: float) -> str:
    """Generates a descriptive group name for a specific simulation case."""
    wd = _format_value_for_name(water_depth)
    wl = _format_value_for_name(water_level)
    fs = _format_value_for_name(forward_speed)
    return f"{mesh_name}_wd_{wd}_wl_{wl}_fs_{fs}"


def _is_single_value(value: float | list[float]) -> bool:
    """Returns True when a setting contains exactly one numeric value."""
    return not isinstance(value, list) or len(value) == 1


def _validate_single_case_netcdf_export(settings: SimulationSettings, mesh_count: int) -> None:
    """Validates that NetCDF export is only requested for one mesh and one case."""
    if settings.output_netcdf_file is None:
        return

    if mesh_count != 1:
        msg = "'output_netcdf_file' requires exactly one mesh in 'stl_files'."
        raise ValueError(msg)

    if settings.drafts:
        msg = "'output_netcdf_file' is not supported together with 'drafts'."
        raise ValueError(msg)

    if not _is_single_value(settings.water_depth):
        msg = "'output_netcdf_file' requires a single 'water_depth' value."
        raise ValueError(msg)
    if not _is_single_value(settings.water_level):
        msg = "'output_netcdf_file' requires a single 'water_level' value."
        raise ValueError(msg)
    if not _is_single_value(settings.forward_speed):
        msg = "'output_netcdf_file' requires a single 'forward_speed' value."
        raise ValueError(msg)


def _write_case_to_netcdf(database: xr.Dataset, netcdf_file: Path, mesh_name: str) -> None:
    """Writes one case to a standalone NetCDF file in Capytaine-style layout."""
    from capytaine.io.xarray import separate_complex_values

    standalone = separate_complex_values(database)
    standalone.attrs["stl_mesh_name"] = mesh_name

    for var_name in standalone.variables:
        if hasattr(standalone[var_name].dtype, "name") and standalone[var_name].dtype.name == "category":
            standalone[var_name] = standalone[var_name].astype(str)

    encoding: dict[str, dict[str, str]] = {}
    for coord_name in ("radiating_dof", "influenced_dof"):
        if coord_name in standalone:
            standalone[coord_name] = standalone[coord_name].astype("U")
            encoding[coord_name] = {"dtype": "U"}

    netcdf_file.parent.mkdir(parents=True, exist_ok=True)
    standalone.to_netcdf(netcdf_file, mode="w", encoding=encoding or None)


def _load_or_generate_mesh(mesh_name: str, mesh_config: MeshConfig, settings: SimulationSettings) -> trimesh.Trimesh:
    """
    Load a mesh from an STL file and apply transformations, or generate it if it doesn't exist.

    - If the STL file specified in `mesh_config.file` exists, it's loaded, and the transformations
      (translation, rotation) from the `mesh_config` are applied.
    - If the file does not exist, this function attempts to generate it by taking the `settings.base_mesh`,
      applying the transformations from `mesh_config`, and saving the result to the path specified
      in `mesh_config.file`.
    """
    target_stl_path = Path(mesh_config.file)

    if target_stl_path.exists():
        logger.info(f"Found existing STL file: '{target_stl_path}'. Loading and applying transformations.")
        # Load the existing STL and apply its specific transformations.
        return _prepare_trimesh_geometry(stl_file=str(target_stl_path), mesh_config=mesh_config)

    # If the STL file does not exist, generate it from the base mesh.
    logger.info(f"STL file not found at '{target_stl_path}'. Attempting to generate from base mesh.")
    source_file_path = settings.base_mesh
    if not source_file_path or not Path(source_file_path).exists():
        err_msg = (
            f"Cannot generate mesh '{mesh_name}'. The source file '{target_stl_path}' does not exist, "
            f"and no valid 'base_mesh' ('{source_file_path}') is configured to generate it from."
        )
        raise FileNotFoundError(err_msg)

    # Load the base STL, apply the specified transformations.
    generated_mesh = _prepare_trimesh_geometry(str(source_file_path), mesh_config)

    # Save the newly generated, transformed mesh to the target path for future runs and inspection.
    logger.info(f"Saving newly generated mesh to: {target_stl_path}")
    target_stl_path.parent.mkdir(parents=True, exist_ok=True)
    generated_mesh.export(target_stl_path)

    return generated_mesh


def _obtain_mesh(
    mesh_name: str, mesh_config: MeshConfig, settings: SimulationSettings, output_file: Path
) -> trimesh.Trimesh:
    """
    Obtains a mesh for processing, prioritizing the database cache.

    1.  If `overwrite_meshes` is False, it first attempts to load the mesh from the HDF5 database.
    2.  If the mesh is not found in the database, or if `overwrite_meshes` is True, it falls back
        to loading or generating the mesh from an STL file via `_load_or_generate_mesh`.
    """
    # 1. Prioritize loading from the HDF5 database if overwrite_meshes is False
    if not settings.overwrite_meshes:
        try:
            if existing_meshes := load_meshes_from_hdf5(output_file, [mesh_name]):
                logger.info(f"Found existing mesh '{mesh_name}' in the database. Using it directly.")
                return existing_meshes[0]
        except FileNotFoundError:
            # The HDF5 file doesn't exist yet, so no meshes can exist. This is expected on the first run.
            pass
    else:  # This means overwrite_meshes is True
        logger.info(
            f"'overwrite_meshes' is True. Mesh '{mesh_name}' will be regenerated from its STL file and updated in the database."
        )

    # 2. If not in DB or if overwriting, load/generate from STL.
    return _load_or_generate_mesh(mesh_name, mesh_config, settings)


def _process_single_stl(
    mesh_config: MeshConfig,
    settings: SimulationSettings,
    output_file: Path,
    mesh_name_override: str | None = None,
    origin_translation: npt.NDArray[np.float64] | None = None,
) -> None:
    """
    Checks if a mesh exists in the database. If so, uses it.
    If not, generates it, saves it, and then uses it for the simulation pipeline.

    Mesh selection priority:
    - If a mesh exists in the database and overwrite_meshes is False, the database mesh is used.
    - If overwrite_meshes is True, the mesh is regenerated from the STL file and replaces the database mesh.
    - If no mesh exists in the database, the mesh is generated from the STL file and saved to the database.

    This ensures that the database mesh is preferred unless the user explicitly requests to overwrite meshes.
    """
    mesh_name = mesh_name_override or Path(mesh_config.file).stem

    # Obtain the mesh, either from the database or by loading/generating it.
    final_mesh_to_process = _obtain_mesh(mesh_name, mesh_config, settings, output_file)

    # Run the complete processing pipeline with the determined mesh.
    engine_mesh = EngineMesh(name=mesh_name, mesh=final_mesh_to_process, config=mesh_config)
    _run_pipeline_for_mesh(engine_mesh, settings, output_file, origin_translation)


def _log_pipeline_parameters(
    engine_mesh: EngineMesh,
    output_file: Path,
    settings: SimulationSettings,
    wave_directions_rad: list[float],
    wave_periods: list[float],
    water_depths: list[float],
    water_levels: list[float],
    forwards_speeds: list[float],
) -> None:
    """Logs all relevant parameters for a pipeline run for better traceability."""
    params = {
        "Base STL file": engine_mesh.config.file,
        "Base STL vertices": engine_mesh.mesh.vertices.shape,
        "Output file": output_file,
        "Grid symmetry": settings.grid_symmetry,
        "Use lid": settings.lid,
        "Add COG": settings.add_center_of_mass,
        "Direction(s) [rad]": wave_directions_rad,
        "Wave period(s) [s]": wave_periods,
        "Water depth(s) [m]": water_depths,
        "Water level(s) [m]": water_levels,
        "Translation X": engine_mesh.config.translation[0],
        "Translation Y": engine_mesh.config.translation[1],
        "Translation Z": engine_mesh.config.translation[2],
        "Rotation Roll [deg]": engine_mesh.config.rotation[0],
        "Rotation Pitch [deg]": engine_mesh.config.rotation[1],
        "Rotation Yaw [deg]": engine_mesh.config.rotation[2],
        "Forward speed(s) [m/s]": forwards_speeds,
    }
    for key, val in params.items():
        logger.info(f"{key:<40}: {val}")


def _run_pipeline_for_mesh(
    engine_mesh: EngineMesh,
    settings: SimulationSettings,
    output_file: Path,
    origin_translation: npt.NDArray[np.float64] | None,
) -> None:
    """
    Run the complete processing pipeline for a single STL file.
    """
    logger.info(f"Processing STL file: {engine_mesh.config.file}")

    # check is done by Settings, so this should no happen anymore
    if settings.lid and settings.grid_symmetry:
        raise LidAndSymmetryEnabledError()

    # Use mesh-specific wave periods and directions if provided, otherwise fall back to global settings.
    periods_to_use = engine_mesh.config.wave_periods or settings.wave_periods
    wave_periods_untyped = periods_to_use if isinstance(periods_to_use, list) else [periods_to_use]
    wave_periods = [float(p) for p in wave_periods_untyped]
    wave_frequencies = (2 * np.pi / np.array(wave_periods)).tolist()

    directions_to_use = engine_mesh.config.wave_directions or settings.wave_directions
    wave_directions_deg_untyped = directions_to_use if isinstance(directions_to_use, list) else [directions_to_use]
    wave_directions_deg = [float(d) for d in wave_directions_deg_untyped]
    wave_directions_rad = np.deg2rad(wave_directions_deg).tolist()

    water_depths_untyped = settings.water_depth if isinstance(settings.water_depth, list) else [settings.water_depth]
    water_depths = [float(d) for d in water_depths_untyped]
    water_levels_untyped = settings.water_level if isinstance(settings.water_level, list) else [settings.water_level]
    water_levels = [float(lvl) for lvl in water_levels_untyped]
    forwards_speeds_untyped = (
        settings.forward_speed if isinstance(settings.forward_speed, list) else [settings.forward_speed]
    )
    forwards_speeds = [float(s) for s in forwards_speeds_untyped]

    _log_pipeline_parameters(
        engine_mesh=engine_mesh,
        output_file=output_file,
        settings=settings,
        wave_directions_rad=wave_directions_rad,
        wave_periods=wave_periods,
        water_depths=water_depths,
        water_levels=water_levels,
        forwards_speeds=forwards_speeds,
    )

    process_all_cases_for_one_stl(
        engine_mesh=engine_mesh,
        wave_frequencies=wave_frequencies,
        wave_directions=wave_directions_rad,
        water_depths=water_depths,
        water_levels=water_levels,
        forwards_speeds=forwards_speeds,
        lid=settings.lid,
        grid_symmetry=settings.grid_symmetry,
        output_file=output_file,
        update_cases=settings.update_cases,
        combine_cases=settings.combine_cases,
        output_netcdf_file=Path(settings.output_netcdf_file) if settings.output_netcdf_file else None,
        origin_translation=origin_translation,
    )


def _process_and_save_single_case(
    boat: Any,  # cpt.FloatingBody is not fully typed, use Any to satisfy mypy
    mesh_name: str,
    case_params: dict[str, Any],
    output_file: Path,
    output_netcdf_file: Path | None,
    origin_translation: npt.NDArray[np.float64] | None,
) -> Any:
    """Process a single simulation case and save its results to the HDF5 file."""
    group_name = _generate_case_group_name(
        mesh_name, case_params["water_depth"], case_params["water_level"], case_params["forward_speed"]
    )

    with h5py.File(output_file, "a") as f:
        if group_name in f:
            if not case_params["update_cases"]:
                logger.info(f"Case '{group_name}' already exists in the database. Skipping.")
                return None
            logger.info(f"Case '{group_name}' exists, but update_cases is True. Overwriting.")
            del f[group_name]

    # Calculate the transformation matrix for this specific case relative to the global origin
    transformation_matrix = None
    if origin_translation is not None:
        origin_translation = np.asarray(origin_translation)
        # The transformation is the translation from the global origin to the mesh's COG for this case.
        # Note: boat.center_of_mass is the COG used for calculation, not necessarily the geometric center.
        translation_vector = boat.center_of_mass - origin_translation
        transformation_matrix = trimesh.transformations.translation_matrix(translation_vector)

    logger.info(
        f"Starting BEM calculations for water_level={case_params['water_level']}, "
        f"water_depth={case_params['water_depth']}, forward_speed={case_params['forward_speed']}"
    )
    # Select only the parameters that make_database expects.
    db_params = {
        "omegas": case_params["omegas"],
        "wave_directions": case_params["wave_directions"],
        "water_depth": case_params["water_depth"],
        "water_level": case_params["water_level"],
        "forward_speed": case_params["forward_speed"],
    }
    database = make_database(body=boat, **db_params)

    if not case_params["combine_cases"]:
        logger.info(f"Writing simulation results to group '{group_name}' in HDF5 file: {output_file}")
        database.attrs["stl_mesh_name"] = mesh_name
        if transformation_matrix is not None:
            database.attrs["transformation_matrix"] = transformation_matrix
        if boat.center_of_mass is not None:
            database.attrs["cog_for_calculation"] = boat.center_of_mass
        database.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")

    if output_netcdf_file is not None:
        logger.info(f"Writing standalone NetCDF file: {output_netcdf_file}")
        _write_case_to_netcdf(database, output_netcdf_file, mesh_name)

    logger.debug(f"Successfully wrote data for case to group {group_name}.")
    return database


def process_all_cases_for_one_stl(
    engine_mesh: EngineMesh,
    wave_frequencies: list | npt.NDArray[np.float64],
    wave_directions: list | npt.NDArray[np.float64],
    water_depths: list | npt.NDArray[np.float64],
    water_levels: list | npt.NDArray[np.float64],
    forwards_speeds: list | npt.NDArray[np.float64],
    lid: bool,
    grid_symmetry: bool,
    output_file: Path,
    update_cases: bool = False,
    combine_cases: bool = False,
    output_netcdf_file: Path | None = None,
    origin_translation: npt.NDArray[np.float64] | None = None,
) -> None:
    # 1. Use the prepared (and possibly translated) geometry to create the Capytaine body
    boat, final_mesh = _prepare_capytaine_body(
        engine_mesh=engine_mesh,
        lid=lid,
        grid_symmetry=grid_symmetry,
    )

    # 2. Add the final, immersed mesh geometry to the database. This version is now the translated one.
    if final_mesh is not None:
        add_mesh_to_database(
            output_file, final_mesh, engine_mesh.name, overwrite=update_cases, mesh_config=engine_mesh.config
        )

    all_datasets = []

    for water_level, water_depth, forward_speed in product(water_levels, water_depths, forwards_speeds):
        case_params = {
            "omegas": wave_frequencies,
            "wave_directions": wave_directions,
            "water_level": water_level,
            "water_depth": water_depth,
            "forward_speed": forward_speed,
            "update_cases": update_cases,
            "combine_cases": combine_cases,
        }
        result_db = _process_and_save_single_case(
            boat,
            engine_mesh.name,
            case_params,
            output_file,
            output_netcdf_file,
            origin_translation,
        )
        if combine_cases and result_db is not None:
            all_datasets.append(result_db)

    if combine_cases:
        if all_datasets:
            logger.info("Combining all calculated cases into a single multi-dimensional dataset.")
            combined_dataset = xr.combine_by_coords(all_datasets, combine_attrs="drop_conflicts")
            combined_group_name = f"{engine_mesh.name}_multi_dim"

            logger.info(f"Writing combined dataset to group '{combined_group_name}' in HDF5 file: {output_file}")
            with h5py.File(output_file, "a") as f:
                if combined_group_name in f:
                    del f[combined_group_name]
            combined_dataset.to_netcdf(output_file, mode="a", group=combined_group_name, engine="h5netcdf")
            with h5py.File(output_file, "a") as f:
                f[combined_group_name].attrs["stl_mesh_name"] = engine_mesh.name
        else:
            logger.warning(
                "The 'combine_cases' option is enabled, but no datasets were generated to combine. "
                "This can happen if all cases were already present in the output file and 'update_cases' was false."
            )

    logger.debug(f"Successfully wrote all data for mesh '{engine_mesh.name}' to HDF5.")


def run_simulation_batch(settings: SimulationSettings) -> None:
    """
    Runs a batch of Capytaine simulations and saves all results to a single HDF5 file.

    If `settings.drafts` is provided, it generates new meshes by translating a single
    base STL file for each draft. Otherwise, it processes the provided list of STL files.

    Args:
        settings: A SimulationSettings object with all necessary parameters.
    """
    logger.info("Starting simulation batch...")
    try:
        output_file = _setup_output_file(settings)
    except ValueError as e:
        logger.warning(e)
        return

    # Determine the base mesh and the origin translation
    all_mesh_configs = [MeshConfig.model_validate(mc) for mc in settings.stl_files]
    _validate_single_case_netcdf_export(settings, len(all_mesh_configs))
    all_files = [mc.file for mc in all_mesh_configs]

    origin_translation = np.array([0.0, 0.0, 0.0])
    base_mesh_path: str | None = settings.base_mesh
    if not base_mesh_path and all_files:
        base_mesh_path = all_files[0]

    if base_mesh_path:
        # Load the base mesh geometry once, as it might be needed for origin calculation or saving.
        base_mesh_trimesh = _prepare_trimesh_geometry(base_mesh_path)
        base_mesh_name = Path(base_mesh_path).stem

        if settings.base_origin:
            # If base_origin is specified, it's a point in the local coordinates of the base_mesh.
            # This point becomes the origin of our world coordinate system.
            origin_translation = np.array(settings.base_origin)
            logger.info(f"Using local point {origin_translation} from '{base_mesh_path}' as the world origin.")
        else:
            origin_translation = base_mesh_trimesh.center_mass
            logger.info(f"Database origin (center of mass of base mesh) set to: {origin_translation}")

        # Add the base mesh to the HDF5 database under the 'meshes' group.
        add_mesh_to_database(output_file, base_mesh_trimesh, base_mesh_name, overwrite=settings.overwrite_meshes)

        # Store the base reference information in the root of the HDF5 file
        with h5py.File(output_file, "a") as f:
            f.attrs["base_mesh"] = base_mesh_name
            if settings.base_origin:
                f.attrs["base_origin"] = settings.base_origin
            else:
                f.attrs["base_origin"] = origin_translation  # Store the calculated CoM as origin
    else:
        logger.warning("No base mesh provided.")

    if settings.drafts and base_mesh_path:
        if len(all_files) != 1:
            msg = f"When using --drafts, exactly one base STL file must be provided, but {len(all_files)} were given."
            logger.error(msg)
            raise ValueError(msg)

        base_mesh_name = Path(base_mesh_path).stem
        for draft in settings.drafts:
            logger.info(f"Processing for draft: {draft}")

            # Create a copy of the settings to modify for this specific draft
            draft_settings = settings.model_copy(deep=True)

            # Create a MeshConfig for this specific draft
            base_mesh_config = next((mc for mc in all_mesh_configs if mc.file == base_mesh_path), None)
            draft_translation = base_mesh_config.translation.copy() if base_mesh_config else [0.0, 0.0, 0.0]
            draft_translation[2] -= draft  # Positive draft means sinking, so subtract from Z

            # Create a unique name for this draft-specific mesh configuration
            draft_str = _format_value_for_name(draft)
            mesh_name_for_draft = f"{base_mesh_name}_draft_{draft_str}"

            draft_mesh_config = MeshConfig(file=base_mesh_path, translation=draft_translation)

            # Process this specific configuration
            _process_single_stl(
                draft_mesh_config,
                draft_settings,
                output_file,
                mesh_name_override=mesh_name_for_draft,
                origin_translation=origin_translation,
            )

    else:
        # Standard mode: process files as they are
        logger.info("Starting standard processing for provided STL files.")
        for mesh_config in all_mesh_configs:
            _process_single_stl(
                mesh_config, settings, output_file, mesh_name_override=None, origin_translation=origin_translation
            )

    logger.info(f"✅ Simulation batch finished. Results saved to {output_file}")


def create_hyddb_from_capytaine_file(filename: str | Path) -> Any:
    """Loads hydrodynamic data from a  dataset produced with capytaine.

    - Wave forces,
    - radiation_damping and
    - added_mass are read.

    See Also:
        Rao.wave_force_from_capytaine
    """

    from capytaine.io.xarray import merge_complex_values

    dataset = merge_complex_values(xr.open_dataset(filename))

    hyddb = create_hyd_from_capytaine_data(dataset)

    return hyddb


def create_hyd_from_capytaine_data(dataset: xr.Dataset) -> Any:
    hyddb = Hyddb1()

    hyddb._force.clear()

    for mode in MotionMode:
        r = create_rao_from_capytaine_wave_force(dataset, mode)
        r.scale(hyddb._N_to_kN)
        hyddb._force.append(r)

    hyddb._damping = dof_names_to_numbers(dataset["radiation_damping"] * hyddb._N_to_kN)
    hyddb._mass = dof_names_to_numbers(dataset["added_mass"] * hyddb._kg_to_mt)

    try:
        hyddb._check_dimensions()  # self-check
    except ValueError:
        logger.exception("Error when reading hydrodynamic data from")

    return hyddb


def create_rao_from_capytaine_wave_force(dataset: xr.Dataset, mode: Any) -> Any:
    """
    Reads hydrodynamic data from a netCFD file created with capytaine and copies the
    data for the requested mode into the object.

    Args:
        dataset: the xarray.Dataset read from the capytaine netCDF file
        mode: Name of the mode to read MotionMode

    Returns:
        None

    Examples:
        _test = Rao()
        _test.wave_force_from_capytaine(r"capytaine.nc", MotionMode.HEAVE)

    """
    rao = Rao()

    wave_direction = dataset["wave_direction"] * (180 / np.pi)  # convert rad to deg
    dataset = dataset.assign_coords(wave_direction=wave_direction)

    if "excitation_force" not in dataset:
        dataset["excitation_force"] = dataset["Froude_Krylov_force"] + dataset["diffraction_force"]

    cmode = MotionModeToStr(mode)

    da = dataset["excitation_force"].sel(influenced_dof=cmode)

    rao._data = xr.Dataset()

    rao._data["amplitude"] = np.abs(da)

    rao._data["phase"] = rao._data["amplitude"]  # To avoid shape mismatch,
    rao._data["phase"].values = np.angle(da)  # first copy with dummy data - then fill

    rao.mode = mode
    return rao
