from __future__ import annotations

import hashlib
import logging
import tempfile
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import capytaine as cpt
import h5py
import mafredo
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


def _create_bem_solver(water_depth: float) -> Any:
    """Create a Capytaine solver with a robust finite-depth Green function setup."""
    if np.isfinite(water_depth):
        logger.info(
            "Using Delhommeau finite-depth Green function with Fortran Prony decomposition for water_depth=%s.",
            water_depth,
        )
        return cpt.BEMSolver(green_function=cpt.Delhommeau(finite_depth_prony_decomposition_method="fortran"))

    return cpt.BEMSolver()


def _build_bem_problems(
    body: Any,
    omegas: list | npt.NDArray[np.float64],
    wave_directions: list | npt.NDArray[np.float64],
    water_depth: float,
    water_level: float,
    forward_speed: float,
) -> list[Any]:
    """Build radiation and diffraction problems for BEM solving."""
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
    return problems


def make_database(
    body: Any,
    omegas: list | npt.NDArray[np.float64],
    wave_directions: list | npt.NDArray[np.float64],
    water_depth: float,
    water_level: float,
    forward_speed: float,
    case_label: str | None = None,
) -> Any:
    """Create a dataset of BEM results for a given body and conditions."""
    bem_solver = _create_bem_solver(water_depth)
    problems = _build_bem_problems(body, omegas, wave_directions, water_depth, water_level, forward_speed)

    total_problems = len(problems)
    label_prefix = f"[{case_label}] " if case_label else ""
    logger.info(f"{label_prefix}Starting BEM solve for {total_problems} problems.")

    results = []
    start_time = time.perf_counter()
    progress_step = max(1, total_problems // 20)  # Roughly every 5%
    for idx, problem in enumerate(problems, start=1):
        problem_name = type(problem).__name__
        omega_val = getattr(problem, "omega", None)
        wave_direction = getattr(problem, "wave_direction", None)
        radiating_dof = getattr(problem, "radiating_dof", None)
        detail_bits = [problem_name]
        if omega_val is not None:
            detail_bits.append(f"omega={omega_val:.4f}")
        if wave_direction is not None:
            detail_bits.append(f"beta={wave_direction:.4f}")
        if radiating_dof is not None:
            detail_bits.append(f"dof={radiating_dof}")
        logger.info(
            "%sSolving %d/%d: %s",
            label_prefix,
            idx,
            total_problems,
            ", ".join(detail_bits),
        )
        results.append(bem_solver.solve(problem))

        if idx == 1 or idx == total_problems or idx % progress_step == 0:
            elapsed = max(time.perf_counter() - start_time, 1e-9)
            rate = idx / elapsed
            remaining = (total_problems - idx) / rate if rate > 0 else 0.0
            progress_pct = 100.0 * idx / total_problems
            logger.info(
                "%sBEM progress: %d/%d (%.1f%%), elapsed %.1fs, ETA %.1fs",
                label_prefix,
                idx,
                total_problems,
                progress_pct,
                elapsed,
                remaining,
            )

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

    mesh = _apply_mesh_translation_and_rotation(
        mesh=mesh,
        translation_vector=mesh_config.translation,
        rotation_vector_deg=mesh_config.rotation,
        cog=mesh_config.cog,
    )

    if mesh_config.clip_to_waterplane:
        mesh = _clip_mesh_at_waterplane(mesh)

    return mesh


def _clip_mesh_at_waterplane(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Clip a mesh at z=0, keeping only the submerged part (z <= 0).

    Panels entirely above the waterplane are removed, and panels that cross
    z=0 are split at the intersection. The resulting mesh is open at the
    waterplane — Capytaine's keep_immersed_part() will handle that correctly.
    """
    new_vertices, new_faces, _ = trimesh.intersections.slice_faces_plane(
        vertices=mesh.vertices,
        faces=mesh.faces,
        plane_normal=[0, 0, -1],  # normal pointing down: keep z <= 0
        plane_origin=[0, 0, 0],
    )
    if new_faces is None or len(new_faces) == 0:
        logger.warning("Mesh is empty after clipping at z=0. Check that the mesh extends below the waterplane.")
        return mesh
    clipped = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    logger.debug(
        f"Clipped mesh at z=0: {len(mesh.faces)} -> {len(clipped.faces)} faces "
        f"(removed {len(mesh.faces) - len(clipped.faces)} above-water panels)."
    )
    return clipped


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


def _export_transformed_mesh_to_stl(
    mesh_to_export: trimesh.Trimesh,
    mesh_name: str,
    output_file: Path,
    output_directory: Path | None,
    overwrite: bool,
) -> Path | None:
    """Export the final transformed mesh to a standalone STL file."""
    if not isinstance(mesh_to_export, trimesh.Trimesh) or mesh_to_export.is_empty:
        logger.warning(f"Skipping STL export for '{mesh_name}' because mesh is empty or invalid.")
        return None

    export_dir = output_directory if output_directory is not None else output_file.parent / "transformed_stl"
    export_dir.mkdir(parents=True, exist_ok=True)

    stl_path = export_dir / f"{mesh_name}.stl"
    if stl_path.exists() and not overwrite:
        logger.info(f"Transformed STL '{stl_path}' already exists. Skipping export.")
        return stl_path

    mesh_to_export.export(stl_path)
    logger.info(f"Exported transformed STL to '{stl_path}'.")
    return stl_path


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


def _should_export_dhyd(settings: SimulationSettings) -> bool:
    """Returns True when a standalone .dhyd file should be exported."""
    return settings.export_to_hyd or settings.output_dhyd_file is not None


def _validate_single_case_dhyd_export(settings: SimulationSettings, mesh_count: int) -> None:
    """Retained as a no-op for backward compatibility with existing tests/imports."""
    del settings, mesh_count


def _write_case_to_dhyd(database: xr.Dataset, dhyd_file: Path, heading_symmetry: bool = False) -> None:
    """Writes one case to a standalone mafredo hydrodynamic database (.dhyd)."""
    hyddb = create_hyd_from_capytaine_data(database)

    # Keep mafredo heading symmetry metadata explicit in exported .dhyd files.
    hyddb.symmetry = mafredo.Symmetry.XZ if heading_symmetry else mafredo.Symmetry.No

    dhyd_file.parent.mkdir(parents=True, exist_ok=True)
    hyddb.save_as(dhyd_file)


def _normalize_wave_directions_for_xz_heading_symmetry(wave_directions_deg: list[float]) -> list[float]:
    """Map headings to the XZ symmetry domain [0, 180] and remove near-duplicates."""
    normalized: list[float] = []
    for direction in wave_directions_deg:
        folded = float(direction) % 360.0
        if folded > 180.0:
            folded = 360.0 - folded

        if np.isclose(folded, 360.0):
            folded = 0.0

        if not any(np.isclose(folded, existing) for existing in normalized):
            normalized.append(folded)

    return normalized


def _resolve_output_dhyd_path(
    output_file: Path, group_name: str, output_dhyd_file: Path | None, export_to_hyd: bool
) -> Path | None:
    """Resolves the .dhyd output path for a case."""
    if output_dhyd_file is not None:
        if export_to_hyd:
            suffix = output_dhyd_file.suffix or ".dhyd"
            return output_dhyd_file.with_name(f"{output_dhyd_file.stem}_{group_name}{suffix}")
        return output_dhyd_file

    if export_to_hyd:
        return output_file.with_name(f"{group_name}.dhyd")

    return None


def _resolve_bulk_output_dhyd_path(
    hdf5_file: Path, case_group: str, output_dhyd_file: Path | None, exporting_multiple_cases: bool
) -> Path:
    """Resolves the output path for exporting a case from an existing HDF5 database."""
    if output_dhyd_file is None:
        return hdf5_file.with_name(f"{hdf5_file.stem}_{case_group}.dhyd")

    if exporting_multiple_cases:
        suffix = output_dhyd_file.suffix or ".dhyd"
        return output_dhyd_file.with_name(f"{output_dhyd_file.stem}_{case_group}{suffix}")

    return output_dhyd_file


def list_case_groups_in_hdf5(hdf5_file: str | Path) -> list[str]:
    """Lists all simulation case groups in an HDF5 database (excluding the meshes group)."""
    hdf5_path = Path(hdf5_file)
    if not hdf5_path.exists():
        msg = f"HDF5 database not found: {hdf5_path}"
        raise FileNotFoundError(msg)

    with h5py.File(hdf5_path, "r") as f:
        return sorted(name for name in f if name != MESH_GROUP_NAME)


def export_hdf5_case_to_dhyd(
    hdf5_file: str | Path,
    case_group: str,
    output_dhyd_file: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Exports one simulation case group from the Fleetmaster HDF5 database to a standalone .dhyd file."""
    hdf5_path = Path(hdf5_file)
    output_path = Path(output_dhyd_file)

    if not hdf5_path.exists():
        msg = f"HDF5 database not found: {hdf5_path}"
        raise FileNotFoundError(msg)

    if output_path.exists() and not overwrite:
        msg = f"Output .dhyd file already exists: {output_path}. Use --overwrite to replace it."
        raise FileExistsError(msg)

    with h5py.File(hdf5_path, "r") as f:
        if case_group not in f:
            available = sorted(name for name in f if name != MESH_GROUP_NAME)
            msg = (
                f"Case group '{case_group}' not found in {hdf5_path}. "
                f"Available cases: {available if available else 'none'}"
            )
            raise ValueError(msg)

    dataset = xr.open_dataset(hdf5_path, group=case_group, engine="h5netcdf")
    try:
        _write_case_to_dhyd(dataset.load(), output_path)
    finally:
        dataset.close()

    return output_path


def export_hdf5_cases_to_dhyd(
    hdf5_file: str | Path,
    case_groups: list[str] | None = None,
    output_dhyd_file: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Exports one or more simulation case groups from the Fleetmaster HDF5 database to standalone .dhyd files."""
    hdf5_path = Path(hdf5_file)

    if not hdf5_path.exists():
        msg = f"HDF5 database not found: {hdf5_path}"
        raise FileNotFoundError(msg)

    available_cases = list_case_groups_in_hdf5(hdf5_path)
    selected_cases = available_cases if case_groups is None else case_groups

    missing_cases = [case for case in selected_cases if case not in available_cases]
    if missing_cases:
        msg = f"Case group(s) not found in {hdf5_path}: {missing_cases}. Available cases: {available_cases}"
        raise ValueError(msg)

    output_template = Path(output_dhyd_file) if output_dhyd_file is not None else None
    exporting_multiple_cases = len(selected_cases) != 1
    exported_paths: list[Path] = []

    for case_group in selected_cases:
        output_path = _resolve_bulk_output_dhyd_path(hdf5_path, case_group, output_template, exporting_multiple_cases)
        exported_paths.append(
            export_hdf5_case_to_dhyd(
                hdf5_file=hdf5_path,
                case_group=case_group,
                output_dhyd_file=output_path,
                overwrite=overwrite,
            )
        )

    return exported_paths


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
    if settings.heading_symmetry:
        original_direction_count = len(wave_directions_deg)
        wave_directions_deg = _normalize_wave_directions_for_xz_heading_symmetry(wave_directions_deg)
        logger.info(
            "Heading symmetry enabled: reduced wave directions from %d to %d values in [0, 180] deg.",
            original_direction_count,
            len(wave_directions_deg),
        )
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
        output_dhyd_file=Path(settings.output_dhyd_file) if settings.output_dhyd_file else None,
        export_to_hyd=settings.export_to_hyd,
        heading_symmetry=settings.heading_symmetry,
        export_transformed_stl=settings.export_transformed_stl,
        output_transformed_stl_directory=(
            Path(settings.output_transformed_stl_directory) if settings.output_transformed_stl_directory else None
        ),
        overwrite_meshes=settings.overwrite_meshes,
        origin_translation=origin_translation,
    )


def _process_and_save_single_case(
    boat: Any,  # cpt.FloatingBody is not fully typed, use Any to satisfy mypy
    mesh_name: str,
    case_params: dict[str, Any],
    output_file: Path,
    output_dhyd_file: Path | None,
    export_to_hyd: bool,
    heading_symmetry: bool,
    origin_translation: npt.NDArray[np.float64] | None,
    case_label: str,
) -> Any:
    """Process a single simulation case and save its results to the HDF5 file."""
    group_name = _generate_case_group_name(
        mesh_name, case_params["water_depth"], case_params["water_level"], case_params["forward_speed"]
    )
    resolved_output_dhyd_file = _resolve_output_dhyd_path(output_file, group_name, output_dhyd_file, export_to_hyd)

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
        f"{case_label} Starting BEM calculations for water_level={case_params['water_level']}, "
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
    database = make_database(body=boat, case_label=case_label, **db_params)

    if not case_params["combine_cases"]:
        logger.info(f"Writing simulation results to group '{group_name}' in HDF5 file: {output_file}")
        database.attrs["stl_mesh_name"] = mesh_name
        if transformation_matrix is not None:
            database.attrs["transformation_matrix"] = transformation_matrix
        if boat.center_of_mass is not None:
            database.attrs["cog_for_calculation"] = boat.center_of_mass
        database.to_netcdf(output_file, mode="a", group=group_name, engine="h5netcdf")

    if resolved_output_dhyd_file is not None:
        logger.info(f"Writing standalone .dhyd file: {resolved_output_dhyd_file}")
        _write_case_to_dhyd(database, resolved_output_dhyd_file, heading_symmetry=heading_symmetry)

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
    output_dhyd_file: Path | None = None,
    export_to_hyd: bool = False,
    heading_symmetry: bool = False,
    export_transformed_stl: bool = False,
    output_transformed_stl_directory: Path | None = None,
    overwrite_meshes: bool = False,
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
        if export_transformed_stl:
            _export_transformed_mesh_to_stl(
                mesh_to_export=final_mesh,
                mesh_name=engine_mesh.name,
                output_file=output_file,
                output_directory=output_transformed_stl_directory,
                overwrite=overwrite_meshes,
            )

    all_datasets = []

    all_cases = list(product(water_levels, water_depths, forwards_speeds))
    total_cases = len(all_cases)

    for case_index, (water_level, water_depth, forward_speed) in enumerate(all_cases, start=1):
        case_label = f"Case {case_index}/{total_cases}"
        logger.info(
            f"{case_label}: wl={water_level}, wd={water_depth}, fs={forward_speed} for mesh '{engine_mesh.name}'"
        )

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
            output_dhyd_file,
            export_to_hyd,
            heading_symmetry,
            origin_translation,
            case_label,
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
