import logging
from pathlib import Path

import h5py
import numpy as np
import trimesh

# import cKDTree directly so linters and static type checkers can detect it,
# and provide a clear ImportError if it's not available at runtime
try:
    from scipy.spatial import cKDTree  # type: ignore  # noqa: PGH003
except Exception as e:
    msg = "scipy.spatial.cKDTree is required for computing Chamfer distances"
    raise ImportError(msg) from e

from fleetmaster.core.engine import (
    EngineMesh,
    _apply_mesh_translation_and_rotation,
    _prepare_capytaine_body,
)
from fleetmaster.core.exceptions import DatabaseFileNotFoundError, HDF5AttributeError, MeshLoadError
from fleetmaster.core.io import load_meshes_from_hdf5
from fleetmaster.core.settings import MESH_GROUP_NAME, MeshConfig

logger = logging.getLogger(__name__)


def _calculate_chamfer_distance(mesh_A: trimesh.Trimesh, mesh_B: trimesh.Trimesh) -> float:
    """
    Calculates the Root Mean Square Chamfer distance between two meshes.

    This provides a robust measure of the average distance between the vertices of two meshes,
    making it suitable for finding the best overall fit.

    Args:
        mesh_A: The first trimesh object.
        mesh_B: The second trimesh object.

    Returns:
        The RMS Chamfer distance. A lower value indicates a better match.
    """
    vertices_A = mesh_A.vertices
    vertices_B = mesh_B.vertices

    num_vertices_A = len(vertices_A)
    num_vertices_B = len(vertices_B)

    if num_vertices_A == 0 or num_vertices_B == 0:
        # If one mesh is empty, distance is infinite unless both are empty.
        return 0.0 if num_vertices_A == num_vertices_B else np.inf

    tree_A = cKDTree(vertices_A)
    tree_B = cKDTree(vertices_B)

    dist_A_to_B, _ = tree_B.query(vertices_A, k=1)
    dist_B_to_A, _ = tree_A.query(vertices_B, k=1)

    # The sum of the squares is commonly used.
    total_chamfer_dist = np.sum(np.square(dist_A_to_B)) + np.sum(np.square(dist_B_to_A))
    rmsd = np.sqrt(total_chamfer_dist / (num_vertices_A + num_vertices_B))
    return float(rmsd)


def _find_best_fit_for_candidates(
    base_mesh: trimesh.Trimesh,
    candidate_meshes: dict[str, trimesh.Trimesh],
    target_translation: list[float],
    target_rotation: list[float],
    water_level: float,
) -> dict[str, float]:
    """
    Finds the best fit for a base mesh against a set of candidate meshes.

    For each candidate, this function transforms a copy of the base mesh using a hybrid
    transformation derived from the candidate and a target transformation.

    - XY translation from the candidate, Z translation from the target.
    - Z rotation (yaw) from the candidate, XY rotation (roll, pitch) from the target.

    It then calculates the Chamfer distance between the wetted surfaces.

    Args:
        base_mesh: The base trimesh object.
        candidate_meshes: A dictionary mapping mesh names to their trimesh objects.
        target_translation: The target global translation [x, y, z].
        target_rotation: The target global rotation [roll, pitch, yaw] in degrees.
        water_level: The water level at which to cut the mesh for a fair comparison.

    Returns:
        A dictionary mapping each candidate mesh name to its calculated Chamfer distance.
    """
    distances = {}
    logger.info(f"Finding best fit for {len(candidate_meshes)} candidate meshes...")

    for name, candidate_mesh in candidate_meshes.items():
        candidate_translation = candidate_mesh.metadata.get("translation")
        candidate_rotation = candidate_mesh.metadata.get("rotation")

        if candidate_translation is None or candidate_rotation is None:
            logger.warning(f"Candidate '{name}' is missing translation/rotation metadata. Skipping.")
            distances[name] = np.inf
            continue

        # The goal of the fitting is to find a mesh from the database that best matches
        # the target's submerged shape, which is primarily determined by Z-translation (draft)
        # and X/Y-rotations (roll, pitch). The database contains meshes with varying roll and pitch,
        # but typically constant XY translation and Z-rotation (yaw).
        #
        # To find the best match, we create a hybrid transformation that respects these assumptions:
        # - We use the target's Z-translation (draft) because that's a key property we're matching.
        # - We use the target's roll and pitch for the same reason.
        # - We take the candidate's XY-translation and yaw, because these are considered irrelevant
        #   for the shape matching and are constant in the database generation process.
        #
        # This allows us to transform the base mesh into a shape that is directly comparable
        # with the candidate's wetted surface.
        new_translation = [
            candidate_translation[0],
            candidate_translation[1],
            target_translation[2],
        ]
        new_rotation = [
            target_rotation[0],
            target_rotation[1],
            candidate_rotation[2],
        ]

        temp_base_mesh = base_mesh.copy()
        transformed_base_mesh = _apply_mesh_translation_and_rotation(
            mesh=temp_base_mesh,
            translation_vector=new_translation,
            rotation_vector_deg=new_rotation,
        )

        # Create a dummy EngineMesh to use the _prepare_capytaine_body function for cutting the mesh.
        dummy_config = MeshConfig(file="dummy")
        engine_mesh_for_cutting = EngineMesh(name="temp_base", mesh=transformed_base_mesh, config=dummy_config)

        _, cut_transformed_base_mesh = _prepare_capytaine_body(
            engine_mesh=engine_mesh_for_cutting,
            lid=False,
            grid_symmetry=False,
            water_level=water_level,
        )

        if not cut_transformed_base_mesh or len(cut_transformed_base_mesh.vertices) == 0:
            logger.warning(
                f"Transformed base mesh for candidate '{name}' is out of the water. Assigning infinite distance."
            )
            distances[name] = np.inf
            continue

        # The candidate mesh from the database is already the wetted surface.
        distance = _calculate_chamfer_distance(cut_transformed_base_mesh, candidate_mesh)
        logger.debug(f"  - Calculated distance to '{name}': {distance:.4f}")
        distances[name] = distance

    return distances


def find_best_matching_mesh(
    hdf5_path: Path,
    target_translation: list[float],
    target_rotation: list[float],
    water_level: float = 0.0,
) -> tuple[str | None, float]:
    """
    Finds the best matching mesh from an HDF5 database for a given target transformation.

    The function works as follows:
    1.  It loads the base mesh and all candidate meshes from the HDF5 file.
    2.  For each candidate, it transforms the base_mesh using a hybrid transformation:
        - XY-translation and Z-rotation from the candidate.
        - Z-translation and XY-rotation from the target transformation.
    3.  It then computes the Chamfer distance between the wetted surface of the transformed
        base mesh and the wetted surface of the candidate mesh.
    4.  It returns the name of the mesh with the smallest Chamfer distance.

    Args:
        hdf5_path (Path): Path to the HDF5 database file.
        target_translation (list[float]): The target translation [x, y, z] to apply to the base mesh.
        target_rotation (list[float]): The target rotation [roll, pitch, yaw] in degrees.
        water_level (float): The water level to use for cutting the meshes for comparison. Defaults to 0.0.

    Returns:
        A tuple containing the name of the best matching mesh and the corresponding Chamfer distance.
        Returns (None, np.inf) if no match is found.
    """
    if not hdf5_path.exists():
        raise DatabaseFileNotFoundError(path=hdf5_path)

    base_mesh_name: str | None = None
    candidate_mesh_names: list[str] = []

    # 1. Identify base mesh and candidate meshes from the HDF5 file
    with h5py.File(hdf5_path, "r") as f:
        if "base_mesh" not in f.attrs:
            raise HDF5AttributeError(attribute_name="base_mesh")
        base_mesh_name = str(f.attrs["base_mesh"])

        if MESH_GROUP_NAME not in f:
            logger.warning(f"No '{MESH_GROUP_NAME}' group found in HDF5 file. Cannot find any meshes.")
            return None, np.inf

        mesh_group = f[MESH_GROUP_NAME]
        if not isinstance(mesh_group, h5py.Group):
            logger.warning(f"'{MESH_GROUP_NAME}' in HDF5 file is not a group as expected.")
            return None, np.inf

        # Candidates are all meshes that are not the base mesh
        candidate_mesh_names = [str(name) for name in mesh_group if name != base_mesh_name]

    if not base_mesh_name or not candidate_mesh_names:
        logger.warning("No base mesh or candidate meshes found to perform a match.")
        return None, np.inf

    # 2. Load all required meshes, including their metadata (translation, rotation)
    all_meshes_to_load = [base_mesh_name, *candidate_mesh_names]
    loaded_meshes = {mesh.metadata["name"]: mesh for mesh in load_meshes_from_hdf5(hdf5_path, all_meshes_to_load)}

    base_mesh = loaded_meshes.get(base_mesh_name)
    if not base_mesh:
        raise MeshLoadError(mesh_name=base_mesh_name)

    candidate_meshes = {name: mesh for name, mesh in loaded_meshes.items() if name != base_mesh_name}

    # 3. Find the distances for all candidates based on the new logic
    all_distances = _find_best_fit_for_candidates(
        base_mesh=base_mesh,
        candidate_meshes=candidate_meshes,
        target_translation=target_translation,
        target_rotation=target_rotation,
        water_level=water_level,
    )

    # 4. Find the minimum distance among the results
    if not all_distances:
        logger.warning("No distances could be calculated.")
        return None, np.inf

    best_match_name = min(all_distances, key=lambda k: all_distances[k])
    min_distance = all_distances[best_match_name]

    logger.info(f"Best match found: '{best_match_name}' with a Chamfer distance of {min_distance:.4f}")
    return best_match_name, min_distance
