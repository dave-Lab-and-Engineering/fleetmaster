import logging

import numpy as np
import trimesh

# import cKDTree directly so linters and static type checkers can detect it,
# and provide a clear ImportError if it's not available at runtime
try:
    from scipy.spatial import cKDTree  # noqa: PGH003  # type: ignore
except Exception as e:
    msg = "scipy.spatial.cKDTree is required for computing Chamfer distances"
    raise ImportError(msg) from e


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
