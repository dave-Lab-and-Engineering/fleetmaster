import io
import logging
from typing import Any

import h5py
import trimesh

logger = logging.getLogger(__name__)


def load_meshes_from_hdf5(
    h5_file: Any,
    mesh_names: list[str],
) -> list[trimesh.Trimesh]:
    """Load and return trimesh objects for the given names from an opened HDF5 file."""
    meshes: list[trimesh.Trimesh] = []

    for name in mesh_names:
        group = h5_file.get(f"meshes/{name}")
        if not isinstance(group, h5py.Group):
            logger.warning(f"Mesh group '{name}' not found or not a group.")
            continue
        stl_dataset = group.get("stl_content")
        if not isinstance(stl_dataset, h5py.Dataset):
            logger.warning(f"'stl_content' in mesh group '{name}' is not a dataset.")
            continue
        raw = stl_dataset[()]
        try:
            mesh = trimesh.load_mesh(io.BytesIO(raw.tobytes()), file_type="stl")
            if isinstance(mesh, trimesh.Trimesh):
                mesh.metadata["name"] = name  # Store the name for later identification
                # Load attributes if they exist
                mesh.metadata["translation"] = group.attrs.get("translation")
                mesh.metadata["rotation"] = group.attrs.get("rotation")
                mesh.metadata["cog"] = group.attrs.get("cog")
                meshes.append(mesh)
        except Exception:
            logger.exception("Failed to parse mesh %r", name)
    return meshes
