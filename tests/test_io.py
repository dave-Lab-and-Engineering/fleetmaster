from pathlib import Path

import h5py
import numpy as np
import trimesh

from fleetmaster.core.io import load_meshes_from_hdf5


def _write_mesh_to_hdf5(hdf5_path: Path, mesh_name: str = "box") -> None:
    mesh = trimesh.creation.box()
    stl_bytes = mesh.export(file_type="stl")
    if isinstance(stl_bytes, str):
        stl_bytes = stl_bytes.encode()

    with h5py.File(hdf5_path, "w") as f:
        mesh_group = f.create_group(f"meshes/{mesh_name}")
        mesh_group.create_dataset("stl_content", data=np.void(stl_bytes))


def test_load_meshes_from_hdf5_with_path(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "meshes.hdf5"
    _write_mesh_to_hdf5(hdf5_path)

    meshes = load_meshes_from_hdf5(hdf5_path, ["box"])

    assert len(meshes) == 1
    assert meshes[0].metadata.get("name") == "box"


def test_load_meshes_from_hdf5_with_open_file(tmp_path: Path) -> None:
    hdf5_path = tmp_path / "meshes.hdf5"
    _write_mesh_to_hdf5(hdf5_path)

    with h5py.File(hdf5_path, "r") as f:
        meshes = load_meshes_from_hdf5(f, ["box"])

    assert len(meshes) == 1
    assert meshes[0].metadata.get("name") == "box"
