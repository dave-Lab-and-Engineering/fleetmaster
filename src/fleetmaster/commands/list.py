"""CLI command for listing meshes or cases from HDF5 databases."""

import io
import logging
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
import trimesh
from trimesh import Trimesh

logger = logging.getLogger(__name__)


def _parse_stl_content(stl_content_dataset: Any) -> Trimesh:
    """
    Parses binary STL data from an HDF5 dataset into a trimesh object.

    Handles both legacy (numpy.void) and current storage formats.
    """
    stl_data = stl_content_dataset[()]
    try:
        # Accommodate legacy storage format (numpy.void)
        stl_bytes = stl_data.tobytes()
    except AttributeError:
        # Current format is already bytes
        stl_bytes = stl_data

    mesh = trimesh.load_mesh(io.BytesIO(stl_bytes), file_type="stl")
    if not isinstance(mesh, Trimesh):
        msg = "Failed to parse STL data into a valid trimesh object."
        raise TypeError(msg)
    return mesh


def _print_mesh_details(mesh_info_group: Any) -> None:
    """Prints formatted geometric properties of a mesh from its HDF5 group."""
    attrs = mesh_info_group.attrs
    vol = attrs.get("volume", "N/A")
    cog = (attrs.get("cog_x", "N/A"), attrs.get("cog_y", "N/A"), attrs.get("cog_z", "N/A"))
    dims = (attrs.get("bbox_lx", "N/A"), attrs.get("bbox_ly", "N/A"), attrs.get("bbox_lz", "N/A"))

    num_faces: str | int = "N/A"
    bounds: np.ndarray | None = None
    if stl_content_dataset := mesh_info_group.get("stl_content"):
        try:
            mesh = _parse_stl_content(stl_content_dataset)
            num_faces = len(mesh.faces)
            bounds = mesh.bounding_box.bounds
        except (ValueError, TypeError, AttributeError) as e:
            mesh_name = Path(mesh_info_group.name).name
            logger.debug(f"Failed to parse STL content for mesh '{mesh_name}': {e}")
            click.echo(f"      Could not parse stored STL content: {e}")

    click.echo(f"      Cells: {num_faces}")
    click.echo(f"      Volume: {vol:.4f}" if isinstance(vol, float) else f"      Volume: {vol}")
    click.echo(
        f"      COG (x,y,z): ({cog[0]:.3f}, {cog[1]:.3f}, {cog[2]:.3f})"
        if all(isinstance(c, float) for c in cog)
        else f"      COG (x,y,z): {cog}"
    )
    click.echo(
        f"      BBox Dims (Lx,Ly,Lz): ({dims[0]:.3f}, {dims[1]:.3f}, {dims[2]:.3f})"
        if all(isinstance(d, float) for d in dims)
        else f"      BBox Dims (Lx,Ly,Lz): {dims}"
    )
    if bounds is not None:
        click.echo(f"      BBox Min (x,y,z): ({bounds[0][0]:.3f}, {bounds[0][1]:.3f}, {bounds[0][2]:.3f})")
        click.echo(f"      BBox Max (x,y,z): ({bounds[1][0]:.3f}, {bounds[1][1]:.3f}, {bounds[1][2]:.3f})")


def _list_cases(stream: Any, hdf5_path: str) -> None:
    """Lists all simulation cases and their mesh properties in an HDF5 file."""
    click.echo(f"\nAvailable cases in '{hdf5_path}':")
    case_names = [name for name in stream if name != "meshes"]
    if not case_names:
        click.echo("  No cases found.")
        return

    for case_name in sorted(case_names):
        case_group = stream[case_name]
        click.echo(f"\n- Case: {case_name}")

        if not (mesh_name := case_group.attrs.get("stl_mesh_name")):
            click.echo("    Mesh: [Unknown]")
            continue

        click.echo(f"    Mesh: {mesh_name}")
        if mesh_info_group := stream.get(f"meshes/{mesh_name}"):
            _print_mesh_details(mesh_info_group)
        else:
            click.echo("      Mesh properties not found in database.")


def _list_meshes(stream: Any, hdf5_path: str) -> None:
    """Lists all available meshes in an HDF5 file."""
    click.echo(f"\nAvailable meshes in '{hdf5_path}':")
    if not (meshes_group := stream.get("meshes")):
        click.echo("  No 'meshes' group found.")
        return

    if available_meshes := list(meshes_group.keys()):
        for name in sorted(available_meshes):
            click.echo(f"  - {name}")
    else:
        click.echo("  No meshes found.")


@click.command(name="list", help="List meshes by default, or use --cases to list simulation cases.")
@click.argument("files", nargs=-1, type=click.Path())
@click.option(
    "--file",
    "-f",
    "option_files",
    multiple=True,
    help="Path to one or more HDF5 database files. Can be specified multiple times.",
)
@click.option("--cases", is_flag=True, help="List simulation cases and their properties instead of meshes.")
def list_command(files: tuple[str, ...], option_files: tuple[str, ...], cases: bool) -> None:
    """CLI command to list meshes or cases."""
    # Combine positional arguments and optional --file arguments
    all_files = set(files) | set(option_files)

    # If no files are provided at all, use the default.
    final_files = list(all_files) if all_files else ["results.hdf5"]

    for hdf5_path in final_files:
        db_file = Path(hdf5_path)
        if not db_file.exists():
            click.echo(f"❌ Error: Database file '{hdf5_path}' not found.", err=True)
            continue
        try:
            with h5py.File(db_file, "r") as stream:
                if cases:
                    _list_cases(stream, hdf5_path)
                else:
                    _list_meshes(stream, hdf5_path)
        except Exception as e:
            click.echo(f"❌ Error reading '{hdf5_path}': {e}", err=True)
