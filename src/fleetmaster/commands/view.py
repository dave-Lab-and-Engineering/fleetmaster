"""CLI command for visualizing meshes from the HDF5 database."""

import logging
from itertools import cycle
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
import trimesh
from trimesh import Trimesh

from fleetmaster.core.io import load_meshes_from_hdf5

logger = logging.getLogger(__name__)

# Try to import vtk, but make it an optional dependency.
try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    VTK_AVAILABLE = True
    VTK_IMPORT_ERROR: Exception | None = None
    # The global import of numpy is sufficient.
except ImportError as exc:
    VTK_AVAILABLE = False
    VTK_IMPORT_ERROR = exc

VTK_COLORS = [
    (0.8, 0.8, 1.0),  # Light Blue
    (1.0, 0.8, 0.8),  # Light Red
    (0.8, 1.0, 0.8),  # Light Green
    (1.0, 1.0, 0.8),  # Light Yellow
]


def show_with_trimesh(mesh: trimesh.Trimesh) -> None:
    """Visualizes the mesh using the built-in trimesh viewer."""
    click.echo("🎨 Displaying mesh with trimesh viewer. Close the window to continue.")
    mesh.show()


def _vtk_actor_from_trimesh(mesh: trimesh.Trimesh, color: tuple[float, float, float]) -> Any:
    """Creates a VTK actor from a trimesh object."""
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(mesh.vertices, deep=True))

    faces = np.hstack((np.full((len(mesh.faces), 1), 3), mesh.faces))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(mesh.faces), numpy_to_vtk(faces.flatten(), array_type=vtk.VTK_ID_TYPE))

    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    poly.SetPolys(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    return actor


def show_with_vtk(meshes: list[Trimesh]) -> None:
    """Visualizes the mesh using a VTK pipeline."""
    if not VTK_AVAILABLE:
        message = "❌ Error: VTK could not be imported."
        if VTK_IMPORT_ERROR is not None:
            message = f"{message} {VTK_IMPORT_ERROR}"
        click.echo(message)
        click.echo("VTK is installed, but one of its native dependencies failed to load.")
        click.echo("On Windows, this usually means a missing or incompatible DLL/runtime.")
        return

    click.echo(f"🎨 Displaying {len(meshes)} mesh(es) with VTK viewer. Close the window to continue.")
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.1, 0.2, 0.3)  # Dark blue/gray
    for mesh, color in zip(meshes, cycle(VTK_COLORS)):
        renderer.AddActor(_vtk_actor_from_trimesh(mesh, color))

    # Add a global axes actor at the origin
    axes_at_origin = vtk.vtkAxesActor()
    axes_at_origin.SetTotalLength(1.0, 1.0, 1.0)  # Set size of the axes
    renderer.AddActor(axes_at_origin)

    # Add an axes actor for context
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()  # This is the small one in the corner
    widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    widget.SetOrientationMarker(axes)

    # RenderWindow: The window on the screen
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    render_window.SetWindowName("VTK Mesh Viewer")

    # Interactor: Handles mouse and keyboard interaction
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Couple the axes widget to the interactor
    widget.SetInteractor(render_window_interactor)
    widget.SetEnabled(1)
    widget.InteractiveOn()

    # 4. Start the visualization
    render_window.Render()
    render_window_interactor.Start()


@click.command(name="view", help="Visualize meshes from an HDF5 database file.")
@click.argument("hdf5_file", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.argument("mesh_names", nargs=-1)
@click.option("--vtk", is_flag=True, help="Use the VTK viewer instead of the default trimesh viewer.")
@click.option("--show-all", is_flag=True, help="Visualize all meshes found in the specified files.")
def view(hdf5_file: str, mesh_names: tuple[str, ...], vtk: bool, show_all: bool) -> None:
    """
    CLI command to load and visualize meshes from HDF5 databases.

    HDF5_FILE: Path to the HDF5 database file.
    [MESH_NAMES]...: Optional names of meshes or cases to visualize.
    """
    # The HDF5 file is now a required positional argument.
    # Mesh names are optional positional arguments.
    names_to_resolve = set(mesh_names)
    resolved_mesh_names = set()

    if show_all:
        # If --show-all, we ignore any provided mesh names and find all meshes in the specified files.
        names_to_resolve = set()
        db_file = Path(hdf5_file)
        with h5py.File(db_file, "r") as f:
            resolved_mesh_names.update(f.get("meshes", {}).keys())

    elif names_to_resolve:
        # Resolve provided names: they can be mesh names or case names.
        with h5py.File(hdf5_file, "r") as f:
            for name in names_to_resolve:
                # Check if it's a direct mesh name
                if f.get(f"meshes/{name}"):
                    resolved_mesh_names.add(name)
                    logger.debug(f"Resolved '{name}' as a direct mesh name.")
                # Check if it's a case name
                elif (case_group := f.get(name)) and (mesh_name := case_group.attrs.get("stl_mesh_name")):
                    resolved_mesh_names.add(mesh_name)
                    logger.debug(f"Resolved case '{name}' to mesh '{mesh_name}'.")
                else:
                    click.echo(
                        f"❌ Warning: Could not resolve '{name}' as a mesh or a case name.",
                        err=True,
                    )

    if not resolved_mesh_names:
        click.echo("No mesh names provided and no meshes found with --show-all.", err=True)
        click.echo(
            "Usage: fleetmaster view <HDF5_FILE> [MESH_NAME...]  OR  fleetmaster view <HDF5_FILE> --show-all", err=True
        )
        return

    meshes = load_meshes_from_hdf5(Path(hdf5_file), sorted(resolved_mesh_names))
    if not meshes:
        click.echo("No valid meshes could be loaded from the database.", err=True)
        return

    if vtk:
        show_with_vtk(meshes)
    else:
        click.echo(f"🎨 Displaying {len(meshes)} mesh(es) with trimesh viewer. Close the window to continue.")
        # To avoid potential rendering glitches with the scene object,
        # we create a scene with an axis and pass the meshes to show directly.
        axis = trimesh.creation.axis(origin_size=0.1)
        scene = trimesh.Scene([axis, *meshes])

        logger.debug("Showing with solid mode. Toggle with w/s to go to wireframe")
        scene.show()
