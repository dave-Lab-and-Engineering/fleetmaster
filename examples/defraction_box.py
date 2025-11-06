"""
Small use case script to generate meshes for a simple box-shaped hull.

This script generates:
- A buoyancy mesh (`_buoy.stl`) with its origin at the stern.
- A base mesh (`.stl`) for wave interaction analysis in Capytaine, centered at the origin.
- Two wetted surface meshes (`_<draft>m.stl`) for two different drafts.

Run with --grid-symmetry to generate half-meshes for symmetry analysis.

Requires the package pymeshup to be installed.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from pymeshup import Box
from trimesh.transformations import compose_matrix

# Constants for the box dimensions
BOX_LENGTH = 10
BOX_WIDTH = 4
BOX_HEIGHT = 3
DRAFTS = [1, 2]
REGRID_PERCENTAGE = 3
FILE_BASE = "defraction_box"


def main(
    grid_symmetry: bool,
    output_dir: Path,
    file_base: str,
    only_base: bool = False,
    generate_fitting_meshes: bool = False,
):
    """
    Generates STL meshes for a defraction box based on specified parameters.

    Args:
        grid_symmetry (bool): If True, enables grid symmetry, cutting the base mesh
                              at the xz plane (port/starboard symmetry).
        output_dir (Path): The directory where the generated STL files will be saved.
        file_base (str): The base name for the generated STL files.
        only_base (bool): If True, only the base mesh will be generated.
        generate_fitting_meshes (bool): If True, generates specific STL files for the fitting example.
    """
    if grid_symmetry:
        print(f"Grid symmetry on with file base {file_base}")
    else:
        print(f"Grid symmetry off with file base {file_base}")

    output_dir.mkdir(exist_ok=True)

    box_buoy_filename = output_dir / f"{file_base}_buoy.stl"
    box_base_filename = output_dir / f"{file_base}.stl"

    half_width = BOX_WIDTH / 2
    half_length = BOX_LENGTH / 2

    # create the buoy mesh with the stern at x = 0
    box_buoy = Box(0, BOX_LENGTH, -half_width, half_width, 0, BOX_HEIGHT)
    print(f"Saving buoy esh {box_buoy_filename}")
    box_buoy.save(str(box_buoy_filename))

    # create the base mesh for the wave interaction with x = 0 in the centre
    box_base = box_buoy.move(x=-half_length)
    if grid_symmetry:
        print("Cutting at xy plane")
        box_base = box_base.cut_at_xz()
    box_base_mesh = box_base.regrid(pct=REGRID_PERCENTAGE)
    print(f"Saving base mesh {box_base_filename}")
    box_base_mesh.save(str(box_base_filename))

    if only_base:
        return

    for draft in DRAFTS:
        # Start from the original buoy and move it, similar to how box_base was created.
        box_draft = box_buoy.move(x=-half_length, z=-draft)
        box_draft = box_draft.cut_at_waterline()
        box_draft_mesh = box_draft.regrid(pct=REGRID_PERCENTAGE)
        box_draft_filename = output_dir / f"{file_base}_{draft}m.stl"
        print(f"Saving draft mesh {box_draft_filename}")
        box_draft_mesh.save(str(box_draft_filename))

    if generate_fitting_meshes:
        generate_fitting_stl_files(output_dir, file_base)


def generate_fitting_stl_files(output_dir: Path, file_base: str):
    """
    Generates specific rotated and translated STL meshes required by settings_rotations.yml
    for the fitting example. These meshes are based on the full 'boxship.stl' and then transformed.
    """
    base_stl_path = output_dir / f"{file_base}.stl"
    if not base_stl_path.exists():
        print(
            f"Error: Base mesh '{base_stl_path}' not found. "
            "Please ensure it's generated first (e.g., by running with --only-base)."
        )
        return

    print(f"\n--- Generating fitting example STL files based on '{base_stl_path.name}' ---")
    loaded_mesh = trimesh.load(base_stl_path)

    # trimesh.load can return a Scene object or None. We need a single Trimesh object.
    if isinstance(loaded_mesh, trimesh.Scene):
        # Combine all geometries in the scene into a single mesh
        base_mesh_untransformed = loaded_mesh.dump(concatenate=True)
    else:
        base_mesh_untransformed = loaded_mesh

    if not isinstance(base_mesh_untransformed, trimesh.Trimesh) or base_mesh_untransformed.is_empty:
        print(f"Error: Failed to load a valid mesh from '{base_stl_path}'. Aborting fitting mesh generation.")
        return

    # The `translation` in the original settings_rotations.yml is the desired position of the mesh's geometric center
    # relative to the database origin. We bake this transformation directly into the STL.
    fitting_cases = [
        ("boxship_t_1_r_00_00_00.stl", -1.0, 0.0, 0.0, 0.0),
        ("boxship_t_2_r_00_00_00.stl", -2.0, 0.0, 0.0, 0.0),
        ("boxship_t_1_r_45_00_00.stl", -1.0, 45.0, 0.0, 0.0),
        ("boxship_t_1_r_00_10_00.stl", -1.0, 0.0, 10.0, 0.0),
        ("boxship_t_1_r_20_20_00.stl", -1.0, 20.0, 20.0, 0.0),
    ]

    for filename, target_z_rel_db_origin, roll_deg, pitch_deg, yaw_deg in fitting_cases:
        # The absolute translation is the target position relative to the database origin.
        translation_vec = [0.0, 0.0, target_z_rel_db_origin]

        transform_matrix = compose_matrix(angles=np.radians([roll_deg, pitch_deg, yaw_deg]), translate=translation_vec)

        transformed_mesh = base_mesh_untransformed.copy()
        transformed_mesh.apply_transform(transform_matrix)

        output_path = output_dir / filename
        transformed_mesh.export(str(output_path))
        print(f"Generated: {output_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate STL meshes for a defraction box, with optional grid symmetry."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="The directory to save the generated STL files.",
    )
    parser.add_argument("-f", "--file-base", type=str, default=FILE_BASE, help="The file base name for the STL files.")
    parser.add_argument(
        "--grid-symmetry",
        action="store_true",
        help="Enable grid symmetry (cuts the base mesh at the xz plane).",
    )
    parser.add_argument(
        "--only-base",
        action="store_true",
        help="Only generate the base mesh.",
    )
    parser.add_argument(
        "--generate-fitting-meshes",
        action="store_true",
        help="Generate specific STL files for the fitting example based on settings_rotations.yml.",
    )
    args = parser.parse_args()
    main(
        grid_symmetry=args.grid_symmetry,
        output_dir=args.output_dir,
        file_base=args.file_base,
        only_base=args.only_base,
        generate_fitting_meshes=args.generate_fitting_meshes,
    )
