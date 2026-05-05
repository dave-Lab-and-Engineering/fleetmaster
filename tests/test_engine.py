from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import trimesh
import xarray as xr

from fleetmaster.core.engine import (
    EngineMesh,
    _format_value_for_name,
    _generate_case_group_name,
    _prepare_capytaine_body,
    _process_single_stl,
    _resolve_bulk_output_dhyd_path,
    _resolve_output_dhyd_path,
    _run_pipeline_for_mesh,
    _setup_output_file,
    _validate_single_case_dhyd_export,
    add_mesh_to_database,
    export_hdf5_case_to_dhyd,
    export_hdf5_cases_to_dhyd,
    list_case_groups_in_hdf5,
    run_simulation_batch,
)
from fleetmaster.core.exceptions import LidAndSymmetryEnabledError
from fleetmaster.core.settings import MeshConfig, SimulationSettings


@pytest.fixture
def mock_settings():
    """Provides a default SimulationSettings object for tests."""
    return SimulationSettings(
        stl_files=[MeshConfig(file="/path/to/dummy.stl")],  # Default value, can be overridden
        water_depth=np.inf,
        water_level=0.0,
        wave_periods=[1.0],
        wave_directions=[0.0],
    )


def test_setup_output_file_no_stl_files(mock_settings):
    """Test that _setup_output_file raises ValueError if no STL files are provided."""
    mock_settings.stl_files = []
    with pytest.raises(ValueError, match=r"No STL files provided to process\."):
        _setup_output_file(mock_settings)


def test_setup_output_file_overwrite(tmp_path, mock_settings):
    """Test that an existing output file is deleted if overwrite_meshes is True."""
    mock_settings.overwrite_meshes = True
    mock_settings.output_hdf5_file = "test.h5"
    mock_settings.output_directory = str(tmp_path)
    output_file = tmp_path / "test.h5"
    output_file.touch()  # Create the file

    result_path = _setup_output_file(mock_settings)

    assert not output_file.exists()  # Should have been deleted
    assert result_path == tmp_path / "test.h5"


def test_setup_output_file_no_overwrite(tmp_path, mock_settings):
    """Test that an existing output file is kept if overwrite_meshes is False."""
    mock_settings.overwrite_meshes = False
    mock_settings.output_hdf5_file = "test.h5"
    mock_settings.stl_files = [MeshConfig(file=str(tmp_path / "dummy.stl"))]
    output_file = tmp_path / "test.h5"
    output_file.touch()  # Create the file

    _setup_output_file(mock_settings)

    assert output_file.exists()  # Should NOT have been deleted


def test_setup_output_file_with_dict_as_stl(tmp_path):
    """Test _setup_output_file with stl_files as a list of dicts."""
    settings = MagicMock(spec=SimulationSettings)
    settings.stl_files = [{"file": str(tmp_path / "test.stl")}]
    settings.output_directory = None
    settings.output_hdf5_file = "results.hdf5"
    settings.overwrite_meshes = False

    # Act
    result_path = _setup_output_file(settings)

    # Assert
    # If no output dir is specified, it should be the parent of the first STL
    assert result_path == tmp_path / "results.hdf5"
    assert result_path.parent.exists()


@patch("fleetmaster.core.engine.cpt")
@patch("fleetmaster.core.engine.tempfile")
def test_prepare_capytaine_body(mock_tempfile, mock_cpt, tmp_path: Path):
    """Test _prepare_capytaine_body configures the body correctly."""
    # Arrange
    mock_source_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_source_mesh.center_mass = [1, 2, 3]
    mock_source_mesh.vertices = np.array([[1, 1, 1]])
    mock_source_mesh.faces = np.array([[0, 0, 0]])

    # Mock the temporary file creation
    mock_temp_file_handle = MagicMock()
    mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file_handle
    mock_temp_file_handle.name = str(tmp_path / "temp.stl")

    mock_hull_mesh = MagicMock()
    mock_cpt.load_mesh.return_value = mock_hull_mesh

    mock_body = MagicMock()
    mock_cpt.FloatingBody.return_value = mock_body

    mock_mesh_config = MagicMock(spec=MeshConfig)
    mock_mesh_config.cog = None

    mock_engine_mesh = EngineMesh(name="test_mesh", mesh=mock_source_mesh, config=mock_mesh_config)

    # Act
    body, final_mesh = _prepare_capytaine_body(
        engine_mesh=mock_engine_mesh,
        lid=True,
        grid_symmetry=False,
    )

    # Assert
    assert body == mock_body
    mock_cpt.FloatingBody.assert_called_once()
    # Check that the center of mass from the mesh was used as a fallback
    assert np.array_equal(mock_cpt.FloatingBody.call_args.kwargs["center_of_mass"], [1, 2, 3])
    mock_hull_mesh.generate_lid.assert_called_once()
    body.keep_immersed_part.assert_called_once_with(free_surface=0.0)
    body.add_all_rigid_body_dofs.assert_called_once()
    assert final_mesh is not None


@patch("fleetmaster.core.engine.cpt")
@patch("fleetmaster.core.engine.tempfile")
def test_prepare_capytaine_body_with_symmetry(mock_tempfile, mock_cpt, tmp_path: Path):
    """Test that grid_symmetry correctly wraps the mesh in ReflectionSymmetricMesh."""
    # Arrange
    mock_source_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_source_mesh.center_mass = [1, 2, 3]
    mock_source_mesh.vertices = np.array([[1, 1, 1]])
    mock_source_mesh.faces = np.array([[0, 0, 0]])

    mock_temp_file_handle = MagicMock()
    mock_tempfile.NamedTemporaryFile.return_value.__enter__.return_value = mock_temp_file_handle
    mock_temp_file_handle.name = str(tmp_path / "temp.stl")

    mock_base_hull_mesh = MagicMock()
    mock_cpt.load_mesh.return_value = mock_base_hull_mesh

    mock_symmetric_mesh = MagicMock()
    mock_cpt.ReflectionSymmetricMesh.return_value = mock_symmetric_mesh

    mock_mesh_config = MagicMock(spec=MeshConfig)
    mock_mesh_config.cog = None

    mock_engine_mesh = EngineMesh(name="test_mesh", mesh=mock_source_mesh, config=mock_mesh_config)

    # Act
    _prepare_capytaine_body(engine_mesh=mock_engine_mesh, lid=False, grid_symmetry=True)

    # Assert
    mock_cpt.ReflectionSymmetricMesh.assert_called_once_with(mock_base_hull_mesh, plane=mock_cpt.xOz_Plane)
    # Check that the FloatingBody was created with the symmetric mesh
    mock_cpt.FloatingBody.assert_called_once()
    assert mock_cpt.FloatingBody.call_args.kwargs["mesh"] == mock_symmetric_mesh


@patch("h5py.File")
def test_add_mesh_to_database_overwrite_warning(mock_h5py_file, caplog):
    """Test that a warning is logged when a different mesh with the same name exists and overwrite is False."""
    # Arrange
    mesh = trimesh.creation.box()

    mock_existing_group = MagicMock()
    mock_existing_group.attrs.get.return_value = "different_hash"
    mock_file = MagicMock()
    mock_file.__contains__.return_value = True
    mock_file.__getitem__.return_value = mock_existing_group
    mock_h5py_file.return_value.__enter__.return_value = mock_file

    # Act
    add_mesh_to_database(Path("test.h5"), mesh, "test_mesh", overwrite=False)

    # Assert
    assert "is different from the one in the database" in caplog.text
    mock_file.create_group.assert_not_called()


@pytest.mark.parametrize(
    "value, expected",
    [(10.0, "10"), (10.5, "10.5"), (10.55, "10.6"), (np.inf, "inf")],
)
def test_format_value_for_name(value, expected):
    assert _format_value_for_name(value) == expected


def test_generate_case_group_name():
    name = _generate_case_group_name("mesh1", 100.0, -2.5, 5.0)
    assert name == "mesh1_wd_100_wl_-2.5_fs_5"


def test_validate_single_case_dhyd_export_ok(mock_settings):
    mock_settings.output_dhyd_file = "single_case.dhyd"
    _validate_single_case_dhyd_export(mock_settings, mesh_count=1)


def test_validate_single_case_dhyd_export_ok_with_auto_export(mock_settings):
    mock_settings.export_to_hyd = True
    _validate_single_case_dhyd_export(mock_settings, mesh_count=1)


def test_validate_single_case_dhyd_export_fails_with_multiple_meshes(mock_settings):
    mock_settings.output_dhyd_file = "single_case.dhyd"
    _validate_single_case_dhyd_export(mock_settings, mesh_count=2)


def test_validate_single_case_dhyd_export_fails_with_multiple_case_values(mock_settings):
    mock_settings.output_dhyd_file = "single_case.dhyd"
    mock_settings.forward_speed = [0.0, 1.0]
    _validate_single_case_dhyd_export(mock_settings, mesh_count=1)


def test_resolve_output_dhyd_path_uses_explicit_filename(tmp_path: Path):
    output_file = tmp_path / "results.hdf5"
    explicit_output = tmp_path / "custom_name.dhyd"

    assert _resolve_output_dhyd_path(output_file, "case_a", explicit_output, False) == explicit_output


def test_resolve_output_dhyd_path_uses_explicit_filename_as_base_for_multiple_cases(tmp_path: Path):
    output_file = tmp_path / "results.hdf5"
    explicit_output = tmp_path / "custom_name.dhyd"

    assert (
        _resolve_output_dhyd_path(output_file, "case_a", explicit_output, True) == tmp_path / "custom_name_case_a.dhyd"
    )


def test_resolve_output_dhyd_path_generates_case_name(tmp_path: Path):
    output_file = tmp_path / "results.hdf5"

    assert _resolve_output_dhyd_path(output_file, "case_a", None, True) == tmp_path / "case_a.dhyd"


def test_resolve_bulk_output_dhyd_path_with_explicit_filename_for_multiple_cases(tmp_path: Path):
    hdf5_path = tmp_path / "results.hdf5"
    explicit_output = tmp_path / "exports.dhyd"

    assert (
        _resolve_bulk_output_dhyd_path(hdf5_path, "case_a", explicit_output, True) == tmp_path / "exports_case_a.dhyd"
    )


@patch("fleetmaster.core.engine.export_hdf5_case_to_dhyd")
def test_export_hdf5_cases_to_dhyd_exports_all_cases(mock_export_case: MagicMock, tmp_path: Path):
    hdf5_path = tmp_path / "db.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_group("meshes")
        f.create_group("case_b")
        f.create_group("case_a")

    mock_export_case.side_effect = [tmp_path / "case_a.dhyd", tmp_path / "case_b.dhyd"]

    result = export_hdf5_cases_to_dhyd(hdf5_path)

    assert result == [tmp_path / "case_a.dhyd", tmp_path / "case_b.dhyd"]
    assert mock_export_case.call_count == 2


def test_list_case_groups_in_hdf5(tmp_path: Path):
    hdf5_path = tmp_path / "db.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_group("meshes")
        f.create_group("case_b")
        f.create_group("case_a")

    assert list_case_groups_in_hdf5(hdf5_path) == ["case_a", "case_b"]


@patch("fleetmaster.core.engine.create_hyd_from_capytaine_data")
def test_export_hdf5_case_to_dhyd(mock_create_hyd: MagicMock, tmp_path: Path):
    hdf5_path = tmp_path / "db.hdf5"
    output_dhyd = tmp_path / "case.dhyd"

    dataset = xr.Dataset(
        data_vars={
            "added_mass": (
                ("omega", "radiating_dof", "influenced_dof"),
                np.ones((1, 1, 1), dtype=float),
            )
        },
        coords={
            "omega": [1.0],
            "radiating_dof": ["Heave"],
            "influenced_dof": ["Heave"],
        },
    )
    dataset.attrs["stl_mesh_name"] = "test_mesh"
    dataset.to_netcdf(hdf5_path, mode="w", group="test_case", engine="h5netcdf")

    mock_hyd = MagicMock()
    mock_create_hyd.return_value = mock_hyd

    result_path = export_hdf5_case_to_dhyd(hdf5_path, "test_case", output_dhyd)

    assert result_path == output_dhyd
    mock_hyd.save_as.assert_called_once_with(output_dhyd)


@patch("fleetmaster.core.engine._run_pipeline_for_mesh")
@patch("fleetmaster.core.engine.load_meshes_from_hdf5", return_value=[])
@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
@patch("pathlib.Path.exists", return_value=True)
def test_process_single_stl(mock_exists, mock_prepare, mock_load, mock_run_pipeline):
    """Test the main processing pipeline for a single STL file."""
    mesh_config = MeshConfig(file="/path/to/dummy.stl")
    settings = SimulationSettings(stl_files=[mesh_config])
    output_file = Path("/fake/output.hdf5")

    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_prepare.return_value = mock_mesh

    _process_single_stl(mesh_config, settings, output_file)

    mock_run_pipeline.assert_called_once()
    # Check that the EngineMesh passed to the pipeline has the correct components
    call_args = mock_run_pipeline.call_args[0]
    engine_mesh_arg = call_args[0]
    assert isinstance(engine_mesh_arg, EngineMesh)
    assert engine_mesh_arg.name == "dummy"
    assert engine_mesh_arg.mesh == mock_mesh
    assert engine_mesh_arg.config == mesh_config


def test_process_single_stl_lid_and_symmetry_error(mock_settings):
    """Test that LidAndSymmetryEnabledError is raised if both are enabled."""
    mock_settings.lid = True
    mock_settings.grid_symmetry = True
    mesh_config = MeshConfig(file="file.stl")
    # This test doesn't need to go deep into the pipeline, it should fail early.
    # We just need to check that _run_pipeline_for_mesh raises the error.
    with pytest.raises(LidAndSymmetryEnabledError):
        _run_pipeline_for_mesh(
            EngineMesh(name="test", mesh=MagicMock(), config=mesh_config),
            mock_settings,
            Path("out.h5"),
            origin_translation=None,
        )


@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
@patch("fleetmaster.core.engine._process_single_stl")
@patch("fleetmaster.core.engine._setup_output_file")
def test_run_simulation_batch_standard(mock_setup, mock_process, mock_prepare, mock_settings, tmp_path: Path):
    mock_settings.stl_files = [MeshConfig(file="file1.stl"), MeshConfig(file="file2.stl")]
    output_file = tmp_path / "output.hdf5"
    mock_setup.return_value = output_file

    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"dummy stl content"

    # --- FIX HIER TOEGEVOEGD ---
    # Configureer de mock om waarden terug te geven die de productiecode verwacht
    mock_mesh.moment_inertia = np.eye(3)
    mock_mesh.volume = 1.0  # Dit voorkomt de TypeError: MagicMock > float
    mock_mesh.center_mass = [0, 0, 0]
    mock_mesh.bounding_box.extents = [1, 1, 1]
    # --- EINDE FIX ---

    mock_prepare.return_value = mock_mesh

    run_simulation_batch(mock_settings)

    assert mock_process.call_count == 2


@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
@patch("fleetmaster.core.engine._process_single_stl")
@patch("fleetmaster.core.engine._setup_output_file", autospec=True)
def test_run_simulation_batch_drafts(mock_setup, mock_process, mock_prepare, mock_settings, tmp_path: Path):
    """Test run_simulation_batch in draft generation mode."""
    # Arrange
    mock_setup.return_value = tmp_path / "output.hdf5"
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"dummy stl content"

    # --- FIX HIER TOEGEVOEGD ---
    # Configureer de mock om waarden terug te geven die de productiecode verwacht
    mock_mesh.moment_inertia = np.eye(3)
    mock_mesh.volume = 1.0  # Dit voorkomt de TypeError: MagicMock > float
    mock_mesh.center_mass = [0, 0, 0]
    mock_mesh.bounding_box.extents = [1, 1, 1]
    # --- EINDE FIX ---

    mock_prepare.return_value = mock_mesh

    mock_settings.stl_files = [MeshConfig(file="base_mesh.stl", translation=[0, 0, 5])]
    mock_settings.drafts = [1.0, 2.5]

    # Act
    run_simulation_batch(mock_settings)

    # Assert
    assert mock_process.call_count == 2
    # Check the first call for draft 1.0
    call1 = mock_process.call_args_list[0]
    config_draft1 = call1.args[0]
    assert isinstance(config_draft1, MeshConfig)
    assert config_draft1.translation == [0.0, 0.0, 4.0]  # 5.0 - 1.0
    assert call1.kwargs["mesh_name_override"] == "base_mesh_draft_1"

    # Check the second call for draft 2.5
    call2 = mock_process.call_args_list[1]
    config_draft2 = call2.args[0]
    assert config_draft2.translation == [0.0, 0.0, 2.5]  # 5.0 - 2.5
    assert call2.kwargs["mesh_name_override"] == "base_mesh_draft_2.5"


@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
def test_run_simulation_batch_drafts_wrong_stl_count(mock_prepare, mock_settings):
    """Test that draft mode raises an error if more than one STL is provided."""
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"dummy stl content"
    mock_prepare.return_value = mock_mesh

    mock_settings.drafts = [1.0]
    mock_settings.stl_files = [MeshConfig(file="file1.stl"), MeshConfig(file="file2.stl")]
    with pytest.raises(ValueError, match="exactly one base STL file must be provided"):
        run_simulation_batch(mock_settings)


@patch("h5py.File")
def test_add_mesh_to_database_overwrite_true_different_content(mock_h5py_file, caplog):
    """Test that with overwrite=True, a different mesh with the same name is replaced."""
    # Arrange
    mock_file = MagicMock()
    mock_h5py_file.return_value.__enter__.return_value = mock_file

    mock_group = MagicMock()
    mock_group.name = "/meshes/test_mesh"
    mock_file.__contains__.return_value = True
    mock_file.__getitem__.return_value = mock_group
    mock_group.attrs.get.return_value = "old_hash"

    output_file = Path("dummy.h5")
    mesh_config = MeshConfig(file="test.stl", name="test_mesh")
    new_mesh = trimesh.creation.box()

    # Act
    add_mesh_to_database(output_file, new_mesh, "test_mesh", overwrite=True, mesh_config=mesh_config)

    # Assert
    mock_file.__delitem__.assert_called_once_with("meshes/test_mesh")
    assert "Overwriting existing mesh" in caplog.text
    mock_file.create_group.assert_called_with("meshes/test_mesh")


@patch("fleetmaster.core.engine._process_single_stl")
@patch("fleetmaster.core.engine._setup_output_file")
def test_run_simulation_batch_with_mesh_config_dict(mock_setup, mock_process, tmp_path):
    """Test run_simulation_batch with stl_files as a list of dicts."""
    settings_dict = {
        "stl_files": [{"file": "box.stl", "translation": [1, 2, 3]}],
        "output_directory": str(tmp_path),
        "output_hdf5_file": "results.h5",
        "overwrite_meshes": True,
        "drafts": [],
        "base_mesh": None,
        "base_origin": None,
        "wave_periods": [10.0],
        "wave_directions": [0.0],
        "water_depth": 100.0,
        "water_level": 0.0,
        "forward_speed": 0.0,
        "grid_symmetry": False,
        "lid": False,
        "add_center_of_mass": False,
        "update_cases": False,
        "combine_cases": False,
    }

    stl_file = tmp_path / "box.stl"
    stl_file.touch()
    settings_dict["stl_files"][0]["file"] = str(stl_file)
    mock_setup.return_value = tmp_path / "results.h5"

    settings = SimulationSettings(**settings_dict)

    run_simulation_batch(settings)

    mock_process.assert_called_once()
    called_mesh_config = mock_process.call_args[0][0]
    assert isinstance(called_mesh_config, MeshConfig)
    assert called_mesh_config.file == str(stl_file)
    assert called_mesh_config.translation == [1, 2, 3]


@patch("fleetmaster.core.engine._process_single_stl")
@patch("h5py.File")
@patch("fleetmaster.core.engine._setup_output_file")
@patch("fleetmaster.core.engine._prepare_trimesh_geometry")
def test_run_simulation_batch_with_base_origin(mock_prepare, mock_setup, mock_h5py, mock_process_stl, tmp_path):
    """Test run_simulation_batch with base_origin set."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    stl_path = tmp_path / "test.stl"
    stl_path.touch()
    mock_setup.return_value = output_dir / "results.h5"

    mock_prepare.return_value = trimesh.creation.box()

    settings = SimulationSettings(
        stl_files=[MeshConfig(file=str(stl_path))],
        base_mesh=str(stl_path),
        base_origin=[1.0, 2.0, 3.0],
        output_directory=str(output_dir),
        wave_periods=[10.0],
        wave_directions=[0.0],
        water_depth=100.0,
        water_level=0.0,
        forward_speed=0.0,
        grid_symmetry=False,
        lid=False,
        add_center_of_mass=False,
        update_cases=False,
        combine_cases=False,
    )

    mock_file = MagicMock()
    mock_h5py.return_value.__enter__.return_value = mock_file

    run_simulation_batch(settings)

    # Check that origin_translation is passed correctly
    mock_process_stl.assert_called_once()
    kwargs = mock_process_stl.call_args.kwargs
    assert "origin_translation" in kwargs
    np.testing.assert_array_equal(kwargs["origin_translation"], np.array([1.0, 2.0, 3.0]))

    # Check that base_origin is saved to HDF5 file attributes
    mock_file.attrs.__setitem__.assert_any_call("base_origin", [1.0, 2.0, 3.0])


@patch("h5py.File")
def test_add_mesh_to_database_with_meshconfig_attrs(mock_h5py_file, tmp_path):
    """Test that MeshConfig attributes are saved to the HDF5 group."""
    # Arrange
    mock_file = MagicMock()
    mock_group = MagicMock()
    mock_h5py_file.return_value.__enter__.return_value = mock_file
    mock_file.create_group.return_value = mock_group

    output_file = tmp_path / "db.h5"
    mesh_config = MeshConfig(
        file="test.stl",
        name="test_mesh",
        translation=[1, 1, 1],
        rotation=[2, 2, 2],
        cog=[3, 3, 3],
    )
    mock_mesh = MagicMock(spec=trimesh.Trimesh)
    mock_mesh.export.return_value = b"content"
    mock_mesh.moment_inertia = np.eye(3)
    mock_mesh.volume = 1.0
    mock_mesh.center_mass = [0, 0, 0]
    mock_mesh.bounding_box.extents = [1, 1, 1]
    mock_mesh.is_empty = False  # Ensure the mock mesh is not considered empty

    # Act
    add_mesh_to_database(output_file, mock_mesh, "test_mesh", mesh_config=mesh_config)

    # Assert
    mock_file.create_group.assert_called_once_with("meshes/test_mesh")
    mock_group.attrs.__setitem__.assert_any_call("translation", [1, 1, 1])
    mock_group.attrs.__setitem__.assert_any_call("rotation", [2, 2, 2])
    mock_group.attrs.__setitem__.assert_any_call("cog", [3, 3, 3])


@patch("fleetmaster.core.engine._load_or_generate_mesh")
@patch("fleetmaster.core.engine.load_meshes_from_hdf5")
@patch("fleetmaster.core.engine._run_pipeline_for_mesh")
def test_process_single_stl_from_db(mock_run_pipeline, mock_load_meshes, mock_load_or_generate, tmp_path):
    """Test _process_single_stl when the mesh is loaded from the database."""
    # Arrange
    settings = MagicMock(spec=SimulationSettings)
    settings.overwrite_meshes = False

    # The name of the mesh is derived from the file name stem
    mesh_config = MeshConfig(file="dummy_name.stl")
    output_file = tmp_path / "db.h5"

    mock_mesh_from_db = MagicMock(spec=trimesh.Trimesh)
    mock_load_meshes.return_value = [mock_mesh_from_db]

    # Act
    _process_single_stl(mesh_config, settings, output_file)

    # Assert
    # Should attempt to load from DB first
    mock_load_meshes.assert_called_once_with(output_file, ["dummy_name"])

    # Should NOT try to load or generate a new mesh because it was found in the DB
    mock_load_or_generate.assert_not_called()

    # Should run the pipeline with the mesh from the DB
    mock_run_pipeline.assert_called_once()
    engine_mesh_arg = mock_run_pipeline.call_args[0][0]
    assert isinstance(engine_mesh_arg, EngineMesh)
    assert engine_mesh_arg.mesh == mock_mesh_from_db
    assert engine_mesh_arg.name == "dummy_name"
    assert engine_mesh_arg.config == mesh_config
