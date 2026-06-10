# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.3] - 2026-06-10

### Added

- A new `plot` command for visualizing the theta/period grid of a simulation case.
- Export support for additional output formats, including `.dhyd`, `.nc`, and `.stl`.
- A new high-level `FleetMaster` API for working with databases, fitting meshes, and retrieving the best match.
- New examples and documentation for the GUI, fitting workflow, and export-related features.

### Changed

- The core engine and fitting pipeline were substantially refactored to support mesh fitting, improved stability handling, symmetry options, multirange support, and better mesh transformations.
- Database loading and mesh handling were expanded so the stored geometry matches the final immersed mesh used by Capytaine.
- Settings and CLI handling were updated to support the broader feature set and new output types.

### Fixed

- Improved compatibility with NetCDF/h5netcdf workflows and `mafredo`-related data handling.
- Fixed several typing, linting, and static-analysis issues.
- Resolved bugs around mesh clipping, export paths, and plotting/writing workflows.

## [0.2.1] - 2025-10-23

### Added

- `list` command can now show a detailed summary of simulation cases with the `--cases` flag, including mesh properties like volume, COG, and cell count.
- `view` command can now visualize a mesh by specifying its case name, in addition to its direct mesh name.

### Changed

- **Breaking Change**: The `view` command interface has been simplified. It now takes the HDF5 file as the first required argument, followed by optional mesh/case names. The `--file` option has been removed for clarity.
- Refactored the `list` command for improved code clarity and maintainability.
- The mesh stored in the database is now the final, immersed, and (if applicable) fully reflected mesh used by Capytaine, ensuring `view` shows the correct geometry.

### Fixed

- Resolved numerous bugs in the `run` command related to mesh transformations, temporary file handling on Windows (`PermissionError`, `OSError`), and the use of symmetric meshes.
- Corrected `isinstance` checks in unit tests when using mocked objects.
- Fixed all `mypy` and `deptry` static analysis errors for a cleaner codebase.

## [0.1.1] - 2025-10-21

### Added

- First implementation of Fleetmaster.
- Core hydrodynamic simulation engine.
- CLI and GUI entrypoints.
- Configuration model with validation.
- Initial documentation, tests, and CI setup.
