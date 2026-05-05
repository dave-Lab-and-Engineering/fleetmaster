# fleetmaster

[![Release](https://img.shields.io/github/v/release/dave-Lab-and-Engineering/fleetmaster)](https://img.shields.io/github/v/release/dave-Lab-and-Engineering/fleetmaster)
[![Build status](https://img.shields.io/github/actions/workflow/status/dave-Lab-and-Engineering/fleetmaster/main.yml?branch=main)](https://github.com/dave-Lab-and-Engineering/fleetmaster/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/dave-Lab-and-Engineering/fleetmaster/branch/main/graph/badge.svg)](https://codecov.io/gh/dave-Lab-and-Engineering/fleetmaster)
[![Commit activity](https://img.shields.io/github/commit-activity/m/dave-Lab-and-Engineering/fleetmaster)](https://img.shields.io/github/commit-activity/m/dave-Lab-and-Engineering/fleetmaster)
[![License](https://img.shields.io/github/license/dave-Lab-and-Engineering/fleetmaster)](https://img.shields.io/github/license/dave-Lab-and-Engineering/fleetmaster)

A command-line tool to run batch processes of Capytaine simulations for hydrodynamic analysis.

- **Github repository**: <https://github.com/dave-Lab-and-Engineering/fleetmaster/>
- **Documentation** <https://dave-lab-and-engineering.github.io/fleetmaster/>

## Installation

Using `uv`:

```bash
uv sync
```

Run CLI commands with:

```bash
uv run fleetmaster --help
```

## Typical usage

Run a simulation batch from a YAML settings file:

```bash
uv run fleetmaster run --settings-file examples/settings_rotations.yml
```

Run directly from CLI options:

```bash
uv run fleetmaster run examples/boxship_t_1_r_00_00_00.stl --wave-periods 8 --wave-directions 0 --water-depth inf --water-level 0 --forward-speed 0
```

## NetCDF workflows

Fleetmaster supports two easy ways to create a standalone NetCDF file (`.nc`).

### 1) Write `.nc` while running simulations

Set `output_netcdf_file` in your settings file, or pass `--output-netcdf-file` on the CLI.

Example (settings file):

```yaml
output_hdf5_file: "single_case.hdf5"
output_netcdf_file: "single_case.nc"
```

Example (CLI):

```bash
uv run fleetmaster run examples/boxship_t_1_r_00_00_00.stl --wave-periods 8 --wave-directions 0 --water-depth inf --water-level 0 --forward-speed 0 --output-netcdf-file examples/single_case.nc
```

Notes:

- `output_netcdf_file` is intended for a single-case run (one mesh and one combination of `water_depth`, `water_level`, and `forward_speed`).
- If a case already exists in the HDF5 and `update_cases` is disabled, that case is skipped.

### 2) Export `.nc` from an existing HDF5 database (without rerunning Capytaine)

List cases in an existing HDF5:

```bash
uv run fleetmaster export-nc results.hdf5 --list-cases
```

Export one specific case:

```bash
uv run fleetmaster export-nc results.hdf5 --case boxship_wd_inf_wl_0_fs_0 --output boxship_case.nc
```

Overwrite an existing output file:

```bash
uv run fleetmaster export-nc results.hdf5 --case boxship_wd_inf_wl_0_fs_0 --output boxship_case.nc --overwrite
```

## Example settings

For a ready-to-use single-case example, see:

- `examples/settings_single_case_nc.yml`
