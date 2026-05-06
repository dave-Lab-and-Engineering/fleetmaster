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

## Dhyd workflows

Fleetmaster supports two easy ways to create a standalone mafredo hydrodynamic database (`.dhyd`).

### 1) Write `.dhyd` while running simulations

Set `output_dhyd_file` in your settings file, or pass `--output-dhyd-file` on the CLI. If you only want
auto-generated files based on the case names, use `export_to_hyd` or `--export-to-hyd`.

Example (settings file):

```yaml
output_hdf5_file: "single_case.hdf5"
output_dhyd_file: "single_case.dhyd"
```

Example (settings file, automatic filename):

```yaml
output_hdf5_file: "single_case.hdf5"
export_to_hyd: true
```

Example (CLI):

```bash
uv run fleetmaster run examples/boxship_t_1_r_00_00_00.stl --wave-periods 8 --wave-directions 0 --water-depth inf --water-level 0 --forward-speed 0 --output-dhyd-file examples/single_case.dhyd
```

Example (CLI, automatic filename based on case):

```bash
uv run fleetmaster run examples/boxship_t_1_r_00_00_00.stl --wave-periods 8 --wave-directions 0 --water-depth inf --water-level 0 --forward-speed 0 --export-to-hyd
```

Notes:

- `export_to_hyd` writes a `.dhyd` file for every case while still writing all results to the HDF5 database.
- Without an explicit `output_dhyd_file`, Fleetmaster uses `<case_name>.dhyd` next to the HDF5 output file.
- With an explicit `output_dhyd_file` and multiple cases, Fleetmaster uses that filename as a base and appends the case name.
- If a case already exists in the HDF5 and `update_cases` is disabled, that case is skipped.

### 2) Inspect cases or plot a theta/period grid

List simulation cases:

```bash
uv run fleetmaster list results.hdf5 --cases
```

Create a quick grid plot for one case:

```bash
uv run fleetmaster plot results.hdf5 boxship_wd_inf_wl_0_fs_0
```

Write the plot to a specific PNG:

```bash
uv run fleetmaster plot results.hdf5 boxship_wd_inf_wl_0_fs_0 --output boxship_grid.png
```

Create additional hydrodynamic plots using mafredo (enabled by default):

```bash
uv run fleetmaster plot results.hdf5 boxship_wd_inf_wl_0_fs_0 --show
```

Save the mafredo hydrodynamic figures as PNG files:

```bash
uv run fleetmaster plot results.hdf5 boxship_wd_inf_wl_0_fs_0 --save-hyd-plots
```

Write mafredo hydrodynamic figures to a custom directory:

```bash
uv run fleetmaster plot results.hdf5 boxship_wd_inf_wl_0_fs_0 --save-hyd-plots --hyd-output-dir output/hyd_plots
```

Disable mafredo hydrodynamic plotting and keep only the theta/period grid plot:

```bash
uv run fleetmaster plot results.hdf5 boxship_wd_inf_wl_0_fs_0 --no-hyd-plot
```

### 3) Export `.dhyd` from an existing HDF5 database (without rerunning Capytaine)

List cases in an existing HDF5:

```bash
uv run fleetmaster export-dhyd results.hdf5 --list-cases
```

Export one specific case:

```bash
uv run fleetmaster export-dhyd results.hdf5 --case boxship_wd_inf_wl_0_fs_0 --output boxship_case.dhyd
```

Export all cases in one go:

```bash
uv run fleetmaster export-dhyd results.hdf5
```

Overwrite an existing output file:

```bash
uv run fleetmaster export-dhyd results.hdf5 --case boxship_wd_inf_wl_0_fs_0 --output boxship_case.dhyd --overwrite
```

## Example settings

For a ready-to-use single-case example, see:

- `examples/settings_single_case_nc.yml`
