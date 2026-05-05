# General Description of `fleetmaster`

## Summary

**Fleetmaster is a data-storage structure for hydrodynamic data such as produced by Capytaine.**

**It allows multiple hydrodynamic databases to be stored together and provides methods for extracting the most suitable one.**

**Intended use-case is the storage of hydrodynamic data for a single vessel but at different drafts, inclinations and/or forwards speeds.**

**Fleetmaster includes convenience functions and command line tools for creating such datasets using capytaine.**

`fleetmaster` is a command-line tool designed to simplify running batch processes with [Capytaine](https://capytaine.github.io/), an open-source Python library for simulating wave-structure interactions. While Capytaine provides powerful tools for hydrodynamic analysis, `fleetmaster` streamlines the process of running multiple simulations with varying parameters, managing inputs, and organizing outputs.

It acts as a wrapper, allowing users to define a fleet of simulations in a structured way and execute them with a single command.

## Core Concepts

The main goal of `fleetmaster` is to automate the execution of multiple hydrodynamic simulations using Capytaine. This is achieved through a few core concepts:

- **Settings File**: The user defines a batch of simulations using a YAML settings file. This file specifies the mesh files, water depth, wave directions, and other parameters for each case to be run.
- **Batch Engine**: The core engine of `fleetmaster` reads the settings file, prepares each individual Capytaine simulation, runs it, and stores the results in an HDF5 database.
- **Command-Line Interface (CLI)**: All operations are handled through the `fleetmaster` command. This allows for easy integration into scripts and automated workflows.

## Solution Database

The solution for each mesh and set of simulation settings are stored in a database. The database can subsequently be used by external programs to quickly access each Capytaine solution. For more details on the database, see [Database](./database.md).

## Mesh Fitting

In addition to running batch simulations, `fleetmaster` also provides a powerful mesh fitting capability. This feature allows you to find the best-matching mesh from a database of pre-calculated meshes based on a target transformation (translation and rotation). **= Draft, heel, trim**

**Consider the following case:**

**Target draft: 3m, target roll 0, target pitch 0**

**Option1:**

**draft 3m, pitch 1 degree, roll 0 degrees**

**Option2:**

**draft 2.8m, pitch 0 degrees, roll 0 degrees**

**which one should we use?**

This is particularly useful for finding the most relevant hydrodynamic data for a specific loading condition without running a new simulation.

For more details, see the [Mesh Fitting](./fitting.md) documentation.

## Typical Workflow

A typical workflow for using `fleetmaster` involves the following steps:

1.  **Prepare Meshes**: Create or obtain the mesh files (e.g., `.obj`, `.stl`) for the floating bodies you want to analyze.
2.  **Create a Settings File**: Write a YAML file that defines the parameters for your batch of simulations. This includes pointing to the mesh files and specifying the desired environmental conditions.
3.  **Run `fleetmaster`**: Execute the tool from your terminal, pointing it to your settings file.
4.  **Analyze Results**: `fleetmaster` will generate an HDF5 file containing the hydrodynamic data for each simulation in the batch.

Below is a diagram illustrating this workflow.

```mermaid
graph LR
    A[Prepare Meshes] --> B[Create Settings File];
    B --> C[Run fleetmaster];
    C --> D[Analyze Results];
```
