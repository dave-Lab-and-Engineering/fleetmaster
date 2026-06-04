# Hydrodynamic Database GUI

## Purpose

This document specifies the Fleetmaster GUI for inspecting, matching, generating, and editing hydrodynamic database solutions.

The GUI has two goals:

- provide a standalone expert workflow in Fleetmaster,
- define a layout and interaction model that can later be reimplemented in DAVE.

## Scope

In scope:

- inspecting candidate meshes and case metadata,
- running match-or-generate for a target state,
- comparing matching strategies,
- editing database entries (add, delete, recompute, relabel),
- visualizing geometry and waterline context in VTK.

Out of scope:

- full scenario setup for non-hydrodynamic modules,
- replacing the core Fleetmaster solver API.

## Architecture

### Module split

- `fleetmaster.gui.hyddb.window`: main Qt window.
- `fleetmaster.gui.hyddb.state_panel`: target state controls and strategy settings.
- `fleetmaster.gui.hyddb.case_table`: case list, filtering, sorting, bulk actions.
- `fleetmaster.gui.hyddb.vtk_view`: VTK viewport and overlays.
- `fleetmaster.gui.hyddb.actions`: command handlers for add/delete/recompute/match.
- `fleetmaster.gui.hyddb.service`: adapter to Fleetmaster core APIs.
- `fleetmaster.gui.hyddb.inspect_dialog`: per-candidate inspection dialog with solution fragments.

### Integration boundary

- GUI calls Fleetmaster core service methods.
- GUI does not implement matching math itself.
- Core exposes `match_or_generate(...)` and `database_mutation(...)` APIs.

## Main user flows

### 1. Inspect existing solutions

1. Open HDF5 file.
2. Load candidate meshes and case entries.
3. Select a row in case table.
4. Show selected candidate in VTK view.
5. Show metadata and provenance.

### 2. Match or generate

1. Enter target state (draft, heel, pitch, optional extras).
2. Select matching strategy and thresholds.
3. Click "Match or Generate".
4. Fleetmaster returns selected existing solution or computes and stores a new one.
5. GUI highlights returned solution and refreshes diagnostics.

### 3. Edit database

1. Select one or more entries.
2. Use action toolbar:
   - Add from computed result,
   - Delete selected,
   - Recompute selected,
   - Update tags/metadata.
3. Confirm destructive operations.
4. Refresh table and provenance panel.

## Matching strategy support

The GUI must expose strategy choice:

- `mesh_distance` (existing algorithm),
- `wip_z_error` (new WIP-based score),
- `hybrid` (reserved, disabled until implemented).

For each match call, show:

- strategy used,
- best candidate id,
- score value,
- acceptance threshold,
- decision (`reused` or `generated`).

## Window layout

Recommended layout:

- Top bar: database path, open/reload, schema status.
- Left panel: candidate mesh list, filtering, and sort controls.
- Center: primary VTK viewport with base mesh and transformed base mesh.
- Right panel: selected entry metadata, diagnostics, and action buttons.
- Bottom panel: operation log and match trace.

The central viewport is the primary working area and must remain visible at all times.

The viewport coordinate convention is fixed as follows:

- `z = 0` is the waterplane.
- `z >= 0` is above water.
- `z < 0` is below water.
- positive `x` points stern to bow.
- positive `y` points to port side.

## VTK requirements

The VTK view must support:

- base mesh rendering,
- transformed base mesh rendering for the current target state,
- selected candidate mesh rendering alongside the target mesh,
- waterline plane visualization,
- WIP point markers and labels,
- color-by-distance or score heat cue (optional),
- camera presets: iso, side, front, top.

When the user selects a candidate mesh, the viewport must show the candidate together with the target mesh so the deviation is visible directly in the same scene.

Candidate meshes must be sortable in the left panel by distance to the target mesh.

The active sort mode must be visible in the UI, and the currently closest candidate should be easy to identify.

The viewport may show multiple candidates at once for comparison, but the selected candidate must always be highlighted.

## Candidate Mesh List

The left-side candidate list must contain at least:

- candidate id,
- distance to target mesh,
- fit strategy result,
- draft / heel / pitch state,
- solver status.

List behavior:

- sort ascending by distance by default,
- allow sort toggle between distance, state, and solver status,
- support single-click selection and double-click inspection,
- support refresh after generation or deletion.

## Inspect Dialog

Clicking a candidate mesh must open a separate inspect dialog.

The inspect dialog must show:

- a focused visualization of the selected candidate solution,
- the transformed target mesh for comparison,
- solution fragments or subcomponents from Capytaine output,
- metadata for the selected candidate,
- score breakdown and matching reason,
- a close button and a link back to the main window selection.

The inspect dialog is for deep inspection only and must not replace the main window.

## Data mutation rules

- All write operations must go through Fleetmaster core APIs.
- Writes must be atomic at case-entry level.
- UI must block conflicting write actions while a write is running.
- Delete must require explicit confirmation dialog.

## Error handling

Display clear error states for:

- unreadable/missing HDF5 file,
- schema mismatch,
- failed match operation,
- failed generation,
- failed mutation (add/delete/recompute),
- invalid target-state input.

All errors should include:

- operation name,
- concise reason,
- suggested next action.

## Logging and traceability

GUI operation log must capture:

- timestamp,
- user action,
- strategy and threshold values,
- selected/generated solution id,
- duration,
- success/failure.

## Acceptance criteria

1. User can open HDF5 database and inspect candidate entries.
2. User can run `mesh_distance` match and obtain deterministic result.
3. User can run `wip_z_error` match and obtain deterministic result.
4. If no acceptable candidate exists, `match_or_generate` creates and returns a new solution.
5. User can delete and recompute entries with confirmation.
6. VTK view updates correctly when selection changes.

## Implementation sequence

1. Build service adapter and table model.
2. Build basic window with file open, table, and metadata panel.
3. Add VTK viewport integration.
4. Add strategy controls and `match_or_generate` action.
5. Add mutation actions with confirmations.
6. Add smoke tests and API integration tests.
