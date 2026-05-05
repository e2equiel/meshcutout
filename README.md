# Mesh Cutout Generator for Blender

Blender add-on for generating fitted storage cavities and boxes from selected STL meshes.

The intended workflow is:

1. Import one or more STL models into Blender.
2. Rotate, scale, and place them as needed.
3. Lay the models down on the Z plane in the orientation you want for storage.
4. Select the mesh objects.
5. Run the add-on from `View3D > Sidebar > Mesh Cutout`.
6. Generate either separate cavity/cutter meshes or one complete box with one cavity per selected object.

The add-on uses Blender for selection, STL import/export, object placement, and UI. The cavity generation runs through the Python scripts in this repository, which keep the more robust sweep/projection pipeline outside Blender's internal Python environment.

## Features

- Generate an `XYZ Entry` or `sweep-z` insertion cavity for each selected Blender mesh object.
- Generate one complete fitted box containing one cutout per selected object.
- Process every selected object as its own cutout source. The add-on never combines multiple selected objects into one shared cavity.
- Add an optional rounded `Finger Scoop` side opening to each cavity.
- Keep the selected objects' world-space placement.
- Configure cavity clearance, sweep resolution, sweep safety overcut, simplification, box margin, top overlap, and custom box dimensions.
- Clamp requested box dimensions to the minimum required by the cutout and margin.
- Optional debug STL export for intermediate geometry.

## Requirements

- Blender 4.x.
- Python 3.10+ or 3.11+.
- Python dependencies from `requirements.txt`.

The default boolean backend is `manifold3d` through Trimesh. PyMeshLab is required by the default final simplification step and by `Offset Mode: Surface`.

## Installation

Clone this repository and create a Python virtual environment:

```bash
cd /path/to/meshcutout
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the Blender add-on:

1. Open Blender.
2. Go to `Edit > Preferences > Add-ons`.
3. Click `Install from Disk...`.
4. Select `meshcutout_blender.py` from this repository.
5. Enable `Mesh Cutout Generator`.
6. In the add-on panel or preferences, set `Scripts Directory` to this repository folder.
7. Set `Python Executable` to the virtual environment Python, for example:

```text
/path/to/meshcutout/.venv/bin/python
```

On Windows this will usually be:

```text
C:\path\to\meshcutout\.venv\Scripts\python.exe
```

Press `Check Setup` in the add-on panel to verify the configured Python and dependencies.

## Basic Usage

1. Import your STL files into Blender.
2. Arrange them in the exact orientation you want for storage.
3. Select the mesh object or objects.
4. Open `View3D > Sidebar > Mesh Cutout`.
5. Use `Min Z by Y` if each selected object should rotate around world Y until its Z height is minimized.
6. Use `Align Z Tops` if selected objects should share the same highest Z top.
7. Use `Precalc Box Size` to fill custom box dimensions from the selected object bounds and current margin.
8. Choose `Output: Box With Cutout` or `Output: Cavity Only`.
9. Click `Generate Mesh Cutout`.

If multiple mesh objects are selected:

- `Cavity Only` creates one cavity mesh per selected object.
- `Box With Cutout` creates one shared box with one cavity per selected object.

For models that you already oriented in Blender, keep `Orientation` set to `Use Selection`.

Most tuning controls are collapsed under `Advanced` in the add-on panel. The default view keeps only preparation, output, box sizing, clearance, cavity method, and generation controls visible.

## Box Settings

`Box Margin` defaults to `2 mm`.

- In X and Y, the margin is applied on both sides.
- In Z, the box is built below the cutout so the top remains open.
- `Top Overlap` moves the cutter through the top face to avoid boolean tolerance issues.

Custom dimensions can be enabled with `Custom Box Size`. If a requested dimension is smaller than the cutout plus the required margin, the script automatically clamps it to the minimum safe size. Enable `Strict Dimensions` if you prefer the operation to fail instead.

`Min Z by Y` processes each selected mesh separately. It samples the evaluated mesh vertices, searches the best world-Y rotation, and applies the rotation around that object's current world-space bounds center.

`Precalc Box Size` uses the current selected object world-space bounds plus the configured `Finger Scoop` estimate when enabled. It does not generate cavities first, so the final generated cutouts may still require the normal minimum-size clamp if clearance or entry settings make them larger.

## Cavity Settings

Important parameters:

- `Clearance`: storage tolerance around the figure. Default: `0.2 mm`.
- `Cavity Method`: `XYZ Entry` uses projection booleans plus the upper Z-entry piece and does not voxelize the result. `Sweep Z` keeps the older voxel sweep method.
- `Finger Scoop`: optional rounded side opening for finger access. It becomes part of the cutout, so generated boxes grow automatically to include it.
- `Scoop Side`: `Auto`, `+X`, `-X`, `+Y`, or `-Y`. `Auto` grows the shorter horizontal axis and uses a stable negative side.
- `Scoop Radius` and `Scoop Depth`: control the finger opening width and how far it extends outside the cavity.
- `Entry Safety`: tiny XY inset used by `XYZ Entry` so the lower Z-extrusion remainder is discarded. Default: `0.005 mm`.
- `Entry Cut Extra`: extra XY expansion for the final `XYZ Entry` top cutter to remove small boolean leftovers near the opening. Default: `0.1 mm`.
- `Sweep Pitch`: voxel resolution for the vertical sweep. Default: `0.1 mm`.
- `Sweep Safety`: extra XY dilation applied during sweep rasterization so the voxel mesh does not block insertion. Default: one sweep pitch.
- `Sweep Slices`: number of horizontal source sections. Default: `64`.
- `Entry Top Extra`: extra height above the cutout. Default: `3 mm`.
- `Simplify Ratio`: final decimation ratio. Default: `0.1`, similar to Blender Decimate Ratio `0.1`.
- `Projection Simplify`: small cleanup tolerance for 2D silhouettes.
- `Buffer Join`: round, mitre, or bevel offset corners.

Smaller `Sweep Pitch` values improve detail but increase processing time and mesh size.

## CLI Fallback

The add-on calls these scripts internally, but they can also be used directly.

Generate only a cavity:

```bash
source .venv/bin/activate
python meshcutout.py figure.stl --orientation none --clearance 0.2
```

Generate a box with the cutout already subtracted:

```bash
source .venv/bin/activate
python boxcutout.py figure.stl --orientation none --clearance 0.2
```

Use `--rotate-x`, `--rotate-y`, and `--rotate-z` when you want command-line orientation control instead of arranging the model in Blender.

## Notes

- STL units are treated as millimeters.
- The add-on exports each selected Blender mesh object to its own temporary STL. For `Cavity Only`, it imports one generated cavity per object. For `Box With Cutout`, it imports one shared box after subtracting every per-object cavity.
- Object transforms and modifiers are applied during export.
- If Blender was launched from the desktop and cannot find `python3`, set `Python Executable` explicitly to the `.venv` Python path.
- For noisy or damaged STLs, repair the source mesh before generating the cutout.

## Debugging

The add-on writes the latest external command output to:

```text
meshcutout_last_run.log
```

The file is created in the configured `Scripts Directory`. If a run fails, the temporary folder is preserved and the log includes its path.

For more geometry details, enable `Keep Temporary Files` before running again. The add-on will keep:

```text
meshcutout_blender_temp/
```

That folder contains one subfolder per selected object. Each object folder contains `selected_input.stl`, the generated output if available, and a `debug/` folder with intermediate STL files.

On macOS, Blender does not normally show a separate system console. To see live stdout/stderr, launch Blender from Terminal:

```bash
/Applications/Blender.app/Contents/MacOS/Blender
```
