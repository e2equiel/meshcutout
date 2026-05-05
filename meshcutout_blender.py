bl_info = {
    "name": "Mesh Cutout Generator",
    "author": "Mesh Cutout",
    "version": (0, 1, 10),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Mesh Cutout",
    "description": "Generate storage cavities or fitted boxes from selected mesh objects.",
    "category": "Object",
}

import subprocess
import shutil
import tempfile
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


def _addon_name() -> str:
    return __package__ or __name__


def _path_from_blender(value: str) -> Path | None:
    if not value:
        return None
    return Path(bpy.path.abspath(value)).expanduser()


def _find_scripts_directory(preferences) -> Path | None:
    configured = _path_from_blender(preferences.scripts_directory)
    if configured:
        return configured

    addon_dir = Path(__file__).resolve().parent
    candidates = [
        addon_dir,
        addon_dir.parent,
        Path.cwd(),
    ]
    for candidate in candidates:
        if (
            (candidate / "meshcutout.py").is_file()
            and (candidate / "boxcutout.py").is_file()
            and (candidate / "boxsetcutout.py").is_file()
        ):
            return candidate
    return None


def _default_venv_python(scripts_dir: Path) -> Path | None:
    candidates = [
        scripts_dir / ".venv" / "bin" / "python",
        scripts_dir / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _resolve_python(preferences, scripts_dir: Path) -> str:
    configured = _path_from_blender(preferences.python_executable)
    if configured:
        return str(configured)

    venv_python = _default_venv_python(scripts_dir)
    if venv_python:
        return str(venv_python)

    return "python3"


def _get_preferences(context):
    addon = context.preferences.addons.get(_addon_name())
    if addon is not None:
        return addon.preferences
    for item in context.preferences.addons.values():
        if isinstance(item.preferences, MeshCutoutAddonPreferences):
            return item.preferences
    return None


def _selected_mesh_objects(context) -> list[bpy.types.Object]:
    return [obj for obj in context.selected_objects if obj.type == "MESH"]


def _world_bounds_for_objects(
    context,
    objects: list[bpy.types.Object],
) -> tuple[Vector, Vector]:
    depsgraph = context.evaluated_depsgraph_get()
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False

    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            world = evaluated.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world.x)
            min_corner.y = min(min_corner.y, world.y)
            min_corner.z = min(min_corner.z, world.z)
            max_corner.x = max(max_corner.x, world.x)
            max_corner.y = max(max_corner.y, world.y)
            max_corner.z = max(max_corner.z, world.z)
            found = True

    if not found:
        raise RuntimeError("Selected mesh objects do not have usable bounds.")
    return min_corner, max_corner


def _sample_world_vertices(
    context,
    obj: bpy.types.Object,
    max_points: int = 25000,
) -> list[Vector]:
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        if len(mesh.vertices) == 0:
            return [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]

        step = max(1, math.ceil(len(mesh.vertices) / max_points))
        vertices = [
            evaluated.matrix_world @ mesh.vertices[index].co
            for index in range(0, len(mesh.vertices), step)
        ]
        return vertices[:max_points]
    finally:
        evaluated.to_mesh_clear()


def _height_after_world_y_rotation(xz_points: list[tuple[float, float]], radians: float) -> float:
    sin_angle = math.sin(radians)
    cos_angle = math.cos(radians)
    min_z = float("inf")
    max_z = float("-inf")

    for x_value, z_value in xz_points:
        rotated_z = -sin_angle * x_value + cos_angle * z_value
        min_z = min(min_z, rotated_z)
        max_z = max(max_z, rotated_z)

    return max_z - min_z


def _best_y_rotation_for_min_z(
    vertices: list[Vector],
    pivot: Vector,
) -> tuple[float, float, float]:
    if not vertices:
        raise RuntimeError("Object has no vertices to rotate.")

    xz_points = [(vertex.x - pivot.x, vertex.z - pivot.z) for vertex in vertices]
    initial_height = _height_after_world_y_rotation(xz_points, 0.0)

    best_degrees = 0.0
    best_height = initial_height
    for degrees in range(-90, 91, 5):
        height = _height_after_world_y_rotation(xz_points, math.radians(degrees))
        if height < best_height - 1e-9 or (
            abs(height - best_height) <= 1e-9 and abs(degrees) < abs(best_degrees)
        ):
            best_degrees = float(degrees)
            best_height = height

    fine_start = max(-90.0, best_degrees - 5.0)
    fine_end = min(90.0, best_degrees + 5.0)
    steps = int(round((fine_end - fine_start) / 0.25))
    for index in range(steps + 1):
        degrees = fine_start + index * 0.25
        height = _height_after_world_y_rotation(xz_points, math.radians(degrees))
        if height < best_height - 1e-9 or (
            abs(height - best_height) <= 1e-9 and abs(degrees) < abs(best_degrees)
        ):
            best_degrees = degrees
            best_height = height

    return math.radians(best_degrees), initial_height, best_height


def _finger_scoop_cli_side(side: str) -> str:
    return {
        "AUTO": "auto",
        "POS_X": "+x",
        "NEG_X": "-x",
        "POS_Y": "+y",
        "NEG_Y": "-y",
    }[side]


def _finger_scoop_auto_axis(extents: Vector) -> str:
    return "Y" if extents.x >= extents.y else "X"


def _apply_finger_scoop_precalc(settings, extents: Vector) -> Vector:
    adjusted = extents.copy()
    if not settings.finger_scoop:
        return adjusted

    side = settings.finger_scoop_side
    axis = _finger_scoop_auto_axis(adjusted) if side == "AUTO" else side[-1]
    diameter = 2.0 * settings.finger_scoop_radius
    depth = settings.finger_scoop_depth

    if axis == "X":
        adjusted.x += depth
        adjusted.y = max(adjusted.y, diameter)
    else:
        adjusted.y += depth
        adjusted.x = max(adjusted.x, diameter)

    return adjusted


def _safe_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value.strip())
    return safe or "object"


def _export_selected_stl(filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(
            filepath=str(filepath),
            export_selected_objects=True,
            apply_modifiers=True,
            ascii_format=False,
        )
        return

    if hasattr(bpy.ops, "export_mesh") and hasattr(bpy.ops.export_mesh, "stl"):
        bpy.ops.export_mesh.stl(
            filepath=str(filepath),
            use_selection=True,
            use_mesh_modifiers=True,
            ascii=False,
        )
        return

    raise RuntimeError("Blender STL exporter is not available. Enable the STL add-on or use Blender 4.x.")


def _export_object_stl(context, obj: bpy.types.Object, filepath: Path) -> None:
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active

    try:
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        context.view_layer.objects.active = obj
        _export_selected_stl(filepath)
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for item in selected:
            if item.name in bpy.data.objects:
                item.select_set(True)
        if active is not None and active.name in bpy.data.objects:
            context.view_layer.objects.active = active


def _import_stl(filepath: Path, object_name: str) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)

    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(filepath))
    elif hasattr(bpy.ops, "import_mesh") and hasattr(bpy.ops.import_mesh, "stl"):
        bpy.ops.import_mesh.stl(filepath=str(filepath))
    else:
        raise RuntimeError("Blender STL importer is not available. Enable the STL add-on or use Blender 4.x.")

    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        imported = list(bpy.context.selected_objects)

    for index, obj in enumerate(imported):
        obj.name = object_name if index == 0 else f"{object_name}.{index:03d}"
        obj.data.name = obj.name
    return imported


def _format_command(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _write_last_run_log(
    log_path: Path,
    selected: list[bpy.types.Object],
    temp_dir: Path,
    entries: list[dict] | None = None,
    command: list[str] | None = None,
    result: subprocess.CompletedProcess | None = None,
    exception: Exception | None = None,
) -> None:
    lines = [
        "Mesh Cutout Last Run",
        "=====================",
        f"Temporary directory: {temp_dir}",
        "Selected objects:",
    ]
    lines.extend(f"- {obj.name} ({obj.type})" for obj in selected)
    if entries:
        for index, entry in enumerate(entries, start=1):
            lines.extend(["", f"Object Run {index}: {entry.get('object_name', '<unknown>')}"])
            lines.append("-" * 48)
            for key in ("input_stl", "output_stl", "debug_dir"):
                if entry.get(key):
                    lines.append(f"{key}: {entry[key]}")
            if entry.get("command"):
                lines.extend(["", "Command:", _format_command(entry["command"])])
            entry_result = entry.get("result")
            if entry_result is not None:
                lines.extend(
                    [
                        "",
                        f"Return code: {entry_result.returncode}",
                        "",
                        "STDOUT:",
                        entry_result.stdout or "",
                        "",
                        "STDERR:",
                        entry_result.stderr or "",
                    ]
                )
            if entry.get("exception"):
                lines.extend(["", "Exception:", repr(entry["exception"])])
    else:
        if command is not None:
            lines.extend(["", "Command:", _format_command(command)])
        if result is not None:
            lines.extend(
                [
                    "",
                    f"Return code: {result.returncode}",
                    "",
                    "STDOUT:",
                    result.stdout or "",
                    "",
                    "STDERR:",
                    result.stderr or "",
                ]
            )
    if exception is not None:
        lines.extend(["", "Exception:", repr(exception)])
    log_path.write_text("\n".join(lines), encoding="utf-8")


def _append_if_number(command: list[str], flag: str, value: float | int, enabled: bool = True) -> None:
    if enabled:
        command.extend([flag, str(value)])


def _build_generation_command(
    settings,
    python_executable: str,
    script_path: Path,
    input_stl: Path | list[Path],
    output_stl: Path,
    debug_dir: Path | None = None,
) -> list[str]:
    input_args = [str(path) for path in input_stl] if isinstance(input_stl, list) else [str(input_stl)]
    command = [
        python_executable,
        str(script_path),
        *input_args,
        "-o",
        str(output_stl),
        "--orientation",
        settings.orientation.lower(),
        "--clearance",
        str(settings.clearance),
        "--offset-mode",
        settings.offset_mode.lower(),
        "--buffer-join",
        settings.buffer_join.lower(),
        "--boolean-engine",
        settings.boolean_engine.lower(),
        "--projection-simplify",
        str(settings.projection_simplify),
        "--entry-top-extra",
        str(settings.entry_top_extra),
        "--cavity-method",
        settings.cavity_method.lower().replace("_", "-"),
        "--entry-clearance-extra",
        str(settings.entry_clearance_extra),
        "--entry-cut-extra",
        str(settings.entry_cut_extra),
        "--sweep-slices",
        str(settings.sweep_slices),
        "--sweep-pitch",
        str(settings.sweep_pitch),
        "--simplify-ratio",
        str(settings.simplify_ratio),
        "--rotate-x",
        str(settings.rotate_x),
        "--rotate-y",
        str(settings.rotate_y),
        "--rotate-z",
        str(settings.rotate_z),
        "--secondary-min-area-ratio",
        str(settings.secondary_min_area_ratio),
        "--stl-format",
        "binary",
    ]

    if settings.keep_world_position:
        command.append("--no-rezero")
    if settings.keep_projection_holes:
        command.append("--keep-projection-holes")
    if not settings.precise_projection:
        command.append("--fast-projection")
    if settings.surface_cell_size > 0:
        _append_if_number(command, "--surface-cell-size", settings.surface_cell_size)
    if settings.extrude_extra >= 0:
        _append_if_number(command, "--extrude-extra", settings.extrude_extra)
    if settings.simplify_faces > 0:
        _append_if_number(command, "--simplify-faces", settings.simplify_faces)
    if settings.sweep_overcut >= 0:
        _append_if_number(command, "--sweep-overcut", settings.sweep_overcut)
    if settings.finger_scoop:
        command.append("--finger-scoop")
        command.extend(["--finger-scoop-side", _finger_scoop_cli_side(settings.finger_scoop_side)])
        command.extend(["--finger-scoop-radius", str(settings.finger_scoop_radius)])
        command.extend(["--finger-scoop-depth", str(settings.finger_scoop_depth)])
        command.extend(["--finger-scoop-z-depth", str(settings.finger_scoop_z_depth)])
        command.extend(["--finger-scoop-overlap", str(settings.finger_scoop_overlap)])
    if debug_dir is not None:
        command.extend(["--debug-dir", str(debug_dir)])

    if settings.output_mode == "BOX":
        command.extend(["--margin", str(settings.box_margin)])
        command.extend(["--top-overlap", str(settings.top_overlap)])
        if settings.use_custom_box_size:
            command.extend(
                [
                    "--box-size",
                    str(settings.box_x),
                    str(settings.box_y),
                    str(settings.box_z),
                ]
            )
        if settings.strict_dimensions:
            command.append("--strict-dimensions")

    return command


class MeshCutoutAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = _addon_name()

    scripts_directory: StringProperty(
        name="Scripts Directory",
        subtype="DIR_PATH",
        description="Directory containing meshcutout.py, boxcutout.py and requirements.txt",
        default="",
    )
    python_executable: StringProperty(
        name="Python Executable",
        subtype="FILE_PATH",
        description="Python with the project requirements installed. Empty uses .venv/bin/python when available.",
        default="",
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "scripts_directory")
        layout.prop(self, "python_executable")


class MeshCutoutSettings(bpy.types.PropertyGroup):
    show_advanced: BoolProperty(
        name="Advanced",
        default=False,
    )
    output_mode: EnumProperty(
        name="Output",
        items=[
            ("BOX", "Box With Cutout", "Generate the fitted box with the cavity already subtracted"),
            ("CAVITY", "Cavity Only", "Generate only the cavity/cutter mesh"),
        ],
        default="BOX",
    )
    result_name: StringProperty(
        name="Result Name",
        default="Mesh Cutout Result",
    )
    hide_source_objects: BoolProperty(
        name="Hide Source Objects",
        default=False,
    )
    keep_world_position: BoolProperty(
        name="Keep World Position",
        description="Preserve the selected objects' world-space placement in the generated mesh",
        default=True,
    )
    keep_temp_files: BoolProperty(
        name="Keep Temporary Files",
        default=False,
    )

    box_margin: FloatProperty(
        name="Box Margin",
        description="Default wall margin in millimeters",
        default=2.0,
        min=0.0,
        precision=3,
    )
    use_custom_box_size: BoolProperty(
        name="Custom Box Size",
        default=False,
    )
    box_x: FloatProperty(name="Box X", default=50.0, min=0.001, precision=3)
    box_y: FloatProperty(name="Box Y", default=50.0, min=0.001, precision=3)
    box_z: FloatProperty(name="Box Z", default=35.0, min=0.001, precision=3)
    strict_dimensions: BoolProperty(
        name="Strict Dimensions",
        description="Fail instead of clamping custom box dimensions to the required minimum",
        default=False,
    )
    top_overlap: FloatProperty(
        name="Top Overlap",
        description="Moves the cutter through the top face so the box opening is robust",
        default=0.5,
        min=0.0,
        precision=3,
    )

    clearance: FloatProperty(
        name="Clearance",
        description="Cavity clearance in millimeters",
        default=0.2,
        min=0.0,
        precision=3,
    )
    offset_mode: EnumProperty(
        name="Offset Mode",
        items=[
            ("SILHOUETTE", "Silhouette", "2D projection buffer, recommended"),
            ("SURFACE", "Surface", "3D PyMeshLab surface offset"),
            ("NONE", "None", "No offset"),
        ],
        default="SILHOUETTE",
    )
    surface_cell_size: FloatProperty(
        name="Surface Cell Size",
        description="PyMeshLab surface offset cell size. Use 0 for auto.",
        default=0.0,
        min=0.0,
        precision=3,
    )
    extrude_extra: FloatProperty(
        name="Extrude Extra",
        description="Total extra extrusion per axis. Use -1 for auto.",
        default=-1.0,
        min=-1.0,
        precision=3,
    )
    orientation: EnumProperty(
        name="Orientation",
        items=[
            ("NONE", "Use Selection", "Use the current Blender orientation"),
            ("AUTO", "Auto", "Try to orient the model automatically"),
        ],
        default="NONE",
    )
    rotate_x: FloatProperty(name="Rotate X", default=0.0, precision=3)
    rotate_y: FloatProperty(name="Rotate Y", default=0.0, precision=3)
    rotate_z: FloatProperty(name="Rotate Z", default=0.0, precision=3)
    secondary_min_area_ratio: FloatProperty(
        name="Secondary Area Ratio",
        default=0.03,
        min=0.0,
        max=1.0,
        precision=4,
    )
    buffer_join: EnumProperty(
        name="Buffer Join",
        items=[
            ("ROUND", "Round", "Rounded buffered corners"),
            ("MITRE", "Mitre", "Sharp buffered corners"),
            ("BEVEL", "Bevel", "Beveled buffered corners"),
        ],
        default="ROUND",
    )
    boolean_engine: EnumProperty(
        name="Boolean Engine",
        items=[
            ("MANIFOLD", "Manifold", "Use manifold3d through Trimesh"),
            ("AUTO", "Auto", "Let Trimesh pick an engine"),
            ("BLENDER", "Blender", "Ask Trimesh to use Blender if available externally"),
        ],
        default="MANIFOLD",
    )
    precise_projection: BoolProperty(
        name="Precise Projection",
        default=True,
    )
    keep_projection_holes: BoolProperty(
        name="Keep Projection Holes",
        default=False,
    )
    projection_simplify: FloatProperty(
        name="Projection Simplify",
        default=0.0001,
        min=0.0,
        precision=6,
    )

    entry_top_extra: FloatProperty(
        name="Entry Top Extra",
        default=3.0,
        min=0.0,
        precision=3,
    )
    cavity_method: EnumProperty(
        name="Cavity Method",
        items=[
            (
                "XYZ_ENTRY",
                "XYZ Entry",
                "Projection intersection plus a top Z-entry piece; no sweep voxels",
            ),
            (
                "SWEEP_Z",
                "Sweep Z",
                "Voxelized vertical sweep from horizontal sections",
            ),
        ],
        default="XYZ_ENTRY",
    )
    finger_scoop: BoolProperty(
        name="Finger Scoop",
        description="Add a rounded side opening for finger access",
        default=False,
    )
    finger_scoop_side: EnumProperty(
        name="Scoop Side",
        items=[
            ("AUTO", "Auto", "Grow the shorter horizontal axis; uses a stable negative side"),
            ("POS_X", "+X", "Place the scoop on the positive X side"),
            ("NEG_X", "-X", "Place the scoop on the negative X side"),
            ("POS_Y", "+Y", "Place the scoop on the positive Y side"),
            ("NEG_Y", "-Y", "Place the scoop on the negative Y side"),
        ],
        default="AUTO",
    )
    finger_scoop_radius: FloatProperty(
        name="Scoop Radius",
        description="Finger scoop radius in millimeters",
        default=8.0,
        min=0.001,
        precision=3,
    )
    finger_scoop_depth: FloatProperty(
        name="Scoop Depth",
        description="How far the scoop extends outside the cavity",
        default=12.0,
        min=0.001,
        precision=3,
    )
    finger_scoop_z_depth: FloatProperty(
        name="Scoop Z Depth",
        description="How far down from the entry top the scoop cuts; 0 means full depth",
        default=10.0,
        min=0.0,
        precision=3,
    )
    finger_scoop_overlap: FloatProperty(
        name="Scoop Overlap",
        description="How far the scoop centerline overlaps the cavity side",
        default=1.0,
        min=0.0,
        precision=3,
    )
    entry_clearance_extra: FloatProperty(
        name="Entry Safety",
        description="Tiny XY inset for XYZ Entry so the lower Z extrusion remainder is discarded",
        default=0.005,
        min=0.0,
        precision=4,
    )
    entry_cut_extra: FloatProperty(
        name="Entry Cut Extra",
        description="Extra XY expansion for the final XYZ Entry top cutter",
        default=0.1,
        min=0.0,
        precision=4,
    )
    sweep_slices: IntProperty(
        name="Sweep Slices",
        default=64,
        min=4,
    )
    sweep_pitch: FloatProperty(
        name="Sweep Pitch",
        default=0.1,
        min=0.001,
        precision=4,
    )
    sweep_overcut: FloatProperty(
        name="Sweep Safety",
        description="Extra XY dilation for the sweep in millimeters. Use -1 for one sweep pitch.",
        default=-1.0,
        min=-1.0,
        precision=4,
    )
    simplify_ratio: FloatProperty(
        name="Simplify Ratio",
        default=0.1,
        min=0.0,
        max=1.0,
        precision=3,
    )
    simplify_faces: IntProperty(
        name="Simplify Faces",
        description="Optional exact face target. 0 means use Simplify Ratio.",
        default=0,
        min=0,
    )


class MESH_CUTOUT_OT_check_setup(bpy.types.Operator):
    bl_idname = "mesh_cutout.check_setup"
    bl_label = "Check Setup"
    bl_description = "Verify the configured scripts directory and Python dependencies"

    def execute(self, context):
        preferences = _get_preferences(context)
        if preferences is None:
            self.report({"ERROR"}, "Mesh Cutout preferences are not available.")
            return {"CANCELLED"}

        scripts_dir = _find_scripts_directory(preferences)
        if scripts_dir is None:
            self.report({"ERROR"}, "Set Scripts Directory to the folder containing meshcutout.py and boxsetcutout.py.")
            return {"CANCELLED"}

        python_executable = _resolve_python(preferences, scripts_dir)
        command = [
            python_executable,
            "-c",
            (
                "import numpy, scipy, shapely, trimesh, manifold3d, skimage; "
                "print('Core dependencies OK')"
            ),
        ]
        result = _run_command(command, scripts_dir)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "Dependency check failed.").strip()
            self.report({"ERROR"}, message[-900:])
            print(message)
            return {"CANCELLED"}

        self.report({"INFO"}, f"Mesh Cutout setup OK: {python_executable}")
        print(result.stdout.strip())
        return {"FINISHED"}


class MESH_CUTOUT_OT_align_z_tops(bpy.types.Operator):
    bl_idname = "mesh_cutout.align_z_tops"
    bl_label = "Align Z Tops"
    bl_description = "Move selected mesh objects in world Z so their top bounds share the highest selected top"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_mesh_objects(context)
        if not selected:
            self.report({"ERROR"}, "Select one or more mesh objects first.")
            return {"CANCELLED"}

        bounds = [(obj, _world_bounds_for_objects(context, [obj])) for obj in selected]
        target_top = max(max_corner.z for _, (_, max_corner) in bounds)
        moved = 0
        for obj, (_, max_corner) in bounds:
            delta = target_top - max_corner.z
            if abs(delta) < 1e-9:
                continue
            matrix = obj.matrix_world.copy()
            matrix.translation.z += delta
            obj.matrix_world = matrix
            moved += 1

        context.view_layer.update()
        self.report({"INFO"}, f"Aligned {moved} object(s) to Z top {target_top:.3f}.")
        return {"FINISHED"}


class MESH_CUTOUT_OT_precalculate_box_size(bpy.types.Operator):
    bl_idname = "mesh_cutout.precalculate_box_size"
    bl_label = "Precalc Box Size"
    bl_description = "Estimate box dimensions from selected object bounds and the current box margin"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_mesh_objects(context)
        if not selected:
            self.report({"ERROR"}, "Select one or more mesh objects first.")
            return {"CANCELLED"}

        settings = context.scene.mesh_cutout_settings
        min_corner, max_corner = _world_bounds_for_objects(context, selected)
        extents = _apply_finger_scoop_precalc(settings, max_corner - min_corner)
        margin = settings.box_margin

        settings.box_x = max(0.001, extents.x + 2.0 * margin)
        settings.box_y = max(0.001, extents.y + 2.0 * margin)
        settings.box_z = max(0.001, extents.z + margin)
        settings.use_custom_box_size = True

        self.report(
            {"INFO"},
            (
                "Precalculated box: "
                f"X {settings.box_x:.3f}, Y {settings.box_y:.3f}, Z {settings.box_z:.3f}."
            ),
        )
        return {"FINISHED"}


class MESH_CUTOUT_OT_minimize_z_by_y(bpy.types.Operator):
    bl_idname = "mesh_cutout.minimize_z_by_y"
    bl_label = "Min Z by Y"
    bl_description = "Rotate each selected mesh around world Y to minimize its Z height"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        selected = _selected_mesh_objects(context)
        if not selected:
            self.report({"ERROR"}, "Select one or more mesh objects first.")
            return {"CANCELLED"}

        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode="OBJECT")

        changed = 0
        summaries = []
        for obj in selected:
            min_corner, max_corner = _world_bounds_for_objects(context, [obj])
            pivot = (min_corner + max_corner) / 2.0
            vertices = _sample_world_vertices(context, obj)
            radians, initial_height, final_height = _best_y_rotation_for_min_z(vertices, pivot)
            degrees = math.degrees(radians)

            if abs(degrees) < 1e-6:
                summaries.append(f"{obj.name}: 0.00 deg")
                continue

            rotation = (
                Matrix.Translation(pivot)
                @ Matrix.Rotation(radians, 4, "Y")
                @ Matrix.Translation(-pivot)
            )
            obj.matrix_world = rotation @ obj.matrix_world
            changed += 1
            summaries.append(
                f"{obj.name}: {degrees:.2f} deg, {initial_height:.3f}->{final_height:.3f}"
            )

        context.view_layer.update()
        print("Min Z by Y:")
        for summary in summaries:
            print(f"- {summary}")
        self.report({"INFO"}, f"Rotated {changed} object(s) to minimize Z height.")
        return {"FINISHED"}


class MESH_CUTOUT_OT_generate(bpy.types.Operator):
    bl_idname = "mesh_cutout.generate"
    bl_label = "Generate Mesh Cutout"
    bl_description = "Generate a cavity or fitted box from the selected mesh objects"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.mesh_cutout_settings
        preferences = _get_preferences(context)
        if preferences is None:
            self.report({"ERROR"}, "Mesh Cutout preferences are not available.")
            return {"CANCELLED"}

        selected = _selected_mesh_objects(context)
        if not selected:
            self.report({"ERROR"}, "Select one or more mesh objects first.")
            return {"CANCELLED"}

        scripts_dir = _find_scripts_directory(preferences)
        if scripts_dir is None:
            self.report({"ERROR"}, "Set Scripts Directory to the folder containing meshcutout.py and boxcutout.py.")
            return {"CANCELLED"}

        script_name = "boxsetcutout.py" if settings.output_mode == "BOX" else "meshcutout.py"
        script_path = scripts_dir / script_name
        if not script_path.is_file():
            self.report({"ERROR"}, f"Missing script: {script_path}")
            return {"CANCELLED"}

        python_executable = _resolve_python(preferences, scripts_dir)
        log_path = scripts_dir / "meshcutout_last_run.log"
        cleanup_temp = not settings.keep_temp_files
        if settings.keep_temp_files:
            temp_dir = scripts_dir / "meshcutout_blender_temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="meshcutout_blender_"))

        log_entries: list[dict] = []

        try:
            if settings.output_mode == "BOX":
                input_stls = []
                for index, obj in enumerate(selected, start=1):
                    object_dir = temp_dir / f"{index:03d}_{_safe_filename(obj.name)}"
                    object_dir.mkdir(parents=True, exist_ok=True)
                    input_stl = object_dir / "selected_input.stl"
                    _export_object_stl(context, obj, input_stl)
                    input_stls.append(input_stl)

                output_stl = temp_dir / "generated_box.stl"
                debug_dir = temp_dir / "debug" if settings.keep_temp_files else None
                command = _build_generation_command(
                    settings,
                    python_executable,
                    script_path,
                    input_stls,
                    output_stl,
                    debug_dir=debug_dir,
                )
                print("Running Mesh Cutout box-set command:")
                print(_format_command(command))
                result = _run_command(command, scripts_dir)
                log_entries.append(
                    {
                        "object_name": "Box With Cutout",
                        "input_stl": ", ".join(str(path) for path in input_stls),
                        "output_stl": str(output_stl),
                        "debug_dir": str(debug_dir) if debug_dir is not None else None,
                        "command": command,
                        "result": result,
                    }
                )
                _write_last_run_log(log_path, selected, temp_dir, entries=log_entries)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)

                if result.returncode != 0 or not output_stl.is_file():
                    message = (result.stderr or result.stdout or "Mesh Cutout generation failed.").strip()
                    cleanup_temp = False
                    self.report({"ERROR"}, f"{message[-740:]} Log: {log_path}")
                    return {"CANCELLED"}

                name = settings.result_name.strip() or "Mesh Cutout Box"
                _import_stl(output_stl, name)
            else:
                for index, obj in enumerate(selected, start=1):
                    object_dir = temp_dir / f"{index:03d}_{_safe_filename(obj.name)}"
                    object_dir.mkdir(parents=True, exist_ok=True)
                    input_stl = object_dir / "selected_input.stl"
                    output_stl = object_dir / "generated_output.stl"
                    debug_dir = object_dir / "debug" if settings.keep_temp_files else None

                    _export_object_stl(context, obj, input_stl)
                    command = _build_generation_command(
                        settings,
                        python_executable,
                        script_path,
                        input_stl,
                        output_stl,
                        debug_dir=debug_dir,
                    )
                    print(f"Running Mesh Cutout command for {obj.name}:")
                    print(_format_command(command))
                    result = _run_command(command, scripts_dir)
                    log_entries.append(
                        {
                            "object_name": obj.name,
                            "input_stl": str(input_stl),
                            "output_stl": str(output_stl),
                            "debug_dir": str(debug_dir) if debug_dir is not None else None,
                            "command": command,
                            "result": result,
                        }
                    )
                    _write_last_run_log(log_path, selected, temp_dir, entries=log_entries)
                    if result.stdout:
                        print(result.stdout)
                    if result.stderr:
                        print(result.stderr)

                    if result.returncode != 0 or not output_stl.is_file():
                        message = (result.stderr or result.stdout or "Mesh Cutout generation failed.").strip()
                        cleanup_temp = False
                        self.report({"ERROR"}, f"{obj.name}: {message[-720:]} Log: {log_path}")
                        return {"CANCELLED"}

                    result_prefix = settings.result_name.strip() or "Mesh Cutout Cavity"
                    name = f"{result_prefix} - {obj.name}" if len(selected) > 1 else result_prefix
                    imported = _import_stl(output_stl, name)
                    for imported_obj in imported:
                        imported_obj.display_type = "WIRE"

            if settings.hide_source_objects:
                for obj in selected:
                    obj.hide_set(True)

            if settings.output_mode == "BOX":
                self.report({"INFO"}, f"Generated one box with {len(selected)} cutout(s).")
            else:
                self.report({"INFO"}, f"Generated {len(selected)} cavity result(s), one per object.")
            return {"FINISHED"}
        except Exception as exc:
            cleanup_temp = False
            log_entries.append({"object_name": "<operator>", "exception": exc})
            _write_last_run_log(log_path, selected, temp_dir, entries=log_entries, exception=exc)
            self.report({"ERROR"}, f"{str(exc)[-760:]} Log: {log_path}")
            print(f"Mesh Cutout error: {exc}")
            return {"CANCELLED"}
        finally:
            if cleanup_temp and temp_dir.exists():
                shutil.rmtree(temp_dir)


class VIEW3D_PT_mesh_cutout(bpy.types.Panel):
    bl_label = "Mesh Cutout"
    bl_idname = "VIEW3D_PT_mesh_cutout"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Cutout"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mesh_cutout_settings
        preferences = _get_preferences(context)

        prep = layout.box()
        prep.label(text="Prepare")
        prep.operator("mesh_cutout.minimize_z_by_y", text="Min Z by Y")
        row = prep.row(align=True)
        row.operator("mesh_cutout.align_z_tops", text="Align Z Tops")
        row.operator("mesh_cutout.precalculate_box_size", text="Precalc Box Size")

        layout.prop(settings, "output_mode")
        if settings.output_mode == "BOX":
            layout.label(text="One shared box, one cutout per selected mesh.")
        else:
            layout.label(text="One cavity per selected mesh.")
        layout.prop(settings, "result_name")

        if settings.output_mode == "BOX":
            box = layout.box()
            box.label(text="Box")
            box.prop(settings, "box_margin")
            box.prop(settings, "use_custom_box_size")
            if settings.use_custom_box_size:
                row = box.row(align=True)
                row.prop(settings, "box_x")
                row.prop(settings, "box_y")
                row.prop(settings, "box_z")

        cavity = layout.box()
        cavity.label(text="Cavity")
        cavity.prop(settings, "clearance")
        cavity.prop(settings, "cavity_method")
        cavity.prop(settings, "finger_scoop")
        if settings.finger_scoop:
            cavity.prop(settings, "finger_scoop_side")
            row = cavity.row(align=True)
            row.prop(settings, "finger_scoop_radius")
            row.prop(settings, "finger_scoop_depth")

        row = layout.row()
        icon = "TRIA_DOWN" if settings.show_advanced else "TRIA_RIGHT"
        row.prop(settings, "show_advanced", text="Advanced", icon=icon, emboss=False)
        if settings.show_advanced:
            if preferences is not None:
                runner = layout.box()
                runner.label(text="External Runner")
                runner.prop(preferences, "scripts_directory")
                runner.prop(preferences, "python_executable")
                runner.operator("mesh_cutout.check_setup", icon="CHECKMARK")

            behavior = layout.box()
            behavior.label(text="Behavior")
            behavior.prop(settings, "keep_world_position")
            behavior.prop(settings, "hide_source_objects")

            if settings.output_mode == "BOX":
                box_advanced = layout.box()
                box_advanced.label(text="Box Advanced")
                box_advanced.prop(settings, "top_overlap")
                box_advanced.prop(settings, "strict_dimensions")

            cavity_advanced = layout.box()
            cavity_advanced.label(text="Cavity Advanced")
            cavity_advanced.prop(settings, "offset_mode")
            if settings.offset_mode == "SURFACE":
                cavity_advanced.prop(settings, "surface_cell_size")
            cavity_advanced.prop(settings, "extrude_extra")
            cavity_advanced.prop(settings, "orientation")
            if settings.orientation == "AUTO":
                cavity_advanced.prop(settings, "secondary_min_area_ratio")
            row = cavity_advanced.row(align=True)
            row.prop(settings, "rotate_x")
            row.prop(settings, "rotate_y")
            row.prop(settings, "rotate_z")

            projection = layout.box()
            projection.label(text="Projection")
            projection.prop(settings, "buffer_join")
            projection.prop(settings, "precise_projection")
            projection.prop(settings, "keep_projection_holes")
            projection.prop(settings, "projection_simplify")
            projection.prop(settings, "boolean_engine")

            entry = layout.box()
            entry.label(text="Entry")
            entry.prop(settings, "entry_top_extra")
            if settings.cavity_method == "XYZ_ENTRY":
                entry.prop(settings, "entry_clearance_extra")
                entry.prop(settings, "entry_cut_extra")
            else:
                entry.prop(settings, "sweep_pitch")
                entry.prop(settings, "sweep_overcut")
                entry.prop(settings, "sweep_slices")
            if settings.finger_scoop:
                entry.prop(settings, "finger_scoop_z_depth")
                entry.prop(settings, "finger_scoop_overlap")

            simplify = layout.box()
            simplify.label(text="Simplify")
            simplify.prop(settings, "simplify_ratio")
            simplify.prop(settings, "simplify_faces")
            simplify.prop(settings, "keep_temp_files")

        layout.operator("mesh_cutout.generate", icon="MOD_BOOLEAN")


classes = (
    MeshCutoutAddonPreferences,
    MeshCutoutSettings,
    MESH_CUTOUT_OT_check_setup,
    MESH_CUTOUT_OT_align_z_tops,
    MESH_CUTOUT_OT_precalculate_box_size,
    MESH_CUTOUT_OT_minimize_z_by_y,
    MESH_CUTOUT_OT_generate,
    VIEW3D_PT_mesh_cutout,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_cutout_settings = PointerProperty(type=MeshCutoutSettings)


def unregister():
    if hasattr(bpy.types.Scene, "mesh_cutout_settings"):
        del bpy.types.Scene.mesh_cutout_settings
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()
