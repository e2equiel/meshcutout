#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

import boxcutout
import meshcutout


AXIS_NAMES = ("X", "Y", "Z")


def requested_box_dimensions(args: argparse.Namespace) -> list[float | None]:
    requested: list[float | None] = [None, None, None]
    if args.box_size is not None:
        requested = list(args.box_size)

    for index, option_name in enumerate(("box_x", "box_y", "box_z")):
        value = getattr(args, option_name)
        if value is not None:
            requested[index] = value

    return requested


def build_cutout_args(args: argparse.Namespace, input_path: Path) -> argparse.Namespace:
    cutout_args = meshcutout.parse_args([str(input_path)])
    fields = (
        "clearance",
        "offset_mode",
        "surface_cell_size",
        "extrude_extra",
        "orientation",
        "rotate_x",
        "rotate_y",
        "rotate_z",
        "secondary_min_area_ratio",
        "buffer_join",
        "boolean_engine",
        "precise_projection",
        "keep_projection_holes",
        "projection_simplify",
        "entry_top_extra",
        "cavity_method",
        "entry_clearance_extra",
        "entry_cut_extra",
        "sweep_slices",
        "sweep_pitch",
        "sweep_overcut",
        "finger_scoop",
        "finger_scoop_side",
        "finger_scoop_radius",
        "finger_scoop_depth",
        "finger_scoop_z_depth",
        "finger_scoop_overlap",
        "simplify_faces",
        "simplify_ratio",
        "stl_format",
        "no_rezero",
    )

    for field in fields:
        value = getattr(args, field)
        if value is not None:
            setattr(cutout_args, field, value)

    if cutout_args.boolean_engine == "auto":
        cutout_args.boolean_engine = None
    if cutout_args.simplify_ratio == 0:
        cutout_args.simplify_ratio = None

    return cutout_args


def validate_args(args: argparse.Namespace) -> None:
    if args.margin < 0:
        raise meshcutout.MeshCutoutError("--margin cannot be negative.")
    if args.top_overlap < 0:
        raise meshcutout.MeshCutoutError("--top-overlap cannot be negative.")

    for index, value in enumerate(requested_box_dimensions(args)):
        if value is not None and value <= 0:
            raise meshcutout.MeshCutoutError(
                f"Box dimension {AXIS_NAMES[index]} must be greater than zero."
            )

    if args.clearance < 0:
        raise meshcutout.MeshCutoutError("--clearance cannot be negative.")
    if args.surface_cell_size is not None and args.surface_cell_size <= 0:
        raise meshcutout.MeshCutoutError("--surface-cell-size must be greater than zero.")
    if args.projection_simplify < 0:
        raise meshcutout.MeshCutoutError("--projection-simplify cannot be negative.")
    if args.entry_top_extra < 0:
        raise meshcutout.MeshCutoutError("--entry-top-extra cannot be negative.")
    if args.entry_clearance_extra < 0:
        raise meshcutout.MeshCutoutError("--entry-clearance-extra cannot be negative.")
    if args.entry_cut_extra < 0:
        raise meshcutout.MeshCutoutError("--entry-cut-extra cannot be negative.")
    if args.sweep_slices < 4:
        raise meshcutout.MeshCutoutError("--sweep-slices must be at least 4.")
    if args.sweep_pitch <= 0:
        raise meshcutout.MeshCutoutError("--sweep-pitch must be greater than zero.")
    if args.sweep_overcut is not None and args.sweep_overcut < 0:
        raise meshcutout.MeshCutoutError("--sweep-overcut cannot be negative.")
    if args.finger_scoop_radius <= 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-radius must be greater than zero.")
    if args.finger_scoop_depth <= 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-depth must be greater than zero.")
    if args.finger_scoop_z_depth < 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-z-depth cannot be negative.")
    if args.finger_scoop_overlap < 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-overlap cannot be negative.")
    if args.simplify_faces is not None and args.simplify_faces <= 0:
        raise meshcutout.MeshCutoutError("--simplify-faces must be greater than zero.")
    if args.simplify_ratio is not None and not 0.0 <= args.simplify_ratio <= 1.0:
        raise meshcutout.MeshCutoutError("--simplify-ratio must be between 0 and 1.")


def build_sliding_lid_system(box_min, box_extents, margin_x, margin_y, orig_box_top, args):
    t_lid = args.lid_thickness
    c = args.lid_clearance
    slot_depth = args.lid_slot_depth
    t_tab = args.lid_thickness
    t_lip = args.lid_top_lip

    Z0 = orig_box_top
    Z1 = Z0 + c
    Z2 = Z1 + t_tab
    Z3 = Z2 + c
    Z4 = Z3 + t_lip
    Z5 = Z4 + c
    Z6 = Z5 + t_lid

    inner_x = box_extents[0] - 2 * margin_x
    inner_y = box_extents[1] - 2 * margin_y
    slide_axis = "Y" if inner_y >= inner_x else "X"

    cutter_parts = []
    lid_parts = []

    X0 = box_min[0]
    X1 = box_min[0] + margin_x
    X2 = box_min[0] + box_extents[0] - margin_x
    X3 = box_min[0] + box_extents[0]
    
    Y0 = box_min[1]
    Y1 = box_min[1] + margin_y
    Y2 = box_min[1] + box_extents[1] - margin_y
    Y3 = box_min[1] + box_extents[1]

    if slide_axis == "Y":
        # 1. Center void
        center = trimesh.creation.box(extents=[X2 - X1, Y2 - Y1, Z4 - Z0 + 10.0])
        center.apply_translation([(X1 + X2)/2, (Y1 + Y2)/2, Z0 + (Z4 - Z0 + 10.0)/2])
        cutter_parts.append(center)

        # 2. Left Groove
        l_groove = trimesh.creation.box(extents=[slot_depth, Y2 - Y0 + 2.0, Z3 - Z1])
        l_groove.apply_translation([X1 - slot_depth/2, (Y0 + Y2)/2, (Z1 + Z3)/2])
        cutter_parts.append(l_groove)

        # 3. Right Groove
        r_groove = trimesh.creation.box(extents=[slot_depth, Y2 - Y0 + 2.0, Z3 - Z1])
        r_groove.apply_translation([X2 + slot_depth/2, (Y0 + Y2)/2, (Z1 + Z3)/2])
        cutter_parts.append(r_groove)

        # 4. Entrance
        entrance = trimesh.creation.box(extents=[X3 - X0 + 2.0, Y1 - Y0 + 1.0, Z4 - Z1 + 10.0])
        entrance.apply_translation([(X0 + X3)/2, Y0 + (Y1 - Y0)/2 - 0.5, Z1 + (Z4 - Z1 + 10.0)/2])
        cutter_parts.append(entrance)

        # LID
        top_plate = trimesh.creation.box(extents=[X3 - X0, Y3 - Y0, Z6 - Z5])
        top_plate.apply_translation([(X0 + X3)/2, (Y0 + Y3)/2, (Z5 + Z6)/2])
        lid_parts.append(top_plate)

        neck = trimesh.creation.box(extents=[X2 - X1 - 2*c, Y2 - Y0 - c, Z5 - Z3])
        neck.apply_translation([(X1 + X2)/2, Y0 + (Y2 - Y0 - c)/2, (Z3 + Z5)/2])
        lid_parts.append(neck)

        tabs = trimesh.creation.box(extents=[X2 - X1 + 2*slot_depth - 2*c, Y2 - Y0 - c, Z2 - Z1 - c])
        tabs.apply_translation([(X1 + X2)/2, Y0 + (Y2 - Y0 - c)/2, Z1 + c + (Z2 - Z1 - c)/2])
        lid_parts.append(tabs)

        stopper = trimesh.creation.box(extents=[X3 - X0, Y1 - Y0 - c, Z5 - Z1 - c])
        stopper.apply_translation([(X0 + X3)/2, Y0 + (Y1 - Y0 - c)/2, Z1 + c + (Z5 - Z1 - c)/2])
        lid_parts.append(stopper)

    else:
        # 1. Center void
        center = trimesh.creation.box(extents=[X2 - X1, Y2 - Y1, Z4 - Z0 + 10.0])
        center.apply_translation([(X1 + X2)/2, (Y1 + Y2)/2, Z0 + (Z4 - Z0 + 10.0)/2])
        cutter_parts.append(center)

        # 2. Bottom Groove (Y1)
        b_groove = trimesh.creation.box(extents=[X2 - X0 + 2.0, slot_depth, Z3 - Z1])
        b_groove.apply_translation([(X0 + X2)/2, Y1 - slot_depth/2, (Z1 + Z3)/2])
        cutter_parts.append(b_groove)

        # 3. Top Groove (Y2)
        t_groove = trimesh.creation.box(extents=[X2 - X0 + 2.0, slot_depth, Z3 - Z1])
        t_groove.apply_translation([(X0 + X2)/2, Y2 + slot_depth/2, (Z1 + Z3)/2])
        cutter_parts.append(t_groove)

        # 4. Entrance (-X wall)
        entrance = trimesh.creation.box(extents=[X1 - X0 + 1.0, Y3 - Y0 + 2.0, Z4 - Z1 + 10.0])
        entrance.apply_translation([X0 + (X1 - X0)/2 - 0.5, (Y0 + Y3)/2, Z1 + (Z4 - Z1 + 10.0)/2])
        cutter_parts.append(entrance)

        # LID
        top_plate = trimesh.creation.box(extents=[X3 - X0, Y3 - Y0, Z6 - Z5])
        top_plate.apply_translation([(X0 + X3)/2, (Y0 + Y3)/2, (Z5 + Z6)/2])
        lid_parts.append(top_plate)

        neck = trimesh.creation.box(extents=[X2 - X0 - c, Y2 - Y1 - 2*c, Z5 - Z3])
        neck.apply_translation([X0 + (X2 - X0 - c)/2, (Y1 + Y2)/2, (Z3 + Z5)/2])
        lid_parts.append(neck)

        tabs = trimesh.creation.box(extents=[X2 - X0 - c, Y2 - Y1 + 2*slot_depth - 2*c, Z2 - Z1 - c])
        tabs.apply_translation([X0 + (X2 - X0 - c)/2, (Y1 + Y2)/2, Z1 + c + (Z2 - Z1 - c)/2])
        lid_parts.append(tabs)

        stopper = trimesh.creation.box(extents=[X1 - X0 - c, Y3 - Y0, Z5 - Z1 - c])
        stopper.apply_translation([X0 + (X1 - X0 - c)/2, (Y0 + Y3)/2, Z1 + c + (Z5 - Z1 - c)/2])
        lid_parts.append(stopper)

    cutter = meshcutout.boolean_union_or_concatenate(cutter_parts, engine=args.boolean_engine, label="Lid cutter")
    lid = meshcutout.boolean_union_or_concatenate(lid_parts, engine=args.boolean_engine, label="Lid mesh")
    return cutter, lid, (Z4 - Z0)


def resolve_box_bounds(
    cavities: list[trimesh.Trimesh],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = np.array([cavity.bounds[0] for cavity in cavities])
    maxs = np.array([cavity.bounds[1] for cavity in cavities])
    combined_min = mins.min(axis=0)
    combined_max = maxs.max(axis=0)
    combined_center = (combined_min + combined_max) / 2.0

    minimum_extents = np.array(
        [
            (combined_max[0] - combined_min[0]) + 2.0 * args.margin,
            (combined_max[1] - combined_min[1]) + 2.0 * args.margin,
            (combined_max[2] - combined_min[2]) + args.margin,
        ],
        dtype=float,
    )

    requested = requested_box_dimensions(args)
    box_extents = np.array(
        [
            minimum_extents[index] if requested[index] is None else requested[index]
            for index in range(3)
        ],
        dtype=float,
    )

    too_small = box_extents < minimum_extents
    if np.any(too_small):
        details = []
        for index, is_small in enumerate(too_small):
            if is_small:
                details.append(
                    f"{AXIS_NAMES[index]} {box_extents[index]:.3f} -> {minimum_extents[index]:.3f}"
                )
        message = "Dimensions clamped to minimum: " + ", ".join(details) + "."
        if args.strict_dimensions:
            raise meshcutout.MeshCutoutError(message)
        print(message)
        box_extents = np.maximum(box_extents, minimum_extents)

    box_top = float(combined_max[2] - args.top_overlap)
    box_min = np.array(
        [
            combined_center[0] - box_extents[0] / 2.0,
            combined_center[1] - box_extents[1] / 2.0,
            box_top - box_extents[2],
        ],
        dtype=float,
    )
    return box_min, box_extents, minimum_extents


def extend_cavity_entry_to_top(
    cavity: trimesh.Trimesh,
    target_top: float,
    args: argparse.Namespace,
    label: str,
) -> trimesh.Trimesh:
    cutter = cavity.copy()
    current_top = float(cavity.bounds[1, 2])
    if current_top >= target_top - 1e-6:
        return cutter

    stretch_depth = max(float(args.sweep_pitch) * 2.0, 0.2)
    max_depth = max(float(cavity.extents[2]) * 0.25, 1e-5)
    stretch_depth = min(stretch_depth, max_depth)
    base_z = max(float(cavity.bounds[0, 2]) + 1e-5, current_top - stretch_depth)
    source_span = current_top - base_z
    if source_span <= 1e-8:
        raise meshcutout.MeshCutoutError(f"Could not find an entry band for {label}.")

    top_vertices = cutter.vertices[:, 2] >= base_z - 1e-8
    if not np.any(top_vertices):
        raise meshcutout.MeshCutoutError(f"Could not find top entry vertices for {label}.")

    scale = (target_top - base_z) / source_span
    cutter.vertices[top_vertices, 2] = base_z + (cutter.vertices[top_vertices, 2] - base_z) * scale

    if not cutter.is_volume:
        cutter = meshcutout.cleanup_mesh(cutter)
    if not cutter.is_volume:
        cutter = meshcutout.repair_volume_mesh(cutter, f"entry-extended {label}")
    if not cutter.is_volume:
        raise meshcutout.MeshCutoutError(f"Entry-extended {label} is not a closed volume.")

    print(
        f"Extended {label} entry: {current_top:.3f} -> {target_top:.3f} mm "
        f"from band z>={base_z:.3f}."
    )
    return cutter


def export_debug_meshes(
    debug_dir: Path | None,
    meshes: dict[str, trimesh.Trimesh],
    ascii_stl: bool,
) -> None:
    if debug_dir is None:
        return
    meshcutout.export_debug(debug_dir, meshes, ascii_stl=ascii_stl)


def mesh_from_boolean_result(result, label: str) -> trimesh.Trimesh:
    if result is None or result.is_empty:
        raise meshcutout.MeshCutoutError(f"{label} returned an empty mesh.")
    if isinstance(result, trimesh.Scene):
        meshes = [g for g in result.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise meshcutout.MeshCutoutError(f"{label} did not return meshes.")
        result = trimesh.util.concatenate(meshes)
    return result


def cutter_components(cutter: trimesh.Trimesh, label: str) -> list[trimesh.Trimesh]:
    components = [
        component
        for component in cutter.split(only_watertight=False)
        if len(component.faces) > 4
    ]
    if not components:
        components = [cutter]

    solids: list[trimesh.Trimesh] = []
    for index, component in enumerate(components, start=1):
        solid = component
        if not solid.is_volume:
            solid = meshcutout.cleanup_mesh(solid)
        if not solid.is_volume:
            solid = meshcutout.repair_volume_mesh(solid, f"{label} component {index}")
        if not solid.is_volume:
            raise meshcutout.MeshCutoutError(
                f"{label} component {index} is not a closed volume."
            )
        solids.append(solid)
    return solids


def subtract_cutters_sequentially(
    box: trimesh.Trimesh,
    cutters: list[trimesh.Trimesh],
    engine: str | None,
) -> trimesh.Trimesh:
    current = box
    solids: list[trimesh.Trimesh] = []
    for index, cutter in enumerate(cutters, start=1):
        solids.extend(cutter_components(cutter, f"cutter {index}"))

    print(f"Subtracting {len(solids)} closed cutter component(s) from one box...")
    for index, solid in enumerate(solids, start=1):
        result = trimesh.boolean.difference(
            [current, solid],
            engine=engine,
            check_volume=True,
        )
        current = mesh_from_boolean_result(result, f"Boolean difference {index}")
        if not current.is_volume:
            current = meshcutout.cleanup_mesh(current)
        if not current.is_volume:
            current = meshcutout.repair_volume_mesh(current, f"box after cutter {index}")
        if not current.is_volume:
            raise meshcutout.MeshCutoutError(
                f"The box is not a closed volume after cutter component {index}."
            )
    return current


def build_box_set(args: argparse.Namespace) -> trimesh.Trimesh:
    validate_args(args)

    cavities: list[trimesh.Trimesh] = []
    debug_meshes: dict[str, trimesh.Trimesh] = {}
    for index, input_path in enumerate(args.inputs, start=1):
        print(f"Generating cutout {index}/{len(args.inputs)}: {input_path}")
        cutout_args = build_cutout_args(args, input_path)
        if args.debug_dir is not None:
            object_debug_dir = args.debug_dir / f"{index:03d}_{input_path.stem}"
            cutout_args.debug_dir = object_debug_dir
        cavity = meshcutout.build_cavity(cutout_args)
        cavities.append(cavity)
        debug_meshes[f"20_cavity_{index:03d}"] = cavity

    box_min, box_extents, minimum_extents = resolve_box_bounds(cavities, args)

    lid_mesh = None
    lid_cutter = None
    if getattr(args, "sliding_lid", False):
        orig_box_top = float(box_min[2] + box_extents[2])
        
        inner_x = box_extents[0] - 2 * args.margin
        inner_y = box_extents[1] - 2 * args.margin
        slide_axis = "Y" if inner_y >= inner_x else "X"
        
        if slide_axis == "Y":
            box_extents[0] += 2 * args.lid_slot_depth
            box_min[0] -= args.lid_slot_depth
            margin_x = args.margin + args.lid_slot_depth
            margin_y = args.margin
        else:
            box_extents[1] += 2 * args.lid_slot_depth
            box_min[1] -= args.lid_slot_depth
            margin_x = args.margin
            margin_y = args.margin + args.lid_slot_depth

        lid_cutter, lid_mesh, extra_z = build_sliding_lid_system(box_min, box_extents, margin_x, margin_y, orig_box_top, args)
        box_extents[2] += extra_z

    box_top = float(box_min[2] + box_extents[2])
    # The cavity should only stretch up to the original top if we have a lid
    extension_top = orig_box_top + args.top_overlap if getattr(args, "sliding_lid", False) else box_top + args.top_overlap
    print(f"Minimum XYZ dimensions: {meshcutout.format_vector(minimum_extents)} mm.")
    print(f"Box XYZ dimensions: {meshcutout.format_vector(box_extents)} mm.")

    box = boxcutout.make_box(box_extents, min_corner=box_min)
    cutters: list[trimesh.Trimesh] = []
    for index, cavity in enumerate(cavities, start=1):
        cutter = extend_cavity_entry_to_top(cavity, extension_top, args, f"cavity {index}")
        cutters.append(cutter)
        debug_meshes[f"21_cutter_{index:03d}"] = cutter

    if lid_cutter is not None:
        cutters.append(lid_cutter)
        debug_meshes["21_lid_cutter"] = lid_cutter

    debug_meshes["22_box_solid"] = box
    export_debug_meshes(
        args.debug_dir,
        debug_meshes,
        ascii_stl=meshcutout.should_export_ascii_stl(args),
    )

    result = subtract_cutters_sequentially(box, cutters, args.boolean_engine)

    if not result.is_volume:
        result = meshcutout.cleanup_mesh(result)
    if not result.is_volume:
        result = meshcutout.repair_volume_mesh(result, "final box")

    if lid_mesh is not None:
        lid_mesh.apply_translation([box_extents[0] + args.margin * 2.0, 0, box_min[2] - lid_mesh.bounds[0, 2]])
        result = meshcutout.boolean_union_or_concatenate([result, lid_mesh], engine=args.boolean_engine, label="box and lid")

    if not result.is_volume:
        raise meshcutout.MeshCutoutError("The final box is not a closed volume.")

    export_debug_meshes(
        args.debug_dir,
        {"23_box_with_cutouts": result},
        ascii_stl=meshcutout.should_export_ascii_stl(args),
    )
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one STL box with one cutout per input STL.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input STL files.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output STL. If omitted, uses boxset_cutout.stl.",
    )
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument(
        "--box-size",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
    )
    parser.add_argument("--box-x", type=float, default=None)
    parser.add_argument("--box-y", type=float, default=None)
    parser.add_argument("--box-z", type=float, default=None)
    parser.add_argument("--strict-dimensions", action="store_true")
    parser.add_argument("--top-overlap", type=float, default=0.5)

    parser.add_argument("--clearance", type=float, default=0.2)
    parser.add_argument(
        "--offset-mode",
        choices=["silhouette", "surface", "none"],
        default="silhouette",
    )
    parser.add_argument("--surface-cell-size", type=float, default=None)
    parser.add_argument("--extrude-extra", type=float, default=None)
    parser.add_argument("--orientation", choices=["auto", "none"], default="auto")
    parser.add_argument("--rotate-x", type=float, default=0.0)
    parser.add_argument("--rotate-y", type=float, default=0.0)
    parser.add_argument("--rotate-z", type=float, default=0.0)
    parser.add_argument("--secondary-min-area-ratio", type=float, default=0.03)
    parser.add_argument(
        "--buffer-join",
        choices=sorted(meshcutout.JOIN_STYLES.keys()),
        default="round",
    )
    parser.add_argument(
        "--boolean-engine",
        choices=["manifold", "blender", "auto"],
        default="manifold",
    )
    parser.add_argument(
        "--fast-projection",
        dest="precise_projection",
        action="store_false",
        default=True,
    )
    parser.add_argument("--keep-projection-holes", action="store_true")
    parser.add_argument("--projection-simplify", type=float, default=0.0001)
    parser.add_argument("--entry-top-extra", type=float, default=3.0)
    parser.add_argument(
        "--cavity-method",
        choices=meshcutout.CAVITY_METHODS,
        default="xyz-entry",
    )
    parser.add_argument("--entry-clearance-extra", type=float, default=0.005)
    parser.add_argument("--entry-cut-extra", type=float, default=0.1)
    parser.add_argument("--sweep-slices", type=int, default=64)
    parser.add_argument("--sweep-pitch", type=float, default=0.1)
    parser.add_argument("--sweep-overcut", type=float, default=None)
    parser.add_argument("--finger-scoop", action="store_true")
    parser.add_argument(
        "--finger-scoop-side",
        choices=meshcutout.FINGER_SCOOP_SIDES,
        default="auto",
    )
    parser.add_argument("--finger-scoop-radius", type=float, default=8.0)
    parser.add_argument("--finger-scoop-depth", type=float, default=12.0)
    parser.add_argument("--finger-scoop-z-depth", type=float, default=10.0)
    parser.add_argument("--finger-scoop-overlap", type=float, default=1.0)
    parser.add_argument("--sliding-lid", action="store_true")
    parser.add_argument("--lid-thickness", type=float, default=2.0)
    parser.add_argument("--lid-clearance", type=float, default=0.2)
    parser.add_argument("--lid-slot-depth", type=float, default=2.0)
    parser.add_argument("--lid-top-lip", type=float, default=2.0)
    parser.add_argument("--simplify-faces", type=int, default=None)
    parser.add_argument("--simplify-ratio", type=float, default=0.1)
    parser.add_argument(
        "--stl-format",
        choices=["auto", "binary", "ascii"],
        default="auto",
    )
    parser.add_argument("--no-rezero", action="store_true")
    parser.add_argument("--debug-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.output is None:
            args.output = Path("boxset_cutout.stl")
        if args.boolean_engine == "auto":
            args.boolean_engine = None

        result = build_box_set(args)
        meshcutout.export_mesh(
            result,
            args.output,
            ascii_stl=meshcutout.should_export_ascii_stl(args),
        )
        print(f"Output: {args.output}")
        print(f"Final XYZ dimensions: {meshcutout.format_vector(result.extents)} mm.")
        print(f"Final volume: {result.volume:.3f} mm^3.")
        print(f"Closed volume: {'yes' if result.is_volume else 'no'}.")
        return 0
    except meshcutout.MeshCutoutError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
