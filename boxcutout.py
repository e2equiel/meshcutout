#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

import meshcutout


AXIS_NAMES = ("X", "Y", "Z")


def boolean_difference(
    base: trimesh.Trimesh,
    cutter: trimesh.Trimesh,
    engine: str | None,
) -> trimesh.Trimesh:
    for name, mesh in (("box", base), ("cutout", cutter)):
        if not mesh.is_volume:
            raise meshcutout.MeshCutoutError(
                f"The {name} solid is not a closed volume; boolean operation is unsafe."
            )

    result = trimesh.boolean.difference([base, cutter], engine=engine, check_volume=True)
    if result is None or result.is_empty:
        raise meshcutout.MeshCutoutError("Boolean difference returned an empty mesh.")

    if isinstance(result, trimesh.Scene):
        meshes = [g for g in result.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise meshcutout.MeshCutoutError("Boolean difference did not return meshes.")
        result = trimesh.util.concatenate(meshes)

    result = meshcutout.cleanup_mesh(result)
    if not result.is_volume:
        raise meshcutout.MeshCutoutError("The final box is not a closed volume.")
    return result


def make_box(extents: np.ndarray, min_corner: np.ndarray | None = None) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents)
    if min_corner is None:
        min_corner = np.zeros(3, dtype=float)
    box.apply_translation(min_corner + extents / 2.0)
    return meshcutout.cleanup_mesh(box)


def position_cutout_for_box(
    cutout: trimesh.Trimesh,
    box_extents: np.ndarray,
    top_overlap: float,
) -> trimesh.Trimesh:
    cutter = cutout.copy()
    cutout_min, cutout_max = cutter.bounds
    cutout_extents = cutout_max - cutout_min

    target_min = np.array(
        [
            (box_extents[0] - cutout_extents[0]) / 2.0,
            (box_extents[1] - cutout_extents[1]) / 2.0,
            box_extents[2] - cutout_extents[2] + top_overlap,
        ],
        dtype=float,
    )
    cutter.apply_translation(target_min - cutout_min)
    return meshcutout.cleanup_mesh(cutter)


def requested_box_dimensions(args: argparse.Namespace) -> list[float | None]:
    requested: list[float | None] = [None, None, None]
    if args.box_size is not None:
        requested = list(args.box_size)

    for index, option_name in enumerate(("box_x", "box_y", "box_z")):
        value = getattr(args, option_name)
        if value is not None:
            requested[index] = value

    return requested


def resolve_box_extents(
    cutout: trimesh.Trimesh,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    cutout_extents = cutout.extents
    minimum = np.array(
        [
            cutout_extents[0] + 2.0 * args.margin,
            cutout_extents[1] + 2.0 * args.margin,
            cutout_extents[2] + args.margin,
        ],
        dtype=float,
    )

    requested = requested_box_dimensions(args)
    box_extents = np.array(
        [
            minimum[index] if requested[index] is None else requested[index]
            for index in range(3)
        ],
        dtype=float,
    )

    too_small = box_extents < minimum
    if np.any(too_small):
        details = []
        for index, is_small in enumerate(too_small):
            if is_small:
                details.append(
                    f"{AXIS_NAMES[index]} {box_extents[index]:.3f} -> {minimum[index]:.3f}"
                )
        message = "Dimensions clamped to minimum: " + ", ".join(details) + "."
        if args.strict_dimensions:
            raise meshcutout.MeshCutoutError(message)
        print(message)
        box_extents = np.maximum(box_extents, minimum)

    return box_extents, minimum


def world_box_min_corner(
    cutout: trimesh.Trimesh,
    box_extents: np.ndarray,
    top_overlap: float,
) -> np.ndarray:
    cutout_min, cutout_max = cutout.bounds
    cutout_center = (cutout_min + cutout_max) / 2.0
    return np.array(
        [
            cutout_center[0] - box_extents[0] / 2.0,
            cutout_center[1] - box_extents[1] / 2.0,
            cutout_max[2] - top_overlap - box_extents[2],
        ],
        dtype=float,
    )


def build_cutout_args(args: argparse.Namespace) -> argparse.Namespace:
    cutout_args = meshcutout.parse_args([str(args.input)])
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
        "debug_dir",
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


def validate_args(args: argparse.Namespace, cutout_args: argparse.Namespace) -> None:
    if args.margin < 0:
        raise meshcutout.MeshCutoutError("--margin cannot be negative.")
    if args.top_overlap < 0:
        raise meshcutout.MeshCutoutError("--top-overlap cannot be negative.")

    for index, value in enumerate(requested_box_dimensions(args)):
        if value is not None and value <= 0:
            raise meshcutout.MeshCutoutError(
                f"Box dimension {AXIS_NAMES[index]} must be greater than zero."
            )

    if cutout_args.clearance < 0:
        raise meshcutout.MeshCutoutError("--clearance cannot be negative.")
    if cutout_args.surface_cell_size is not None and cutout_args.surface_cell_size <= 0:
        raise meshcutout.MeshCutoutError("--surface-cell-size must be greater than zero.")
    if cutout_args.projection_simplify < 0:
        raise meshcutout.MeshCutoutError("--projection-simplify cannot be negative.")
    if cutout_args.entry_top_extra < 0:
        raise meshcutout.MeshCutoutError("--entry-top-extra cannot be negative.")
    if cutout_args.entry_clearance_extra < 0:
        raise meshcutout.MeshCutoutError("--entry-clearance-extra cannot be negative.")
    if cutout_args.entry_cut_extra < 0:
        raise meshcutout.MeshCutoutError("--entry-cut-extra cannot be negative.")
    if cutout_args.sweep_slices < 4:
        raise meshcutout.MeshCutoutError("--sweep-slices must be at least 4.")
    if cutout_args.sweep_pitch <= 0:
        raise meshcutout.MeshCutoutError("--sweep-pitch must be greater than zero.")
    if cutout_args.sweep_overcut is not None and cutout_args.sweep_overcut < 0:
        raise meshcutout.MeshCutoutError("--sweep-overcut cannot be negative.")
    if cutout_args.finger_scoop_radius <= 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-radius must be greater than zero.")
    if cutout_args.finger_scoop_depth <= 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-depth must be greater than zero.")
    if cutout_args.finger_scoop_z_depth < 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-z-depth cannot be negative.")
    if cutout_args.finger_scoop_overlap < 0:
        raise meshcutout.MeshCutoutError("--finger-scoop-overlap cannot be negative.")
    if cutout_args.simplify_faces is not None and cutout_args.simplify_faces <= 0:
        raise meshcutout.MeshCutoutError("--simplify-faces must be greater than zero.")
    if cutout_args.simplify_ratio is not None and not 0.0 <= cutout_args.simplify_ratio <= 1.0:
        raise meshcutout.MeshCutoutError("--simplify-ratio must be between 0 and 1.")


def export_debug_meshes(
    args: argparse.Namespace,
    meshes: dict[str, trimesh.Trimesh],
) -> None:
    if args.debug_dir is None:
        return
    meshcutout.export_debug(
        args.debug_dir,
        meshes,
        ascii_stl=meshcutout.should_export_ascii_stl(args),
    )


def build_box_cutout(args: argparse.Namespace) -> trimesh.Trimesh:
    cutout_args = build_cutout_args(args)
    validate_args(args, cutout_args)

    print("Generating cutout...")
    cutout = meshcutout.build_cavity(cutout_args)
    if args.cutout_output is not None:
        meshcutout.export_mesh(
            cutout,
            args.cutout_output,
            ascii_stl=meshcutout.should_export_ascii_stl(args),
        )
        print(f"Cutout saved: {args.cutout_output}")

    box_extents, minimum = resolve_box_extents(cutout, args)
    print(f"Minimum XYZ dimensions: {meshcutout.format_vector(minimum)} mm.")
    print(f"Box XYZ dimensions: {meshcutout.format_vector(box_extents)} mm.")

    if args.no_rezero:
        box = make_box(
            box_extents,
            min_corner=world_box_min_corner(cutout, box_extents, args.top_overlap),
        )
        cutter = cutout.copy()
    else:
        box = make_box(box_extents)
        cutter = position_cutout_for_box(cutout, box_extents, args.top_overlap)

    export_debug_meshes(
        args,
        {
            "10_box_solid": box,
            "11_cutout_positioned": cutter,
        },
    )

    print("Subtracting cutout from box...")
    result = boolean_difference(box, cutter, cutout_args.boolean_engine)
    export_debug_meshes(args, {"12_box_with_cutout": result})
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an STL box with the meshcutout.py cutout already subtracted.",
    )
    parser.add_argument("input", type=Path, help="Input STL.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output STL. If omitted, uses <input>_box.stl.",
    )
    parser.add_argument(
        "--cutout-output",
        type=Path,
        default=None,
        help="Optional: also save the intermediate cutout.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=2.0,
        help=(
            "Margin/wall in mm. In X/Y it is applied per side; in Z it stays below "
            "the cutout to keep the top entry open. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--box-size",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Outer box dimensions in mm. Values are clamped to the minimum if too small.",
    )
    parser.add_argument("--box-x", type=float, default=None, help="Outer X dimension in mm.")
    parser.add_argument("--box-y", type=float, default=None, help="Outer Y dimension in mm.")
    parser.add_argument("--box-z", type=float, default=None, help="Outer Z dimension in mm.")
    parser.add_argument(
        "--strict-dimensions",
        action="store_true",
        help="Fail if a requested dimension is below the minimum, instead of clamping it.",
    )
    parser.add_argument(
        "--top-overlap",
        type=float,
        default=0.5,
        help="Moves the cutter in Z so it passes through the box top face. Default: 0.5.",
    )

    parser.add_argument("--clearance", type=float, default=None, help="Cutout clearance in mm.")
    parser.add_argument(
        "--offset-mode",
        choices=["silhouette", "surface", "none"],
        default=None,
        help="meshcutout.py offset mode.",
    )
    parser.add_argument("--surface-cell-size", type=float, default=None)
    parser.add_argument("--extrude-extra", type=float, default=None)
    parser.add_argument("--orientation", choices=["auto", "none"], default=None)
    parser.add_argument("--rotate-x", type=float, default=None)
    parser.add_argument("--rotate-y", type=float, default=None)
    parser.add_argument("--rotate-z", type=float, default=None)
    parser.add_argument("--secondary-min-area-ratio", type=float, default=None)
    parser.add_argument(
        "--buffer-join",
        choices=sorted(meshcutout.JOIN_STYLES.keys()),
        default=None,
    )
    parser.add_argument(
        "--boolean-engine",
        choices=["manifold", "blender", "auto"],
        default=None,
    )
    parser.add_argument(
        "--fast-projection",
        dest="precise_projection",
        action="store_false",
        default=None,
    )
    parser.add_argument("--keep-projection-holes", action="store_true", default=None)
    parser.add_argument("--projection-simplify", type=float, default=None)
    parser.add_argument("--entry-top-extra", type=float, default=None)
    parser.add_argument(
        "--cavity-method",
        choices=meshcutout.CAVITY_METHODS,
        default=None,
    )
    parser.add_argument("--entry-clearance-extra", type=float, default=None)
    parser.add_argument("--entry-cut-extra", type=float, default=None)
    parser.add_argument("--sweep-slices", type=int, default=None)
    parser.add_argument("--sweep-pitch", type=float, default=None)
    parser.add_argument("--sweep-overcut", type=float, default=None)
    parser.add_argument("--finger-scoop", action="store_true", default=None)
    parser.add_argument(
        "--finger-scoop-side",
        choices=meshcutout.FINGER_SCOOP_SIDES,
        default=None,
    )
    parser.add_argument("--finger-scoop-radius", type=float, default=None)
    parser.add_argument("--finger-scoop-depth", type=float, default=None)
    parser.add_argument("--finger-scoop-z-depth", type=float, default=None)
    parser.add_argument("--finger-scoop-overlap", type=float, default=None)
    parser.add_argument("--simplify-faces", type=int, default=None)
    parser.add_argument("--simplify-ratio", type=float, default=None)
    parser.add_argument(
        "--stl-format",
        choices=["auto", "binary", "ascii"],
        default=None,
        help="Output STL format. Default: auto.",
    )
    parser.add_argument(
        "--no-rezero",
        action="store_true",
        help="Preserve the cutout absolute position and build the box around that position.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional folder for intermediate STL exports.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.output is None:
            args.output = args.input.with_name(f"{args.input.stem}_box.stl")
        if args.stl_format is None:
            args.stl_format = "auto"

        box = build_box_cutout(args)
        meshcutout.export_mesh(
            box,
            args.output,
            ascii_stl=meshcutout.should_export_ascii_stl(args),
        )

        print(f"Output: {args.output}")
        print(f"Final XYZ dimensions: {meshcutout.format_vector(box.extents)} mm.")
        print(f"Final volume: {box.volume:.3f} mm^3.")
        print(f"Closed volume: {'yes' if box.is_volume else 'no'}.")
        return 0
    except meshcutout.MeshCutoutError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
