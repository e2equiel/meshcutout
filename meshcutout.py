#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import shapely
import trimesh
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from trimesh.transformations import rotation_matrix, transform_points


AXIS_X = np.array([1.0, 0.0, 0.0])
AXIS_Y = np.array([0.0, 1.0, 0.0])
AXIS_Z = np.array([0.0, 0.0, 1.0])

JOIN_STYLES = {
    "round": 1,
    "mitre": 2,
    "bevel": 3,
}

CAVITY_METHODS = ("xyz-entry", "sweep-z")
FINGER_SCOOP_SIDES = ("auto", "+x", "-x", "+y", "-y")


class MeshCutoutError(RuntimeError):
    pass


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise MeshCutoutError("Zero-length vector.")
    return vector / length


def load_mesh(path: Path) -> trimesh.Trimesh:
    if not path.exists():
        raise MeshCutoutError(f"Input file does not exist: {path}")

    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise MeshCutoutError("The file does not contain triangulated meshes.")
        loaded = trimesh.util.concatenate(meshes)

    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshCutoutError("Could not read the file as a Trimesh mesh.")

    return cleanup_mesh(loaded)


def cleanup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.remove_infinite_values()
    if len(mesh.faces) == 0:
        raise MeshCutoutError("The mesh has no faces.")

    try:
        mesh.update_faces(mesh.nondegenerate_faces())
    except Exception:
        pass

    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    try:
        mesh.fix_normals(multibody=True)
    except TypeError:
        mesh.fix_normals()

    return mesh


def repair_volume_mesh(mesh: trimesh.Trimesh, label: str) -> trimesh.Trimesh:
    def fill_and_cleanup(candidate: trimesh.Trimesh) -> trimesh.Trimesh:
        trimesh.repair.fill_holes(candidate)
        try:
            trimesh.repair.fix_normals(candidate, multibody=True)
        except TypeError:
            trimesh.repair.fix_normals(candidate)
        return cleanup_mesh(candidate)

    repaired = cleanup_mesh(mesh)
    if repaired.is_volume:
        return repaired

    broken_before = len(trimesh.repair.broken_faces(repaired))
    repaired = fill_and_cleanup(repaired)

    if not repaired.is_volume:
        exported = repaired.export(file_type="stl")
        if isinstance(exported, str):
            exported = exported.encode("utf-8")
        reloaded = trimesh.load(
            BytesIO(exported),
            file_type="stl",
            force="mesh",
            process=True,
        )
        repaired = fill_and_cleanup(reloaded)

    broken_after = len(trimesh.repair.broken_faces(repaired))
    if repaired.is_volume:
        print(f"Repaired {label}: broken faces {broken_before} -> {broken_after}.")

    return repaired


def align_vector(mesh: trimesh.Trimesh, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = unit(source)
    target = unit(target)

    if np.dot(source, target) > 1.0 - 1e-10:
        matrix = np.eye(4)
    else:
        matrix = trimesh.geometry.align_vectors(source, target)

    mesh.apply_transform(matrix)
    return matrix


def rotate_around_z_to_y(mesh: trimesh.Trimesh, normal: np.ndarray) -> np.ndarray:
    projected = np.array([normal[0], normal[1], 0.0])
    if np.linalg.norm(projected) < 1e-9:
        return np.eye(4)

    source = unit(projected)
    target = AXIS_Y
    angle = np.arctan2(np.cross(source, target)[2], np.dot(source, target))
    matrix = rotation_matrix(angle, AXIS_Z)
    mesh.apply_transform(matrix)
    return matrix


def apply_manual_rotations(
    mesh: trimesh.Trimesh,
    rotate_x: float,
    rotate_y: float,
    rotate_z: float,
) -> trimesh.Trimesh:
    mesh = mesh.copy()
    rotations = [
        (rotate_x, AXIS_X, "X"),
        (rotate_y, AXIS_Y, "Y"),
        (rotate_z, AXIS_Z, "Z"),
    ]

    for degrees, axis, label in rotations:
        if abs(degrees) < 1e-12:
            continue
        matrix = rotation_matrix(np.deg2rad(degrees), axis, point=mesh.centroid)
        mesh.apply_transform(matrix)
        print(f"Manual {label} rotation: {degrees:.3f} degrees.")

    return mesh


def orient_with_facets(
    mesh: trimesh.Trimesh,
    secondary_min_area_ratio: float,
) -> tuple[trimesh.Trimesh, list[str]]:
    mesh = mesh.copy()
    notes: list[str] = []

    if len(mesh.facets) == 0:
        notes.append("No coplanar facets found; using bounding-box orientation.")
        return orient_with_bounds(mesh), notes

    areas = np.asarray(mesh.facets_area)
    normals = np.asarray(mesh.facets_normal)
    if len(areas) == 0 or not np.isfinite(areas).all():
        notes.append("Invalid facets; using bounding-box orientation.")
        return orient_with_bounds(mesh), notes

    primary_index = int(np.argmax(areas))
    primary_normal = normals[primary_index]
    primary_area = float(areas[primary_index])
    align_vector(mesh, primary_normal, AXIS_Z)
    notes.append(
        f"Primary face aligned to Z: area {primary_area:.3f}, normal {format_vector(primary_normal)}."
    )

    areas = np.asarray(mesh.facets_area)
    normals = np.asarray(mesh.facets_normal)
    if len(areas) == 0:
        notes.append("Could not recalculate facets for Y orientation.")
        return mesh, notes

    largest = float(np.max(areas))
    vertical = np.abs(normals @ AXIS_Z) < 0.35
    significant = areas >= largest * secondary_min_area_ratio
    candidates = np.nonzero(vertical & significant)[0]

    if len(candidates) == 0:
        notes.append("No significant side face found for Y alignment.")
        return mesh, notes

    secondary_index = int(candidates[np.argmin(areas[candidates])])
    secondary_normal = normals[secondary_index]
    secondary_area = float(areas[secondary_index])
    rotate_around_z_to_y(mesh, secondary_normal)
    notes.append(
        f"Smallest side face aligned to Y: area {secondary_area:.3f}, normal {format_vector(secondary_normal)}."
    )

    return mesh, notes


def orient_with_bounds(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    to_box, extents = trimesh.bounds.oriented_bounds(mesh, ordered=True)
    mesh.apply_transform(to_box)

    order = np.argsort(extents)
    smallest = int(order[0])
    largest = int(order[-1])
    middle = int(order[1])

    # New frame: X = middle dimension, Y = largest dimension, Z = thickness.
    permutation = np.eye(4)
    permutation[:3, :3] = np.eye(3)[[middle, largest, smallest]]
    mesh.apply_transform(permutation)
    return mesh


def format_vector(vector: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.3f}" for v in vector) + "]"


def pymeshlab_surface_offset(
    mesh: trimesh.Trimesh,
    distance: float,
    cell_size: float | None,
) -> trimesh.Trimesh:
    if distance <= 0:
        return mesh.copy()

    if cell_size is None:
        cell_size = distance

    try:
        import pymeshlab
    except ImportError as exc:
        raise MeshCutoutError(
            "Surface mode requires PyMeshLab. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ),
        "input",
    )
    ms.generate_resampled_uniform_mesh(
        cellsize=pymeshlab.PureValue(float(cell_size)),
        offset=pymeshlab.PureValue(float(distance)),
        mergeclosevert=True,
        absdist=True,
    )
    current = ms.current_mesh()
    return cleanup_mesh(
        trimesh.Trimesh(
            vertices=current.vertex_matrix(),
            faces=current.face_matrix(),
            process=True,
        )
    )


def pymeshlab_simplify(
    mesh: trimesh.Trimesh,
    target_faces: int,
) -> trimesh.Trimesh:
    if target_faces <= 0:
        raise MeshCutoutError("--simplify-faces must be greater than zero.")
    if len(mesh.faces) <= target_faces:
        print(
            "Simplification skipped: the mesh already has "
            f"{len(mesh.faces)} faces, less than or equal to the target {target_faces}."
        )
        return mesh.copy()

    try:
        import pymeshlab
    except ImportError as exc:
        raise MeshCutoutError(
            "Simplification requires PyMeshLab. Install dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ),
        "input",
    )
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        preservetopology=True,
        preserveboundary=True,
        preservenormal=True,
        optimalplacement=True,
        planarquadric=True,
        autoclean=True,
    )
    simplified_pm = ms.current_mesh()
    simplified = cleanup_mesh(
        trimesh.Trimesh(
            vertices=simplified_pm.vertex_matrix(),
            faces=simplified_pm.face_matrix(),
            process=True,
        )
    )
    if not simplified.is_volume:
        raise MeshCutoutError("Simplification did not produce a closed volume.")
    return simplified


def polygons_from_projection(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    clearance: float,
    join_style: int,
    precise_projection: bool,
    keep_projection_holes: bool,
    projection_simplify: float,
) -> list[Polygon]:
    path = mesh.projected(normal=normal, precise=precise_projection)
    try:
        polygons = list(path.polygons_full)
    except ModuleNotFoundError as exc:
        missing = exc.name or "an optional dependency"
        raise MeshCutoutError(
            f"Missing {missing}, which Trimesh uses to convert projections to polygons. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    polygons = [p for p in polygons if isinstance(p, Polygon) and not p.is_empty and p.area > 1e-9]
    if not polygons:
        raise MeshCutoutError(f"Projection {format_vector(normal)} did not generate polygons.")

    geometry = unary_union(polygons)
    if clearance > 0:
        geometry = geometry.buffer(clearance, join_style=join_style)
    geometry = geometry.buffer(0)
    if not keep_projection_holes:
        geometry = fill_projection_holes(geometry)
    if projection_simplify > 0:
        geometry = geometry.simplify(projection_simplify, preserve_topology=True).buffer(0)

    result = [p for p in iter_polygons(geometry) if p.area > 1e-9]
    if not result:
        raise MeshCutoutError(f"Projection {format_vector(normal)} is empty after buffering.")
    return result


def iter_polygons(geometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            if isinstance(item, Polygon):
                yield item


def fill_projection_holes(geometry):
    polygons = [Polygon(p.exterior) for p in iter_polygons(geometry)]
    if not polygons:
        return geometry
    return unary_union(polygons).buffer(0)


def clean_2d_geometry(
    geometry,
    keep_holes: bool,
    simplify: float,
):
    geometry = geometry.buffer(0)
    if not keep_holes:
        geometry = fill_projection_holes(geometry)
    if simplify > 0:
        geometry = geometry.simplify(simplify, preserve_topology=True).buffer(0)
    return geometry


def make_projection_prism(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    clearance: float,
    extrude_extra: float,
    join_style: int,
    precise_projection: bool,
    keep_projection_holes: bool,
    projection_simplify: float,
) -> trimesh.Trimesh:
    normal = unit(normal)
    to_2d = trimesh.geometry.plane_transform(origin=None, normal=normal)
    local_vertices = transform_points(mesh.vertices, to_2d)
    depth_min = float(local_vertices[:, 2].min())
    depth_max = float(local_vertices[:, 2].max())
    return make_projection_prism_span(
        mesh=mesh,
        normal=normal,
        clearance=clearance,
        depth_min=depth_min - extrude_extra / 2.0,
        depth_max=depth_max + extrude_extra / 2.0,
        join_style=join_style,
        precise_projection=precise_projection,
        keep_projection_holes=keep_projection_holes,
        projection_simplify=projection_simplify,
    )


def make_projection_prism_span(
    mesh: trimesh.Trimesh,
    normal: np.ndarray,
    clearance: float,
    depth_min: float,
    depth_max: float,
    join_style: int,
    precise_projection: bool,
    keep_projection_holes: bool,
    projection_simplify: float,
) -> trimesh.Trimesh:
    normal = unit(normal)
    to_2d = trimesh.geometry.plane_transform(origin=None, normal=normal)
    depth = depth_max - depth_min
    if depth <= 0:
        raise MeshCutoutError(f"Invalid depth for normal {format_vector(normal)}.")

    pieces = []
    for polygon in polygons_from_projection(
        mesh=mesh,
        normal=normal,
        clearance=clearance,
        join_style=join_style,
        precise_projection=precise_projection,
        keep_projection_holes=keep_projection_holes,
        projection_simplify=projection_simplify,
    ):
        pieces.append(trimesh.creation.extrude_polygon(polygon, height=depth))

    prism = trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    prism.apply_translation([0.0, 0.0, depth_min])
    prism.apply_transform(np.linalg.inv(to_2d))
    return repair_volume_mesh(prism, f"projection prism {format_vector(normal)}")


def boolean_intersection(meshes: list[trimesh.Trimesh], engine: str | None) -> trimesh.Trimesh:
    repaired_meshes = [
        repair_volume_mesh(mesh, f"intermediate solid {index}")
        for index, mesh in enumerate(meshes, start=1)
    ]
    for index, mesh in enumerate(repaired_meshes, start=1):
        if not mesh.is_volume:
            broken = len(trimesh.repair.broken_faces(mesh))
            raise MeshCutoutError(
                f"Intermediate solid {index} is not a closed volume; boolean operation is unsafe. "
                f"watertight={mesh.is_watertight}, winding={mesh.is_winding_consistent}, "
                f"broken_faces={broken}."
            )

    result = trimesh.boolean.intersection(repaired_meshes, engine=engine, check_volume=True)
    if result is None or result.is_empty:
        raise MeshCutoutError("Boolean intersection returned an empty mesh.")
    return cleanup_mesh(result)


def boolean_union_or_concatenate(
    meshes: list[trimesh.Trimesh],
    engine: str | None,
    label: str,
) -> trimesh.Trimesh:
    repaired_meshes = [
        repair_volume_mesh(mesh, f"{label} solid {index}")
        for index, mesh in enumerate(meshes, start=1)
    ]
    for index, mesh in enumerate(repaired_meshes, start=1):
        if not mesh.is_volume:
            raise MeshCutoutError(f"{label} solid {index} is not a closed volume.")

    try:
        result = trimesh.boolean.union(repaired_meshes, engine=engine, check_volume=True)
        result = mesh_from_boolean_result(result, f"{label} union")
        if result.is_volume:
            return result
        result = repair_volume_mesh(result, f"{label} union")
        if result.is_volume:
            return result
    except Exception as exc:
        print(f"{label} boolean union skipped: {exc}")

    result = cleanup_mesh(trimesh.util.concatenate(repaired_meshes))
    if not result.is_volume:
        result = repair_volume_mesh(result, label)
    if not result.is_volume:
        raise MeshCutoutError(f"{label} is not a closed volume.")
    return result


def mesh_from_boolean_result(result, label: str) -> trimesh.Trimesh:
    if result is None or result.is_empty:
        raise MeshCutoutError(f"{label} returned an empty mesh.")
    if isinstance(result, trimesh.Scene):
        meshes = [g for g in result.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise MeshCutoutError(f"{label} did not return meshes.")
        result = trimesh.util.concatenate(meshes)
    return cleanup_mesh(result)


def build_xyz_entry_cavity(
    base_cavity: trimesh.Trimesh,
    working: trimesh.Trimesh,
    projection_clearance: float,
    args: argparse.Namespace,
) -> trimesh.Trimesh:
    split_clearance = max(0.0, projection_clearance - args.entry_clearance_extra)
    entry_depth_min = float(base_cavity.bounds[0, 2])
    entry_depth_max = float(base_cavity.bounds[1, 2] + args.entry_top_extra)

    print("Building XYZ-entry cavity without sweep voxels...")
    split_prism = make_projection_prism_span(
        mesh=working,
        normal=AXIS_Z,
        clearance=split_clearance,
        depth_min=entry_depth_min,
        depth_max=entry_depth_max,
        join_style=JOIN_STYLES[args.buffer_join],
        precise_projection=args.precise_projection,
        keep_projection_holes=args.keep_projection_holes,
        projection_simplify=args.projection_simplify,
    )

    print("Cutting XYZ cavity from the vertical Z entry prism...")
    entry_remainder = trimesh.boolean.difference(
        [split_prism, base_cavity],
        engine=args.boolean_engine,
        check_volume=True,
    )
    entry_remainder = mesh_from_boolean_result(entry_remainder, "Entry prism difference")
    if not entry_remainder.is_volume:
        entry_remainder = repair_volume_mesh(entry_remainder, "entry prism difference")

    top_z = float(split_prism.bounds[1, 2])
    tolerance = max(1e-5, float(split_prism.extents[2]) * 1e-6)
    components = entry_remainder.split(only_watertight=False)
    top_components = [
        component
        for component in components
        if float(component.bounds[1, 2]) >= top_z - tolerance
    ]
    if not top_components:
        raise MeshCutoutError("Could not identify the upper entry piece.")

    split_top_entry = (
        trimesh.util.concatenate(top_components)
        if len(top_components) > 1
        else top_components[0]
    )
    if not split_top_entry.is_volume:
        split_top_entry = cleanup_mesh(split_top_entry)
    if not split_top_entry.is_volume:
        repaired_top_entry = repair_volume_mesh(split_top_entry, "upper entry piece")
        if repaired_top_entry.is_volume:
            split_top_entry = repaired_top_entry
        else:
            broken = len(trimesh.repair.broken_faces(split_top_entry))
            print(
                "Upper entry reference is not a closed volume; "
                f"continuing with bounds only. broken_faces={broken}."
            )

    print(f"Keeping {len(top_components)} upper entry piece(s); removing lower remainder.")
    entry_cut_clearance = projection_clearance + args.entry_cut_extra
    entry_start_z = float(
        min(component.bounds[0, 2] for component in top_components)
    )
    top_entry = make_projection_prism_span(
        mesh=working,
        normal=AXIS_Z,
        clearance=entry_cut_clearance,
        depth_min=entry_start_z,
        depth_max=entry_depth_max,
        join_style=JOIN_STYLES[args.buffer_join],
        precise_projection=args.precise_projection,
        keep_projection_holes=args.keep_projection_holes,
        projection_simplify=args.projection_simplify,
    )
    if not top_entry.is_volume:
        top_entry = repair_volume_mesh(top_entry, "expanded upper entry cutter")
    if not top_entry.is_volume:
        raise MeshCutoutError("Expanded upper entry cutter is not a closed volume.")

    print(
        "Expanded upper entry cutter: "
        f"clearance {entry_cut_clearance:.3f} mm from z={entry_start_z:.3f}."
    )
    cavity = trimesh.util.concatenate([base_cavity, top_entry])
    if not cavity.is_volume:
        cavity = repair_volume_mesh(cavity, "XYZ-entry cavity")
    if not cavity.is_volume:
        broken = len(trimesh.repair.broken_faces(cavity))
        raise MeshCutoutError(f"XYZ-entry cavity is not a closed volume. broken_faces={broken}.")

    if args.debug_dir:
        export_debug(
            args.debug_dir,
            {
                "08_entry_z_prism": split_prism,
                "09_entry_remainder": entry_remainder,
                "10_entry_top_piece": split_top_entry,
                "11_entry_cut_prism": top_entry,
            },
            ascii_stl=should_export_ascii_stl(args),
        )

    return cavity


def section_geometry(
    section,
    clearance: float,
    join_style: int,
    keep_holes: bool,
    simplify: float,
):
    if section is None:
        return None
    polygons = [p for p in section.polygons_full if p.area > 1e-8]
    if not polygons:
        return None
    geometry = unary_union(polygons)
    if clearance > 0:
        geometry = geometry.buffer(clearance, join_style=join_style)
    geometry = clean_2d_geometry(geometry, keep_holes=keep_holes, simplify=simplify)
    if geometry.is_empty:
        return None
    return geometry


def build_sweep_z_mesh(
    mesh: trimesh.Trimesh,
    clearance: float,
    z_min: float,
    z_max: float,
    slices: int,
    pitch: float,
    overcut: float,
    join_style: int,
    keep_holes: bool,
    simplify: float,
) -> trimesh.Trimesh:
    source_min, source_max = mesh.bounds[:, 2]
    source_height = source_max - source_min
    if source_height <= 0:
        raise MeshCutoutError("Cannot build sweep-z from a mesh with no height.")

    margin = max(source_height * 0.001, 0.01)
    section_min = source_min + margin
    section_max = source_max - margin
    if section_max <= section_min:
        section_min = source_min
        section_max = source_max

    heights = np.linspace(section_min, section_max, slices)
    sections = mesh.section_multiplane(
        plane_origin=[0.0, 0.0, 0.0],
        plane_normal=AXIS_Z,
        heights=heights,
    )

    cumulative = None
    rows: list[tuple[float, object]] = []
    for z_value, section in zip(heights, sections):
        geometry = section_geometry(
            section,
            clearance=clearance,
            join_style=join_style,
            keep_holes=keep_holes,
            simplify=simplify,
        )
        if geometry is None:
            continue
        cumulative = geometry if cumulative is None else unary_union([cumulative, geometry])
        cumulative = clean_2d_geometry(cumulative, keep_holes=keep_holes, simplify=simplify)
        rows.append((float(z_value), cumulative))

    if not rows:
        raise MeshCutoutError("Could not compute sweep-z sections.")

    full_geometry = rows[-1][1]
    min_x, min_y, max_x, max_y = full_geometry.bounds
    grid_margin = pitch * 3.0
    x_values = np.arange(min_x - grid_margin, max_x + grid_margin + pitch, pitch)
    y_values = np.arange(min_y - grid_margin, max_y + grid_margin + pitch, pitch)
    z_values = np.arange(z_min - pitch, z_max + pitch + pitch, pitch)
    if len(x_values) < 3 or len(y_values) < 3 or len(z_values) < 3:
        raise MeshCutoutError("Sweep-z grid is too small.")

    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    occupancy = np.zeros((len(x_values), len(y_values), len(z_values)), dtype=bool)

    row_index = 0
    current = rows[0][1]
    for z_index, z_value in enumerate(z_values):
        if z_value < z_min or z_value > z_max:
            continue
        while row_index + 1 < len(rows) and rows[row_index + 1][0] <= z_value:
            row_index += 1
            current = rows[row_index][1]
        # Dilating avoids marching cubes underestimating the cavity and blocking insertion.
        raster_geometry = current.buffer(overcut, join_style=join_style).buffer(0)
        occupancy[:, :, z_index] = shapely.contains_xy(raster_geometry, xx, yy)

    if not occupancy.any():
        raise MeshCutoutError("Sweep-z did not generate occupied voxels.")

    result = trimesh.voxel.ops.matrix_to_marching_cubes(occupancy, pitch=pitch)
    result.apply_translation([x_values[0], y_values[0], z_values[0]])
    result = cleanup_mesh(result)
    if not result.is_volume:
        raise MeshCutoutError("Sweep-z did not produce a closed volume.")
    return result


def choose_finger_scoop_side(cavity: trimesh.Trimesh, side: str) -> str:
    if side != "auto":
        return side

    # Prefer growing the shorter horizontal axis; negative side is a stable default.
    extents = cavity.extents
    return "-y" if extents[0] >= extents[1] else "-x"


def finger_scoop_polygon(
    bounds: np.ndarray,
    side: str,
    radius: float,
    depth: float,
    overlap: float,
) -> Polygon:
    min_corner, max_corner = bounds
    center = (min_corner + max_corner) / 2.0
    line_extra = max(0.0, depth - radius)

    if side == "+x":
        line = LineString(
            [
                (max_corner[0] - overlap, center[1]),
                (max_corner[0] + line_extra, center[1]),
            ]
        )
    elif side == "-x":
        line = LineString(
            [
                (min_corner[0] + overlap, center[1]),
                (min_corner[0] - line_extra, center[1]),
            ]
        )
    elif side == "+y":
        line = LineString(
            [
                (center[0], max_corner[1] - overlap),
                (center[0], max_corner[1] + line_extra),
            ]
        )
    elif side == "-y":
        line = LineString(
            [
                (center[0], min_corner[1] + overlap),
                (center[0], min_corner[1] - line_extra),
            ]
        )
    else:
        raise MeshCutoutError(f"Unknown finger scoop side: {side}")

    polygon = line.buffer(radius, cap_style=1, join_style=1).buffer(0)
    if polygon.is_empty or not isinstance(polygon, Polygon):
        raise MeshCutoutError("Finger scoop polygon is empty.")
    return polygon


def make_finger_scoop_mesh(
    cavity: trimesh.Trimesh,
    side: str,
    radius: float,
    depth: float,
    z_depth: float,
    overlap: float,
) -> tuple[trimesh.Trimesh, str]:
    selected_side = choose_finger_scoop_side(cavity, side)
    bounds = cavity.bounds
    top_z = float(bounds[1, 2])
    if z_depth <= 0:
        bottom_z = float(bounds[0, 2])
    else:
        bottom_z = max(float(bounds[0, 2]), top_z - z_depth)
    height = top_z - bottom_z
    if height <= 1e-8:
        raise MeshCutoutError("Finger scoop Z depth is too small.")

    polygon = finger_scoop_polygon(
        bounds=bounds,
        side=selected_side,
        radius=radius,
        depth=depth,
        overlap=overlap,
    )
    scoop = trimesh.creation.extrude_polygon(polygon, height=height)
    scoop.apply_translation([0.0, 0.0, bottom_z])
    scoop = repair_volume_mesh(scoop, "finger scoop")
    if not scoop.is_volume:
        raise MeshCutoutError("Finger scoop is not a closed volume.")
    return scoop, selected_side


def add_finger_scoop_to_cavity(
    cavity: trimesh.Trimesh,
    args: argparse.Namespace,
) -> trimesh.Trimesh:
    if not args.finger_scoop:
        return cavity

    scoop, selected_side = make_finger_scoop_mesh(
        cavity=cavity,
        side=args.finger_scoop_side,
        radius=args.finger_scoop_radius,
        depth=args.finger_scoop_depth,
        z_depth=args.finger_scoop_z_depth,
        overlap=args.finger_scoop_overlap,
    )
    print(
        "Finger scoop: "
        f"side={selected_side}, radius={args.finger_scoop_radius:.3f}, "
        f"depth={args.finger_scoop_depth:.3f}, z_depth={args.finger_scoop_z_depth:.3f}."
    )

    cavity = boolean_union_or_concatenate(
        [cavity, scoop],
        engine=args.boolean_engine,
        label="finger-scoop cavity",
    )

    if args.debug_dir:
        export_debug(
            args.debug_dir,
            {
                "12_finger_scoop": scoop,
                "13_cavity_with_finger_scoop": cavity,
            },
            ascii_stl=should_export_ascii_stl(args),
        )

    return cavity


def export_mesh(mesh: trimesh.Trimesh, path: Path, ascii_stl: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ascii_stl and path.suffix.lower() == ".stl":
        exported = mesh.export(file_type="stl_ascii")
        if isinstance(exported, bytes):
            path.write_bytes(exported)
        else:
            path.write_text(exported)
    else:
        mesh.export(path)


def export_debug(
    debug_dir: Path,
    meshes: dict[str, trimesh.Trimesh],
    ascii_stl: bool = False,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for name, mesh in meshes.items():
        export_mesh(mesh, debug_dir / f"{name}.stl", ascii_stl=ascii_stl)


def should_export_ascii_stl(args: argparse.Namespace) -> bool:
    if args.stl_format == "ascii":
        return True
    if args.stl_format == "binary":
        return False
    return False


def build_cavity(args: argparse.Namespace) -> trimesh.Trimesh:
    source = load_mesh(args.input)
    print(f"Input: {len(source.vertices)} vertices, {len(source.faces)} faces.")

    if args.orientation == "auto":
        oriented, notes = orient_with_facets(source, args.secondary_min_area_ratio)
        for note in notes:
            print(note)
    else:
        oriented = source
        print("Automatic orientation disabled.")

    oriented = apply_manual_rotations(
        oriented,
        rotate_x=args.rotate_x,
        rotate_y=args.rotate_y,
        rotate_z=args.rotate_z,
    )
    oriented = cleanup_mesh(oriented)
    working_origin = oriented.bounds[0].copy()
    oriented.apply_translation(-working_origin)

    if args.offset_mode == "surface":
        print("Applying 3D offset with PyMeshLab...")
        working = pymeshlab_surface_offset(oriented, args.clearance, args.surface_cell_size)
        projection_clearance = 0.0
    elif args.offset_mode == "silhouette":
        print("Applying clearance as a 2D buffer on each projection.")
        working = oriented
        projection_clearance = args.clearance
    else:
        print("Offset disabled.")
        working = oriented
        projection_clearance = 0.0

    if args.extrude_extra is None:
        extrude_extra = 2.0 * args.clearance if args.offset_mode == "silhouette" else 0.0
    else:
        extrude_extra = args.extrude_extra

    print(f"Total extra extrusion per axis: {extrude_extra:.3f} mm.")
    hull = cleanup_mesh(working.convex_hull)

    z_prism = make_projection_prism(
        working,
        AXIS_Z,
        projection_clearance,
        extrude_extra,
        JOIN_STYLES[args.buffer_join],
        args.precise_projection,
        args.keep_projection_holes,
        args.projection_simplify,
    )
    y_prism = make_projection_prism(
        hull,
        AXIS_Y,
        projection_clearance,
        extrude_extra,
        JOIN_STYLES[args.buffer_join],
        args.precise_projection,
        args.keep_projection_holes,
        args.projection_simplify,
    )
    x_prism = make_projection_prism(
        hull,
        AXIS_X,
        projection_clearance,
        extrude_extra,
        JOIN_STYLES[args.buffer_join],
        args.precise_projection,
        args.keep_projection_holes,
        args.projection_simplify,
    )

    if args.debug_dir:
        export_debug(
            args.debug_dir,
            {
                "01_oriented": oriented,
                "02_working_offset_source": working,
                "03_convex_hull": hull,
                "04_prism_z": z_prism,
                "05_prism_y": y_prism,
                "06_prism_x": x_prism,
            },
        )

    print("Computing boolean intersection...")
    cavity = boolean_intersection([z_prism, y_prism, x_prism], args.boolean_engine)

    if args.cavity_method == "sweep-z":
        print(f"Building sweep-z with {args.sweep_slices} horizontal sections...")
        sweep_overcut = args.sweep_pitch if args.sweep_overcut is None else args.sweep_overcut
        sweep = build_sweep_z_mesh(
            mesh=cavity,
            clearance=0.0,
            z_min=cavity.bounds[0, 2],
            z_max=cavity.bounds[1, 2] + args.entry_top_extra,
            slices=args.sweep_slices,
            pitch=args.sweep_pitch,
            overcut=sweep_overcut,
            join_style=JOIN_STYLES[args.buffer_join],
            keep_holes=args.keep_projection_holes,
            simplify=args.projection_simplify,
        )
        if args.debug_dir:
            export_debug(
                args.debug_dir,
                {
                    "07_cavity_xyz": cavity,
                    "08_sweep_z": sweep,
                },
                ascii_stl=should_export_ascii_stl(args),
            )
        cavity = sweep
    elif args.cavity_method == "xyz-entry":
        if args.debug_dir:
            export_debug(
                args.debug_dir,
                {"07_cavity_xyz": cavity},
                ascii_stl=should_export_ascii_stl(args),
            )
        cavity = build_xyz_entry_cavity(
            base_cavity=cavity,
            working=working,
            projection_clearance=projection_clearance,
            args=args,
        )
    else:
        raise MeshCutoutError(f"Unknown cavity method: {args.cavity_method}")

    cavity = add_finger_scoop_to_cavity(cavity, args)

    if not args.no_rezero:
        cavity.apply_translation(-cavity.bounds[0])
    else:
        cavity.apply_translation(working_origin)

    simplify_faces = args.simplify_faces
    if simplify_faces is None and args.simplify_ratio is not None:
        simplify_faces = max(4, int(len(cavity.faces) * args.simplify_ratio))

    if simplify_faces is not None:
        before_faces = len(cavity.faces)
        print(f"Simplifying final mesh: {before_faces} -> {simplify_faces} faces...")
        try:
            cavity = pymeshlab_simplify(cavity, simplify_faces)
            print(f"Final simplification: {before_faces} -> {len(cavity.faces)} faces.")
        except MeshCutoutError:
            if args.cavity_method != "xyz-entry":
                raise
            print("Simplification skipped: XYZ-entry simplification did not preserve a closed volume.")

    if args.debug_dir:
        export_debug(
            args.debug_dir,
            {"09_cavity": cavity},
            ascii_stl=should_export_ascii_stl(args),
        )

    return cavity


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an STL cavity from X/Y/Z projections of an input STL.",
    )
    parser.add_argument("input", type=Path, help="Input STL.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output STL. If omitted, uses <input>_cavity.stl.",
    )
    parser.add_argument(
        "--clearance",
        type=float,
        default=0.2,
        help="Expansion/clearance in mm. Default: 0.2.",
    )
    parser.add_argument(
        "--offset-mode",
        choices=["silhouette", "surface", "none"],
        default="silhouette",
        help=(
            "silhouette: 2D projection buffer (robust). "
            "surface: 3D offset with PyMeshLab (more faithful, heavier). "
            "none: no offset."
        ),
    )
    parser.add_argument(
        "--surface-cell-size",
        type=float,
        default=None,
        help="Cell size in mm for surface offset. Default: same as --clearance.",
    )
    parser.add_argument(
        "--extrude-extra",
        type=float,
        default=None,
        help=(
            "Total extra extrusion per axis, in mm. "
            "If omitted: 2 * clearance in silhouette mode; 0 in other modes."
        ),
    )
    parser.add_argument(
        "--orientation",
        choices=["auto", "none"],
        default="auto",
        help="auto aligns the largest face to Z and the smallest side face to Y. Default: auto.",
    )
    parser.add_argument(
        "--rotate-x",
        type=float,
        default=0.0,
        help="Manual rotation in degrees around X, applied after --orientation.",
    )
    parser.add_argument(
        "--rotate-y",
        type=float,
        default=0.0,
        help="Manual rotation in degrees around Y, applied after --orientation.",
    )
    parser.add_argument(
        "--rotate-z",
        type=float,
        default=0.0,
        help="Manual rotation in degrees around Z, applied after --orientation.",
    )
    parser.add_argument(
        "--secondary-min-area-ratio",
        type=float,
        default=0.03,
        help="Ignore side faces below this ratio of the largest face area. Default: 0.03.",
    )
    parser.add_argument(
        "--buffer-join",
        choices=sorted(JOIN_STYLES.keys()),
        default="round",
        help="Corner type for the 2D buffer. Default: round.",
    )
    parser.add_argument(
        "--boolean-engine",
        choices=["manifold", "blender", "auto"],
        default="manifold",
        help="Trimesh boolean backend. Default: manifold.",
    )
    parser.add_argument(
        "--fast-projection",
        dest="precise_projection",
        action="store_false",
        help="Use Trimesh fast projection instead of precise Shapely projection.",
    )
    parser.add_argument(
        "--keep-projection-holes",
        action="store_true",
        help="Keep internal holes in projected silhouettes. By default they are filled.",
    )
    parser.add_argument(
        "--projection-simplify",
        type=float,
        default=0.0001,
        help="Silhouette topology cleanup tolerance in mm. Default: 0.0001.",
    )
    parser.add_argument(
        "--entry-top-extra",
        type=float,
        default=3.0,
        help="How far the vertical entry extends above the cutout, in mm. Default: 3.0.",
    )
    parser.add_argument(
        "--cavity-method",
        choices=CAVITY_METHODS,
        default="xyz-entry",
        help=(
            "xyz-entry: XYZ projection intersection plus a top Z-entry piece. "
            "sweep-z: voxelized vertical sweep. Default: xyz-entry."
        ),
    )
    parser.add_argument(
        "--entry-clearance-extra",
        type=float,
        default=0.005,
        help=(
            "Tiny XY inset for the xyz-entry prism, in mm. "
            "Keeps upper and lower entry remainders separated. Default: 0.005."
        ),
    )
    parser.add_argument(
        "--entry-cut-extra",
        type=float,
        default=0.1,
        help=(
            "Extra XY expansion for the final xyz-entry top cutter, in mm. "
            "Cleans up small boolean leftovers at the entry. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--sweep-slices",
        type=int,
        default=64,
        help="Number of horizontal sections for sweep-z. Default: 64.",
    )
    parser.add_argument(
        "--sweep-pitch",
        type=float,
        default=0.1,
        help="Voxel resolution in mm for sweep-z. Default: 0.1.",
    )
    parser.add_argument(
        "--sweep-overcut",
        type=float,
        default=None,
        help="Extra XY dilation in mm before sweep voxelization. Default: sweep pitch.",
    )
    parser.add_argument(
        "--finger-scoop",
        action="store_true",
        help="Add a rounded side cutout for finger access.",
    )
    parser.add_argument(
        "--finger-scoop-side",
        choices=FINGER_SCOOP_SIDES,
        default="auto",
        help="Side for the finger scoop. Default: auto.",
    )
    parser.add_argument(
        "--finger-scoop-radius",
        type=float,
        default=8.0,
        help="Finger scoop radius in mm. Default: 8.0.",
    )
    parser.add_argument(
        "--finger-scoop-depth",
        type=float,
        default=12.0,
        help="How far the scoop extends outside the cavity in mm. Default: 12.0.",
    )
    parser.add_argument(
        "--finger-scoop-z-depth",
        type=float,
        default=10.0,
        help="How far down from the top the scoop cuts in mm. Use 0 for full depth. Default: 10.0.",
    )
    parser.add_argument(
        "--finger-scoop-overlap",
        type=float,
        default=1.0,
        help="How far the scoop centerline overlaps the cavity side in mm. Default: 1.0.",
    )
    parser.add_argument(
        "--simplify-faces",
        type=int,
        default=None,
        help="Target face count for final PyMeshLab simplification.",
    )
    parser.add_argument(
        "--simplify-ratio",
        type=float,
        default=0.1,
        help=(
            "Face ratio to preserve when simplifying the final mesh. "
            "Default: 0.1. Use --simplify-ratio 0 to disable."
        ),
    )
    parser.add_argument(
        "--stl-format",
        choices=["auto", "binary", "ascii"],
        default="auto",
        help="Output STL format. Default: auto.",
    )
    parser.add_argument(
        "--no-rezero",
        action="store_true",
        help="Do not translate the result so the minimum XYZ becomes 0.",
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
        if args.clearance < 0:
            raise MeshCutoutError("--clearance cannot be negative.")
        if args.surface_cell_size is not None and args.surface_cell_size <= 0:
            raise MeshCutoutError("--surface-cell-size must be greater than zero.")
        if args.projection_simplify < 0:
            raise MeshCutoutError("--projection-simplify cannot be negative.")
        if args.entry_top_extra < 0:
            raise MeshCutoutError("--entry-top-extra cannot be negative.")
        if args.entry_clearance_extra < 0:
            raise MeshCutoutError("--entry-clearance-extra cannot be negative.")
        if args.entry_cut_extra < 0:
            raise MeshCutoutError("--entry-cut-extra cannot be negative.")
        if args.sweep_slices < 4:
            raise MeshCutoutError("--sweep-slices must be at least 4.")
        if args.sweep_pitch <= 0:
            raise MeshCutoutError("--sweep-pitch must be greater than zero.")
        if args.sweep_overcut is not None and args.sweep_overcut < 0:
            raise MeshCutoutError("--sweep-overcut cannot be negative.")
        if args.finger_scoop_radius <= 0:
            raise MeshCutoutError("--finger-scoop-radius must be greater than zero.")
        if args.finger_scoop_depth <= 0:
            raise MeshCutoutError("--finger-scoop-depth must be greater than zero.")
        if args.finger_scoop_z_depth < 0:
            raise MeshCutoutError("--finger-scoop-z-depth cannot be negative.")
        if args.finger_scoop_overlap < 0:
            raise MeshCutoutError("--finger-scoop-overlap cannot be negative.")
        if args.simplify_faces is not None and args.simplify_faces <= 0:
            raise MeshCutoutError("--simplify-faces must be greater than zero.")
        if not 0.0 <= args.simplify_ratio <= 1.0:
            raise MeshCutoutError("--simplify-ratio must be between 0 and 1.")
        if args.simplify_ratio == 0:
            args.simplify_ratio = None

        if args.output is None:
            args.output = args.input.with_name(f"{args.input.stem}_cavity.stl")

        if args.boolean_engine == "auto":
            args.boolean_engine = None

        cavity = build_cavity(args)
        export_mesh(cavity, args.output, ascii_stl=should_export_ascii_stl(args))

        extents = cavity.extents
        print(f"Output: {args.output}")
        print(f"Final XYZ dimensions: {format_vector(extents)} mm.")
        print(f"Final volume: {cavity.volume:.3f} mm^3.")
        print(f"Closed volume: {'yes' if cavity.is_volume else 'no'}.")
        return 0
    except MeshCutoutError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
