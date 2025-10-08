#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import argparse
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import cKDTree


# ---------------------------
# Core pipeline functions
# ---------------------------

def load_point_cloud(path: Path):
    """Load PLY and return full + downsampled point clouds and color array."""
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    has_colors = (np.asarray(pcd.colors).size != 0)
    colors = np.asarray(pcd.colors) if has_colors else None
    return pcd, points, has_colors, colors


def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel_size: float):
    pcd_down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    pts_down = np.asarray(pcd_down.points)
    return pcd_down, pts_down


def build_slice_grid(points_down: np.ndarray,
                     zmin: float,
                     zmax: float,
                     grid_res: float):
    """Slice by Z, rasterize to occupancy grid, return grid + metadata."""
    mask = (points_down[:, 2] > zmin) & (points_down[:, 2] < zmax)
    slice_points = points_down[mask]
    if slice_points.shape[0] == 0:
        raise ValueError("No points found in the selected Z range — adjust SLICE_HEIGHT_*.")

    x, y = slice_points[:, 0], slice_points[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    width = int(np.ceil((x_max - x_min) / grid_res))
    height = int(np.ceil((y_max - y_min) / grid_res))
    if width <= 0 or height <= 0:
        raise ValueError("Computed grid has non-positive width/height. Check grid resolution & data bounds.")

    grid = np.zeros((height, width), dtype=np.uint8)

    ix = ((x - x_min) / grid_res).astype(int)
    iy = ((y - y_min) / grid_res).astype(int)
    iy = height - iy - 1  # flip Y to image coords

    grid[iy, ix] = 255
    return grid, (x_min, x_max, y_min, y_max), (width, height)


def label_rooms_from_grid(grid: np.ndarray,
                          grid_res: float,
                          min_room_area_m2: float):
    """Make free-space image, label rooms, filter small ones, return labeled image."""
    # close small gaps in walls
    grid_closed = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    free_space = cv2.bitwise_not(grid_closed)
    _, free_space_bin = cv2.threshold(free_space, 127, 255, cv2.THRESH_BINARY)

    num_labels, labels = cv2.connectedComponents(free_space_bin)
    # Filter rooms by area
    label_areas_px = {i: np.sum(labels == i) for i in range(1, num_labels)}
    area_m2 = {i: count * (grid_res ** 2) for i, count in label_areas_px.items()}
    valid = [i for i, a in area_m2.items() if a >= min_room_area_m2]

    labels_filtered = labels.copy()
    for i in range(1, num_labels):
        if i not in valid:
            labels_filtered[labels_filtered == i] = 0  # back to background

    # Relabel to compact indices 0..K (0 is background)
    num_lbls_filtered, labels_filtered = cv2.connectedComponents((labels_filtered > 0).astype(np.uint8))
    return labels_filtered, num_lbls_filtered - 1  # (labels, count_without_background)


def fill_walls_evenly(labels_filtered: np.ndarray, steps: int = 10):
    """Propagate labels into wall pixels to close gaps."""
    label_filled = labels_filtered.copy()
    wall_mask = (label_filled == 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(steps):
        dil = cv2.dilate(label_filled.astype(np.uint16), kernel, iterations=1)
        label_filled[wall_mask == 1] = dil[wall_mask == 1]
        if np.all(label_filled != 0):
            break
    return label_filled


def label_points_from_grid(points_xyz: np.ndarray,
                           label_img: np.ndarray,
                           x_min: float, y_min: float,
                           grid_res: float,
                           width: int, height: int):
    """Assign each 3D point a label by projecting to the grid."""
    xs, ys = points_xyz[:, 0], points_xyz[:, 1]
    ixs = np.floor((xs - x_min) / grid_res).astype(int)
    iys = np.floor((ys - y_min) / grid_res).astype(int)
    iy_img = height - iys - 1

    valid = (ixs >= 0) & (ixs < width) & (iy_img >= 0) & (iy_img < height)
    labels_per_point = np.full(points_xyz.shape[0], -1, dtype=int)
    idx_valid = np.nonzero(valid)[0]
    if idx_valid.size:
        px = ixs[idx_valid]
        py = iy_img[idx_valid]
        labels_per_point[idx_valid] = label_img[py, px]
    return labels_per_point


def assign_unlabeled_with_kdtree(labels_per_point: np.ndarray,
                                 points_xyz: np.ndarray,
                                 label_img: np.ndarray,
                                 x_min: float, y_min: float,
                                 grid_res: float,
                                 width: int, height: int):
    """For points with label -1, assign nearest labeled pixel using 2D KD-tree."""
    unlabeled = np.where(labels_per_point == -1)[0]
    if unlabeled.size == 0:
        return labels_per_point

    yy, xx = np.where(label_img > 0)
    if xx.size == 0:
        # No labeled pixels; return as-is
        return labels_per_point

    pixel_labels = label_img[yy, xx]
    world_x = xx * grid_res + x_min + grid_res / 2.0
    world_y = (height - yy - 1) * grid_res + y_min + grid_res / 2.0
    pixel_coords = np.column_stack((world_x, world_y))

    tree = cKDTree(pixel_coords)
    unl_xy = np.column_stack((points_xyz[unlabeled, 0], points_xyz[unlabeled, 1]))
    _, nearest_idx = tree.query(unl_xy, k=1)
    labels_per_point[unlabeled] = pixel_labels[nearest_idx]
    return labels_per_point


def split_and_save_rooms(points_xyz: np.ndarray,
                         colors: np.ndarray | None,
                         labels_per_point: np.ndarray,
                         out_dir: Path):
    """Save each room as a PLY. Returns list of (room_id, path, n_points)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unique_labels = np.unique(labels_per_point[labels_per_point >= 0])

    manifest = []
    for lab in unique_labels:
        idx = np.where(labels_per_point == lab)[0]
        room_pcd = o3d.geometry.PointCloud()
        room_pcd.points = o3d.utility.Vector3dVector(points_xyz[idx])
        if colors is not None:
            room_pcd.colors = o3d.utility.Vector3dVector(colors[idx])
        out_path = out_dir / f"room_{int(lab):03d}.ply"
        o3d.io.write_point_cloud(str(out_path), room_pcd)
        manifest.append((int(lab), str(out_path), int(idx.size)))
    return manifest


def save_combined_colored(points_xyz: np.ndarray,
                          labels_per_point: np.ndarray,
                          out_dir: Path,
                          rng_seed: int = 42):
    """Write a single colored PLY (random color per room label)."""
    rng = np.random.default_rng(rng_seed)
    labels = labels_per_point.copy()
    mask = labels >= 0
    unique = np.unique(labels[mask])
    color_map = {lab: rng.random(3) for lab in unique}
    colors = np.zeros((points_xyz.shape[0], 3), dtype=float)
    for lab in unique:
        colors[labels == lab] = color_map[lab]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_path = out_dir / "dense_colored_rooms.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    return out_path


def write_manifest(manifest_rows, out_dir: Path):
    """Write a simple CSV manifest for downstream use (optional but handy)."""
    mpath = out_dir / "rooms_manifest.csv"
    with mpath.open("w", encoding="utf-8") as f:
        f.write("room_id,ply_path,n_points\n")
        for lab, pth, n in manifest_rows:
            f.write(f"{lab},{pth},{n}\n")
    return mpath


# ---------------------------
# Orchestrator
# ---------------------------

def run_split(
    input_file: Path,
    out_dir: Path,
    voxel_size: float = 0.15,
    slice_min: float = 0.25,
    slice_max: float = 2.25,
    grid_res: float = 0.10,
    min_room_area_m2: float = 1.0,
    save_colored_all: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load & preprocess
    pcd, points, has_colors, colors = load_point_cloud(Path(input_file))
    print(f"[load] points: {points.shape}")

    pcd_down, pts_down = voxel_downsample(pcd, voxel_size)
    print(f"[voxel] points: {pts_down.shape}")

    # 2) Slice and rasterize
    grid, (x_min, x_max, y_min, y_max), (width, height) = build_slice_grid(
        pts_down, slice_min, slice_max, grid_res
    )

    # 3) Label rooms in 2D, filter, and fill
    labels_2d, count_rooms = label_rooms_from_grid(grid, grid_res, min_room_area_m2)
    print(f"[rooms] after filtering: {count_rooms}")

    labels_2d_filled = fill_walls_evenly(labels_2d, steps=10)

    # 4) Project labels to 3D points
    labels_per_point = label_points_from_grid(
        points, labels_2d_filled, x_min, y_min, grid_res, width, height
    )
    print(f"[project] initially labeled: {np.count_nonzero(labels_per_point != -1)} / {points.shape[0]}")

    labels_per_point = assign_unlabeled_with_kdtree(
        labels_per_point, points, labels_2d_filled, x_min, y_min, grid_res, width, height
    )
    print(f"[fill] total labeled: {np.count_nonzero(labels_per_point != -1)} / {points.shape[0]}")

    # 5) Save each room PLY + optional combined colored
    manifest = split_and_save_rooms(points, colors if has_colors else None, labels_per_point, out_dir)
    print(f"[save] saved {len(manifest)} room files to {out_dir}")

    if save_colored_all:
        colored_path = save_combined_colored(points, labels_per_point, out_dir)
        print(f"[save] combined colored: {colored_path}")

    mpath = write_manifest(manifest, out_dir)
    print(f"[save] manifest: {mpath}")

    print("✅ Done. Each dense room exported to:", out_dir)


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Split rooms from a PLY point cloud and save per-room PLYs.")
    p.add_argument("--input", type=Path, default=Path("input_xyz/output.ply"))
    p.add_argument("--outdir", type=Path, default=Path("rooms_plys"))
    p.add_argument("--voxel", type=float, default=0.15, help="Voxel downsample size in meters.")
    p.add_argument("--slice-min", type=float, default=0.25, help="Lower Z bound (m).")
    p.add_argument("--slice-max", type=float, default=2.25, help="Upper Z bound (m).")
    p.add_argument("--grid", type=float, default=0.10, help="Grid resolution in meters per pixel.")
    p.add_argument("--min-room-area", type=float, default=1.0, help="Minimum room area (m²) to keep.")
    p.add_argument("--save-colored-all", action="store_true", default=True,
                   help="Write a single dense colored PLY.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_split(
        input_file=args.input,
        out_dir=args.outdir,
        voxel_size=args.voxel,
        slice_min=args.slice_min,
        slice_max=args.slice_max,
        grid_res=args.grid,
        min_room_area_m2=args.min_room_area,
        save_colored_all=args.save_colored_all,
    )
