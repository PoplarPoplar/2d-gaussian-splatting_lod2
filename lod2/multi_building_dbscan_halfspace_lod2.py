#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于 DBSCAN 实例分割与半空间求交的多建筑 LOD2 重建脚本。

流程严格遵循：
1. 全局 DBSCAN 对多建筑点云做单体化剥离。
2. 对每栋建筑分别进行：
   - roof / wall 语义分离
   - 序列化 RANSAC 拟合垂直墙面和平面屋顶
   - 加入底面约束并统一半空间法向
   - HalfspaceIntersection 纯代数求交
   - ConvexHull 生成绝对水密网格
3. 合并所有单体网格，导出 OBJ 场景模型。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
from sklearn.cluster import DBSCAN

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


# ----------------------------
# 全局可调参数
# ----------------------------
DEFAULT_INPUT = (
    "/home/liu/code/2d-gaussian-splatting_lod2/output/110kv_test/"
    "point_cloud/iteration_30000/building_cleaned.ply"
)
DEFAULT_OUTPUT = (
    "/home/liu/code/2d-gaussian-splatting_lod2/lod2/multi_building_lod2.obj"
)

DBSCAN_EPS = 2.0
DBSCAN_MIN_SAMPLES = 30
MIN_BUILDING_POINTS = 200

WALL_RANSAC_DIST_THRESHOLD = 0.18
ROOF_RANSAC_DIST_THRESHOLD = 0.20
WALL_VERTICAL_NORMAL_Z_MAX = 0.15

WALL_MIN_INLIERS = 180
ROOF_MIN_INLIERS = 180
MAX_WALL_PLANES = 16
MAX_ROOF_PLANES = 8
MAX_RANSAC_ITERS = 800

HALFSPACE_INTERIOR_EPS = 1e-5
PLANE_DUPLICATE_ANGLE_DEG = 5.0
PLANE_DUPLICATE_OFFSET = 0.25


@dataclass
class Plane:
    normal: np.ndarray
    d: float
    support: int
    kind: str

    def as_halfspace(self) -> np.ndarray:
        return np.asarray(
            [self.normal[0], self.normal[1], self.normal[2], self.d],
            dtype=np.float64,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="多建筑点云的 DBSCAN 单体化 + 半空间求交 LOD2 重建。"
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="输入 PLY 路径。")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="输出 OBJ 路径。")
    parser.add_argument("--eps", type=float, default=DBSCAN_EPS, help="全局 DBSCAN eps。")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DBSCAN_MIN_SAMPLES,
        help="全局 DBSCAN min_samples。",
    )
    parser.add_argument(
        "--min-building-points",
        type=int,
        default=MIN_BUILDING_POINTS,
        help="最小建筑点数，小于该值的 DBSCAN 碎块直接跳过。",
    )
    parser.add_argument(
        "--wall-ransac-threshold",
        type=float,
        default=WALL_RANSAC_DIST_THRESHOLD,
        help="墙面 RANSAC 距离阈值。",
    )
    parser.add_argument(
        "--roof-ransac-threshold",
        type=float,
        default=ROOF_RANSAC_DIST_THRESHOLD,
        help="屋顶 RANSAC 距离阈值。",
    )
    parser.add_argument(
        "--wall-normal-z-max",
        type=float,
        default=WALL_VERTICAL_NORMAL_Z_MAX,
        help="墙面法向量允许的 |nz| 最大值，用于强制垂直平面。",
    )
    parser.add_argument(
        "--wall-min-inliers",
        type=int,
        default=WALL_MIN_INLIERS,
        help="接受一个墙面的最小内点数。",
    )
    parser.add_argument(
        "--roof-min-inliers",
        type=int,
        default=ROOF_MIN_INLIERS,
        help="接受一个屋顶面的最小内点数。",
    )
    parser.add_argument(
        "--max-wall-planes",
        type=int,
        default=MAX_WALL_PLANES,
        help="每栋建筑最多提取的墙面数量。",
    )
    parser.add_argument(
        "--max-roof-planes",
        type=int,
        default=MAX_ROOF_PLANES,
        help="每栋建筑最多提取的屋顶数量。",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=MAX_RANSAC_ITERS,
        help="每轮 RANSAC 最大迭代次数。",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def load_semantic_ply(path: str) -> Dict[str, np.ndarray]:
    if PlyData is None:
        raise ImportError(
            "缺少依赖 plyfile。请先安装：pip install plyfile"
        )

    ply = PlyData.read(path)
    vertex = ply["vertex"].data

    required = ["x", "y", "z", "nx", "ny", "nz", "surface_label"]
    missing = [name for name in required if name not in vertex.dtype.names]
    if missing:
        raise ValueError(f"PLY 缺少必要字段: {missing}")

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    normals = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]]).astype(np.float64)
    surface_label = np.asarray(vertex["surface_label"], dtype=np.int32)
    return {"xyz": xyz, "normals": normals, "surface_label": surface_label}


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("零向量无法归一化。")
    return v / n


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    d = -float(np.dot(normal, centroid))
    return normal, d


def fit_vertical_plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    if points.shape[0] < 2:
        raise ValueError("拟合垂直平面至少需要两个点。")

    xy = points[:, :2]
    centroid_xy = xy.mean(axis=0)
    centered_xy = xy - centroid_xy
    _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
    normal_xy = normalize(vh[-1])
    normal = np.array([normal_xy[0], normal_xy[1], 0.0], dtype=np.float64)
    normal = normalize(normal)
    point_on_plane = np.array([centroid_xy[0], centroid_xy[1], points[:, 2].mean()], dtype=np.float64)
    d = -float(np.dot(normal, point_on_plane))
    return normal, d


def plane_point_distances(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return np.abs(points @ normal + d)


def orient_plane_outward(
    normal: np.ndarray,
    d: float,
    interior_point: np.ndarray,
    strict_margin: float = HALFSPACE_INTERIOR_EPS,
) -> Tuple[np.ndarray, float]:
    signed = float(np.dot(normal, interior_point) + d)
    if signed > -strict_margin:
        normal = -normal
        d = -d
        signed = -signed

    if signed > -strict_margin:
        d -= strict_margin + signed
    return normal, d


def is_duplicate_plane(candidate: Plane, planes: Sequence[Plane]) -> bool:
    cos_thresh = np.cos(np.deg2rad(PLANE_DUPLICATE_ANGLE_DEG))
    for plane in planes:
        cos_angle = float(np.dot(candidate.normal, plane.normal))
        if cos_angle < cos_thresh:
            continue
        if abs(candidate.d - plane.d) < PLANE_DUPLICATE_OFFSET:
            return True
    return False


def sample_roof_triplet(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_points = points.shape[0]
    for _ in range(64):
        idx = rng.choice(n_points, size=3, replace=False)
        tri = points[idx]
        cross = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        if np.linalg.norm(cross) > 1e-8:
            return idx
    raise ValueError("屋顶点采样退化，无法找到有效三点组合。")


def sequential_ransac_wall_planes(
    wall_points: np.ndarray,
    interior_point: np.ndarray,
    dist_threshold: float,
    min_inliers: int,
    max_planes: int,
    max_iters: int,
    wall_normal_z_max: float,
) -> List[Plane]:
    if wall_points.shape[0] < min_inliers:
        return []

    remaining = wall_points.copy()
    rng = np.random.default_rng(42)
    planes: List[Plane] = []

    while remaining.shape[0] >= min_inliers and len(planes) < max_planes:
        best_inliers: np.ndarray | None = None
        best_normal: np.ndarray | None = None
        best_d: float | None = None

        for _ in range(max_iters):
            if remaining.shape[0] < 2:
                break
            sample_idx = rng.choice(remaining.shape[0], size=2, replace=False)
            sample = remaining[sample_idx]
            try:
                normal, d = fit_vertical_plane_from_points(sample)
            except ValueError:
                continue

            if abs(normal[2]) > wall_normal_z_max:
                continue

            distances = plane_point_distances(remaining, normal, d)
            inlier_mask = distances <= dist_threshold
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < min_inliers:
                continue

            try:
                refined_normal, refined_d = fit_vertical_plane_from_points(remaining[inlier_mask])
            except ValueError:
                continue

            if abs(refined_normal[2]) > wall_normal_z_max:
                continue

            refined_distances = plane_point_distances(remaining, refined_normal, refined_d)
            refined_mask = refined_distances <= dist_threshold
            refined_count = int(np.count_nonzero(refined_mask))

            if best_inliers is None or refined_count > int(np.count_nonzero(best_inliers)):
                best_inliers = refined_mask
                best_normal = refined_normal
                best_d = refined_d

        if best_inliers is None or best_normal is None or best_d is None:
            break

        support = int(np.count_nonzero(best_inliers))
        best_normal, best_d = orient_plane_outward(best_normal, best_d, interior_point)
        plane = Plane(normal=best_normal, d=best_d, support=support, kind="wall")

        if not is_duplicate_plane(plane, planes):
            planes.append(plane)

        remaining = remaining[~best_inliers]

    return planes


def sequential_ransac_roof_planes(
    roof_points: np.ndarray,
    interior_point: np.ndarray,
    dist_threshold: float,
    min_inliers: int,
    max_planes: int,
    max_iters: int,
) -> List[Plane]:
    if roof_points.shape[0] < min_inliers:
        return []

    remaining = roof_points.copy()
    rng = np.random.default_rng(123)
    planes: List[Plane] = []

    while remaining.shape[0] >= min_inliers and len(planes) < max_planes:
        best_inliers: np.ndarray | None = None
        best_normal: np.ndarray | None = None
        best_d: float | None = None

        for _ in range(max_iters):
            if remaining.shape[0] < 3:
                break
            try:
                sample_idx = sample_roof_triplet(remaining, rng)
            except ValueError:
                break

            sample = remaining[sample_idx]
            try:
                normal, d = fit_plane_svd(sample)
            except ValueError:
                continue

            distances = plane_point_distances(remaining, normal, d)
            inlier_mask = distances <= dist_threshold
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < min_inliers:
                continue

            try:
                refined_normal, refined_d = fit_plane_svd(remaining[inlier_mask])
            except ValueError:
                continue

            refined_distances = plane_point_distances(remaining, refined_normal, refined_d)
            refined_mask = refined_distances <= dist_threshold
            refined_count = int(np.count_nonzero(refined_mask))

            if best_inliers is None or refined_count > int(np.count_nonzero(best_inliers)):
                best_inliers = refined_mask
                best_normal = refined_normal
                best_d = refined_d

        if best_inliers is None or best_normal is None or best_d is None:
            break

        support = int(np.count_nonzero(best_inliers))
        best_normal, best_d = orient_plane_outward(best_normal, best_d, interior_point)
        plane = Plane(normal=best_normal, d=best_d, support=support, kind="roof")

        if not is_duplicate_plane(plane, planes):
            planes.append(plane)

        remaining = remaining[~best_inliers]

    return planes


def build_halfspaces(
    wall_planes: Sequence[Plane],
    roof_planes: Sequence[Plane],
    z_min: float,
    interior_point: np.ndarray,
) -> np.ndarray:
    all_planes: List[Plane] = list(wall_planes) + list(roof_planes)

    bottom_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    bottom_d = float(z_min)
    bottom_normal, bottom_d = orient_plane_outward(bottom_normal, bottom_d, interior_point)
    all_planes.append(Plane(normal=bottom_normal, d=bottom_d, support=0, kind="bottom"))

    if len(all_planes) < 4:
        raise ValueError("有效平面数量不足，无法构成闭合体。")

    halfspaces = np.vstack([plane.as_halfspace() for plane in all_planes])
    interior_eval = halfspaces[:, :3] @ interior_point + halfspaces[:, 3]
    if np.any(interior_eval >= -HALFSPACE_INTERIOR_EPS):
        raise ValueError("内部点不在所有半空间严格内部，HalfspaceIntersection 将失败。")
    return halfspaces


def halfspace_vertices(halfspaces: np.ndarray, interior_point: np.ndarray) -> np.ndarray:
    hs = HalfspaceIntersection(halfspaces, interior_point)
    vertices = np.asarray(hs.intersections, dtype=np.float64)
    if vertices.shape[0] < 4:
        raise ValueError("半空间求交得到的顶点过少。")
    return deduplicate_vertices(vertices)


def deduplicate_vertices(vertices: np.ndarray, decimals: int = 6) -> np.ndarray:
    rounded = np.round(vertices, decimals=decimals)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return vertices[unique_idx]


def convex_hull_mesh(vertices: np.ndarray) -> trimesh.Trimesh:
    hull = ConvexHull(vertices)
    faces = np.asarray(hull.simplices, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    trimesh.repair.fix_normals(mesh, multibody=False)
    if not mesh.is_watertight:
        raise ValueError("生成网格不是水密体。")
    return mesh


def extract_building_mesh(
    building_id: int,
    building_xyz: np.ndarray,
    building_surface_label: np.ndarray,
    args: argparse.Namespace,
) -> trimesh.Trimesh:
    roof_points = building_xyz[building_surface_label == 0]
    wall_points = building_xyz[building_surface_label == 1]

    if roof_points.shape[0] < args.roof_min_inliers:
        raise ValueError(f"Roof 点过少: {roof_points.shape[0]}")
    if wall_points.shape[0] < args.wall_min_inliers:
        raise ValueError(f"Wall 点过少: {wall_points.shape[0]}")

    z_min = float(np.min(building_xyz[:, 2]))
    interior_point = np.mean(building_xyz, axis=0)

    wall_planes = sequential_ransac_wall_planes(
        wall_points=wall_points,
        interior_point=interior_point,
        dist_threshold=args.wall_ransac_threshold,
        min_inliers=args.wall_min_inliers,
        max_planes=args.max_wall_planes,
        max_iters=args.ransac_iters,
        wall_normal_z_max=args.wall_normal_z_max,
    )
    if not wall_planes:
        raise ValueError("未拟合到有效墙面平面。")

    roof_planes = sequential_ransac_roof_planes(
        roof_points=roof_points,
        interior_point=interior_point,
        dist_threshold=args.roof_ransac_threshold,
        min_inliers=args.roof_min_inliers,
        max_planes=args.max_roof_planes,
        max_iters=args.ransac_iters,
    )
    if not roof_planes:
        raise ValueError("未拟合到有效屋顶平面。")

    halfspaces = build_halfspaces(
        wall_planes=wall_planes,
        roof_planes=roof_planes,
        z_min=z_min,
        interior_point=interior_point,
    )
    vertices = halfspace_vertices(halfspaces, interior_point)
    mesh = convex_hull_mesh(vertices)
    mesh.metadata["building_id"] = building_id
    return mesh


def cluster_buildings(xyz: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(xyz)


def valid_building_ids(labels: np.ndarray, min_building_points: int) -> List[int]:
    ids: List[int] = []
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        if label == -1:
            continue
        if count < min_building_points:
            continue
        ids.append(label)
    return ids


def reconstruct_scene(data: Dict[str, np.ndarray], args: argparse.Namespace) -> trimesh.Trimesh:
    xyz = data["xyz"]
    surface_label = data["surface_label"]

    labels = cluster_buildings(xyz, eps=args.eps, min_samples=args.min_samples)
    building_ids = valid_building_ids(labels, min_building_points=args.min_building_points)
    if not building_ids:
        raise RuntimeError("DBSCAN 未得到可用建筑实例，请调大 eps 或减小 min_samples。")

    print(
        f"[INFO] DBSCAN 完成: 共 {len(building_ids)} 栋有效建筑，"
        f"噪声点 {int(np.count_nonzero(labels == -1))} 个。"
    )

    all_building_meshes: List[trimesh.Trimesh] = []

    for building_id in building_ids:
        building_mask = labels == building_id
        building_xyz = xyz[building_mask]
        building_surface_label = surface_label[building_mask]

        roof_count = int(np.count_nonzero(building_surface_label == 0))
        wall_count = int(np.count_nonzero(building_surface_label == 1))
        print(
            f"[INFO] Building {building_id}: 点数={building_xyz.shape[0]}, "
            f"roof={roof_count}, wall={wall_count}"
        )

        try:
            mesh = extract_building_mesh(
                building_id=building_id,
                building_xyz=building_xyz,
                building_surface_label=building_surface_label,
                args=args,
            )
        except (ValueError, QhullError, np.linalg.LinAlgError) as exc:
            print(f"[WARN] Building {building_id} 跳过: {exc}")
            continue
        except Exception as exc:
            print(f"[WARN] Building {building_id} 发生未知错误，已跳过: {exc}")
            continue

        all_building_meshes.append(mesh)
        print(
            f"[INFO] Building {building_id} 完成: 顶点={len(mesh.vertices)}, "
            f"三角面={len(mesh.faces)}, watertight={mesh.is_watertight}"
        )

    if not all_building_meshes:
        raise RuntimeError("没有任何建筑成功完成半空间求交重建。")

    return trimesh.util.concatenate(all_building_meshes)


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output)

    data = load_semantic_ply(args.input)
    scene_mesh = reconstruct_scene(data, args)
    scene_mesh.export(args.output)

    print(
        f"[INFO] 场景导出完成: {args.output} | "
        f"顶点={len(scene_mesh.vertices)}, 三角面={len(scene_mesh.faces)}, "
        f"watertight={scene_mesh.is_watertight}"
    )


if __name__ == "__main__":
    main()
