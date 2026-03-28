#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
免网格化 2.5D 语义挤压 (Semantic-Driven 2.5D Extrusion)

功能：
1. 读取带语义标签的纯净建筑点云（PLY）。
2. 使用墙面点生成 2D 建筑足迹。
3. 使用屋顶点法向聚类 + 加权 RANSAC 拟合多个屋顶平面。
4. 基于足迹顶点做 2.5D 高程映射，直接组装为水密 LOD2 白膜。
5. 输出 OBJ 网格。

输入点云属性要求：
- x, y, z
- nx, ny, nz
- opacity
- surface_label: 0=Roof, 1=Wall

说明：
- 该脚本严格遵循“先语义分离，再屋顶平面拟合，最后2.5D垂直挤压”的逻辑。
- 不走体素化 / 网格重建 / 面面布尔求交流程，避免复杂拓扑操作。
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from plyfile import PlyData
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.cluster import DBSCAN, KMeans


EPS = 1e-8


@dataclass
class RoofPlane:
    """保存单个屋顶坡面的拟合结果。"""

    plane_id: int
    coefficients: np.ndarray  # [a, b, c], z = a*x + b*y + c
    point_indices: np.ndarray
    inlier_indices: np.ndarray
    centroid_xy: np.ndarray
    z_range: Tuple[float, float]
    mean_normal: np.ndarray
    weight_sum: float


@dataclass
class BuildingLOD2Result:
    """保存单栋建筑的 LOD2 重建结果。"""

    building_id: int
    wall_point_count: int
    roof_point_count: int
    footprint_area: float
    bottom_z: float
    top_z: np.ndarray
    footprint_2d: np.ndarray
    roof_planes: List[RoofPlane]
    mesh: trimesh.Trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将带 roof/wall 语义标签的建筑点云直接转换为 LOD2 白膜 OBJ。"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/liu/code/2d-gaussian-splatting_lod2/output/110kv_test/point_cloud/iteration_30000/building_cleaned.ply",
        help="输入 PLY 点云路径。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/liu/code/2d-gaussian-splatting_lod2/lod2/building_lod2.obj",
        help="输出 OBJ 路径。",
    )
    parser.add_argument(
        "--ground-z-mode",
        type=str,
        default="wall_min",
        choices=["wall_min", "global_min", "zero"],
        help="底面高程策略：墙面最小值 / 全局最小值 / 0。",
    )
    parser.add_argument(
        "--opacity-threshold",
        type=float,
        default=0.5,
        help="低于该透明度的屋顶点默认不参与拟合；若剩余点太少则回退到全部点。",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=400,
        help="每个屋顶簇的加权 RANSAC 迭代次数。",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=0.2,
        help="RANSAC 内点距离阈值（单位与点云坐标一致，通常为米）。",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.18,
        help="法向量 DBSCAN 聚类半径。",
    )
    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=30,
        help="法向量 DBSCAN 最小样本数。",
    )
    parser.add_argument(
        "--max-kmeans-clusters",
        type=int,
        default=4,
        help="当 DBSCAN 失败时，KMeans 的最大簇数。",
    )
    parser.add_argument(
        "--z-margin",
        type=float,
        default=0.8,
        help="顶点映射时，允许屋顶平面在观测 z 范围外的冗余边界。",
    )
    parser.add_argument(
        "--wall-dbscan-eps",
        type=float,
        default=5.0,
        help="墙点 XY 平面聚类半径，用于筛选主建筑簇。",
    )
    parser.add_argument(
        "--wall-dbscan-min-samples",
        type=int,
        default=80,
        help="墙点 XY 平面聚类最小样本数。",
    )
    parser.add_argument(
        "--roof-footprint-buffer",
        type=float,
        default=3.0,
        help="用主建筑足迹筛选屋顶点时的二维缓冲距离。",
    )
    parser.add_argument(
        "--sequential-ransac-min-inliers",
        type=int,
        default=1500,
        help="顺序 RANSAC 接受一个屋顶平面的最小内点数。",
    )
    parser.add_argument(
        "--max-roof-planes",
        type=int,
        default=6,
        help="顺序 RANSAC 最多提取的屋顶平面数量。",
    )
    parser.add_argument(
        "--building-mode",
        type=str,
        default="all",
        choices=["all", "largest"],
        help="处理所有建筑簇，或仅处理最大建筑簇。",
    )
    parser.add_argument(
        "--min-wall-cluster-size",
        type=int,
        default=300,
        help="有效建筑簇的最小墙点数量。",
    )
    parser.add_argument(
        "--save-individual-buildings",
        action="store_true",
        help="除合并 OBJ 外，同时导出每栋建筑的单独 OBJ。",
    )
    return parser.parse_args()


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_semantic_point_cloud(ply_path: str) -> Dict[str, np.ndarray]:
    """
    读取 PLY，并提取几何、法向、透明度和语义标签。

    返回字段：
    - xyz: (N, 3)
    - normals: (N, 3)
    - opacity: (N,)
    - surface_label: (N,)
    """
    ply = PlyData.read(ply_path)
    vertex = ply["vertex"].data

    required = ["x", "y", "z", "nx", "ny", "nz", "opacity", "surface_label"]
    missing = [name for name in required if name not in vertex.dtype.names]
    if missing:
        raise ValueError(f"PLY 缺少必要字段: {missing}")

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
    normals = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]]).astype(np.float64)
    opacity = np.asarray(vertex["opacity"], dtype=np.float64)
    surface_label = np.asarray(vertex["surface_label"], dtype=np.int32)

    return {
        "xyz": xyz,
        "normals": normals,
        "opacity": opacity,
        "surface_label": surface_label,
    }


def split_roof_and_wall_points(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """根据 surface_label 分离屋顶点与墙面点。"""
    xyz = data["xyz"]
    normals = data["normals"]
    opacity = data["opacity"]
    labels = data["surface_label"]

    roof_mask = labels == 0
    wall_mask = labels == 1

    if not np.any(roof_mask):
        raise ValueError("未找到 surface_label == 0 的 Roof 点。")
    if not np.any(wall_mask):
        raise ValueError("未找到 surface_label == 1 的 Wall 点。")

    return {
        "roof_xyz": xyz[roof_mask],
        "roof_normals": normals[roof_mask],
        "roof_opacity": opacity[roof_mask],
        "wall_xyz": xyz[wall_mask],
        "wall_normals": normals[wall_mask],
        "wall_opacity": opacity[wall_mask],
        "all_xyz": xyz,
    }


def polygon_signed_area(coords: np.ndarray) -> float:
    """计算 2D 多边形有向面积，正值表示逆时针。"""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def make_ccw(coords: np.ndarray) -> np.ndarray:
    """确保二维顶点序列为逆时针，便于后续外法向一致。"""
    if polygon_signed_area(coords) < 0:
        return coords[::-1].copy()
    return coords.copy()


def build_footprint_polygon(wall_xyz: np.ndarray) -> Tuple[Polygon, np.ndarray]:
    """
    根据墙面点的 XY 投影生成 2D 足迹。

    这里优先采用：
    1. MultiPoint(...).convex_hull
    2. convex_hull.minimum_rotated_rectangle

    原因：
    - 用户输入为“纯净建筑点云”，目标又强调白膜与规整性；
    - 最小外接旋转矩形能显著提升足迹的正交性与稳定性；
    - 相比 Alpha Shape，参数更少、失败模式更少，更适合作为鲁棒默认方案。
    """
    if wall_xyz.shape[0] < 4:
        raise ValueError("墙面点数量不足，无法构建足迹。")

    wall_xy = wall_xyz[:, :2]
    hull = MultiPoint(wall_xy).convex_hull
    if hull.is_empty:
        raise ValueError("墙面点凸包为空，无法构建足迹。")

    rectangle = hull.minimum_rotated_rectangle
    if rectangle.is_empty or not isinstance(rectangle, Polygon):
        raise ValueError("最小外接旋转矩形构建失败。")

    coords = np.asarray(rectangle.exterior.coords[:-1], dtype=np.float64)
    if coords.shape[0] < 3:
        raise ValueError("足迹顶点数不足。")

    coords = make_ccw(coords)
    polygon = Polygon(coords)
    if not polygon.is_valid or polygon.area < EPS:
        raise ValueError("生成的足迹多边形无效。")
    return polygon, coords


def extract_main_building_wall_cluster(
    wall_xyz: np.ndarray,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> np.ndarray:
    """
    在 XY 平面上从墙点中提取主建筑簇。

    这样做是因为实际语义清洗后的点云里，仍可能残留相邻建筑、附属构件、
    楼间连接物或局部离群立面。若直接对全体墙点做最小外接矩形，
    足迹会被严重放大。
    """
    if wall_xyz.shape[0] < dbscan_min_samples:
        return wall_xyz

    wall_xy = wall_xyz[:, :2]
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(wall_xy)
    cluster_ids = [cid for cid in np.unique(labels) if cid >= 0]
    if not cluster_ids:
        return wall_xyz

    best_cluster_xyz = wall_xyz
    best_score = -math.inf

    for cid in cluster_ids:
        cluster_xyz = wall_xyz[labels == cid]
        if cluster_xyz.shape[0] < 100:
            continue
        try:
            polygon, _ = build_footprint_polygon(cluster_xyz)
            area = max(float(polygon.area), EPS)
        except Exception:
            continue

        count = cluster_xyz.shape[0]
        density_score = count / area
        # 综合偏好：点数多、平面占据紧凑的簇更可能是主建筑实体。
        score = 0.65 * count + 500.0 * density_score
        if score > best_score:
            best_score = score
            best_cluster_xyz = cluster_xyz

    return best_cluster_xyz


def extract_building_wall_clusters(
    wall_xyz: np.ndarray,
    dbscan_eps: float,
    dbscan_min_samples: int,
    min_cluster_size: int,
) -> List[np.ndarray]:
    """
    在 XY 平面聚类墙点，提取多个建筑簇。

    返回值按簇大小从大到小排序，每个元素是一栋候选建筑的墙点集合。
    """
    if wall_xyz.shape[0] < dbscan_min_samples:
        return [wall_xyz]

    wall_xy = wall_xyz[:, :2]
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(wall_xy)
    cluster_ids = [cid for cid in np.unique(labels) if cid >= 0]
    if not cluster_ids:
        return [wall_xyz]

    clusters = []
    for cid in cluster_ids:
        cluster_xyz = wall_xyz[labels == cid]
        if cluster_xyz.shape[0] >= min_cluster_size:
            clusters.append(cluster_xyz)

    if not clusters:
        return [wall_xyz]

    clusters.sort(key=lambda arr: arr.shape[0], reverse=True)
    return clusters


def filter_roof_points_by_footprint(
    roof_xyz: np.ndarray,
    roof_normals: np.ndarray,
    roof_opacity: np.ndarray,
    footprint_polygon: Polygon,
    buffer_distance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    用主建筑足迹在 XY 平面筛选屋顶点。

    由于屋檐、噪声点和相邻建筑的 roof 标签也可能混进来，
    这里用 footprint 的缓冲区做一次语义后的几何裁剪。
    """
    buffered_polygon = footprint_polygon.buffer(buffer_distance)
    mask = np.array(
        [buffered_polygon.contains(Point(x, y)) or buffered_polygon.touches(Point(x, y)) for x, y in roof_xyz[:, :2]],
        dtype=bool,
    )

    return roof_xyz[mask], roof_normals[mask], roof_opacity[mask]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, EPS, None)
    return vectors / norms


def prepare_roof_normals_for_clustering(normals: np.ndarray) -> np.ndarray:
    """
    对屋顶法向做标准化，并消除方向二义性。

    由于同一屋顶坡面可能出现法向朝上/朝下的数值噪声，
    这里将 nz<0 的法向整体翻转到上半球，以免同一坡面被误分成两簇。
    """
    normals = normalize_vectors(normals)
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] *= -1.0
    return normals


def cluster_roof_planes(
    roof_xyz: np.ndarray,
    roof_normals: np.ndarray,
    max_kmeans_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> np.ndarray:
    """
    按法向量聚类屋顶坡面。

    优先 DBSCAN：
    - 不需要先验簇数，适合平顶/双坡/多坡等不同建筑。
    - 能自动过滤离群法向。

    回退 KMeans：
    - 当 DBSCAN 只得到噪声或结果过碎时，使用 KMeans 兜底。
    """
    features = prepare_roof_normals_for_clustering(roof_normals)
    n_points = features.shape[0]

    if n_points < 50:
        return np.zeros(n_points, dtype=np.int32)

    dbscan = DBSCAN(eps=dbscan_eps, min_samples=min(dbscan_min_samples, n_points))
    labels = dbscan.fit_predict(features)

    valid_cluster_ids = [cid for cid in np.unique(labels) if cid >= 0]
    if len(valid_cluster_ids) >= 1:
        if np.mean(labels >= 0) > 0.5:
            noise_mask = labels < 0
            if np.any(noise_mask):
                valid_mask = labels >= 0
                valid_features = features[valid_mask]
                valid_labels = labels[valid_mask]
                centroids = []
                for cid in sorted(valid_cluster_ids):
                    centroids.append(valid_features[valid_labels == cid].mean(axis=0))
                centroids = np.asarray(centroids)

                noise_features = features[noise_mask]
                distances = np.linalg.norm(
                    noise_features[:, None, :] - centroids[None, :, :], axis=2
                )
                nearest = np.argmin(distances, axis=1)
                mapped = np.asarray(sorted(valid_cluster_ids), dtype=np.int32)[nearest]
                labels[noise_mask] = mapped
            return relabel_consecutive(labels)

    # DBSCAN 失败时用 KMeans 兜底，k 不宜过大，否则容易把一个斜屋面切碎。
    max_k = max(1, min(max_kmeans_clusters, n_points // 2000 + 1))
    if max_k <= 1:
        return np.zeros(n_points, dtype=np.int32)

    best_labels = np.zeros(n_points, dtype=np.int32)
    best_inertia = math.inf
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        candidate = model.fit_predict(features)
        inertia = model.inertia_
        # 轻微偏向更少簇，防止无必要的过分裂。
        penalized = inertia * (1.0 + 0.08 * (k - 1))
        if penalized < best_inertia:
            best_inertia = penalized
            best_labels = candidate

    return relabel_consecutive(best_labels)


def relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    unique = sorted(np.unique(labels).tolist())
    mapping = {label: idx for idx, label in enumerate(unique)}
    return np.asarray([mapping[label] for label in labels], dtype=np.int32)


def weighted_plane_fit(points_xyz: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    对 z = a*x + b*y + c 做带权最小二乘。

    记 A = [x, y, 1]，目标是最小化：
        sum_i w_i * (A_i @ beta - z_i)^2
    """
    if points_xyz.shape[0] < 3:
        raise ValueError("拟合平面至少需要 3 个点。")

    A = np.column_stack(
        [points_xyz[:, 0], points_xyz[:, 1], np.ones(points_xyz.shape[0], dtype=np.float64)]
    )
    z = points_xyz[:, 2]
    w = np.clip(weights.astype(np.float64), EPS, None)
    Aw = A * np.sqrt(w)[:, None]
    zw = z * np.sqrt(w)
    coeffs, _, _, _ = np.linalg.lstsq(Aw, zw, rcond=None)
    return coeffs.astype(np.float64)


def plane_predict(coeffs: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """用平面方程 z=ax+by+c 预测高程。"""
    return coeffs[0] * xy[:, 0] + coeffs[1] * xy[:, 1] + coeffs[2]


def fit_plane_from_three_points(points_xyz: np.ndarray) -> Optional[np.ndarray]:
    """
    用三个点显式恢复 z=ax+by+c。

    若三点在 XY 平面近乎共线，则矩阵病态，返回 None。
    """
    A = np.column_stack(
        [
            points_xyz[:, 0],
            points_xyz[:, 1],
            np.ones(points_xyz.shape[0], dtype=np.float64),
        ]
    )
    z = points_xyz[:, 2]
    if abs(np.linalg.det(A)) < EPS:
        return None
    try:
        coeffs = np.linalg.solve(A, z)
    except np.linalg.LinAlgError:
        return None
    return coeffs.astype(np.float64)


def weighted_ransac_plane(
    points_xyz: np.ndarray,
    weights: np.ndarray,
    max_iters: int,
    residual_threshold: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    自实现带权 RANSAC 平面拟合。

    设计动机：
    - sklearn 的 RANSACRegressor 对这种 3D 平面拟合并不够直接；
    - 用户明确要求使用 opacity 作为采样权重或内点评分权重；
    - 因此这里用 opacity 影响“抽样概率 + 内点评分”，再对最佳内点做带权重精拟合。
    """
    n = points_xyz.shape[0]
    if n < 3:
        raise ValueError("RANSAC 平面拟合至少需要 3 个点。")

    rng = np.random.default_rng(random_state)
    weights = np.clip(weights.astype(np.float64), EPS, None)
    probs = weights / np.sum(weights)

    best_score = -math.inf
    best_coeffs: Optional[np.ndarray] = None
    best_inlier_mask: Optional[np.ndarray] = None

    xy = points_xyz[:, :2]
    z = points_xyz[:, 2]

    for _ in range(max_iters):
        sample_ids = rng.choice(n, size=3, replace=False, p=probs)
        coeffs = fit_plane_from_three_points(points_xyz[sample_ids])
        if coeffs is None:
            continue

        pred_z = plane_predict(coeffs, xy)
        residuals = np.abs(pred_z - z)
        inlier_mask = residuals <= residual_threshold
        if np.count_nonzero(inlier_mask) < 3:
            continue

        # 评分同时考虑“内点总权重”和“残差惩罚”，高 opacity 点更容易决定模型优劣。
        inlier_score = float(np.sum(weights[inlier_mask]))
        residual_penalty = float(np.sum(weights[inlier_mask] * residuals[inlier_mask]))
        score = inlier_score - 0.25 * residual_penalty

        if score > best_score:
            best_score = score
            best_coeffs = coeffs
            best_inlier_mask = inlier_mask

    if best_coeffs is None or best_inlier_mask is None:
        # RANSAC 失败时退化为全体点的带权最小二乘。
        coeffs = weighted_plane_fit(points_xyz, weights)
        inlier_mask = np.ones(n, dtype=bool)
        return coeffs, inlier_mask

    refined_coeffs = weighted_plane_fit(points_xyz[best_inlier_mask], weights[best_inlier_mask])
    return refined_coeffs, best_inlier_mask


def fit_roof_planes(
    roof_xyz: np.ndarray,
    roof_normals: np.ndarray,
    roof_opacity: np.ndarray,
    labels: np.ndarray,
    opacity_threshold: float,
    ransac_iters: int,
    ransac_threshold: float,
) -> List[RoofPlane]:
    """对每个屋顶法向簇执行加权 RANSAC 拟合。"""
    planes: List[RoofPlane] = []

    for plane_id in sorted(np.unique(labels).tolist()):
        cluster_idx = np.where(labels == plane_id)[0]
        cluster_xyz = roof_xyz[cluster_idx]
        cluster_normals = roof_normals[cluster_idx]
        cluster_opacity = roof_opacity[cluster_idx]

        valid_mask = cluster_opacity >= opacity_threshold
        if np.count_nonzero(valid_mask) >= 30:
            fit_xyz = cluster_xyz[valid_mask]
            fit_weights = cluster_opacity[valid_mask]
            fit_local_indices = np.where(valid_mask)[0]
        else:
            # 若高 opacity 点太少，则回退到全体点，避免小屋顶直接失效。
            fit_xyz = cluster_xyz
            fit_weights = np.clip(cluster_opacity, 0.1, 1.0)
            fit_local_indices = np.arange(cluster_xyz.shape[0])

        coeffs, inlier_mask = weighted_ransac_plane(
            points_xyz=fit_xyz,
            weights=fit_weights,
            max_iters=ransac_iters,
            residual_threshold=ransac_threshold,
            random_state=42 + plane_id,
        )

        inlier_local_indices = fit_local_indices[inlier_mask]
        inlier_global_indices = cluster_idx[inlier_local_indices]
        inlier_points = roof_xyz[inlier_global_indices]
        inlier_normals = roof_normals[inlier_global_indices]
        inlier_opacity = roof_opacity[inlier_global_indices]

        if inlier_points.shape[0] < 3:
            # 极端退化情况下仍保留整个簇，以保证后续顶点映射可用。
            inlier_global_indices = cluster_idx
            inlier_points = cluster_xyz
            inlier_normals = cluster_normals
            inlier_opacity = np.clip(cluster_opacity, 0.1, 1.0)
            coeffs = weighted_plane_fit(inlier_points, inlier_opacity)

        planes.append(
            RoofPlane(
                plane_id=plane_id,
                coefficients=coeffs,
                point_indices=cluster_idx,
                inlier_indices=inlier_global_indices,
                centroid_xy=inlier_points[:, :2].mean(axis=0),
                z_range=(float(np.min(inlier_points[:, 2])), float(np.max(inlier_points[:, 2]))),
                mean_normal=normalize_vectors(inlier_normals).mean(axis=0),
                weight_sum=float(np.sum(inlier_opacity)),
            )
        )

    if not planes:
        raise ValueError("屋顶平面拟合失败，未得到任何有效屋顶簇。")
    return planes


def fit_roof_planes_sequential_ransac(
    roof_xyz: np.ndarray,
    roof_normals: np.ndarray,
    roof_opacity: np.ndarray,
    ransac_iters: int,
    ransac_threshold: float,
    min_inliers: int,
    max_planes: int,
) -> List[RoofPlane]:
    """
    顺序提取多个屋顶平面。

    适用场景：
    - 屋顶法向分布过于连续，DBSCAN/KMeans 很容易把多个坡面并成一簇；
    - 或者一个大簇内部实际上包含多个空间分离的屋顶平面。

    实现方式：
    1. 在剩余点上做加权 RANSAC，找到当前最强平面；
    2. 取其内点并移除；
    3. 继续在剩余点中寻找下一主平面，直到支持度不足。
    """
    remaining_idx = np.arange(roof_xyz.shape[0], dtype=np.int32)
    planes: List[RoofPlane] = []

    for plane_id in range(max_planes):
        if remaining_idx.shape[0] < max(min_inliers, 3):
            break

        points_xyz = roof_xyz[remaining_idx]
        points_normals = roof_normals[remaining_idx]
        points_weights = np.clip(roof_opacity[remaining_idx], 0.1, 1.0)

        coeffs, inlier_mask = weighted_ransac_plane(
            points_xyz=points_xyz,
            weights=points_weights,
            max_iters=ransac_iters,
            residual_threshold=ransac_threshold,
            random_state=1234 + plane_id,
        )

        if int(np.count_nonzero(inlier_mask)) < min_inliers:
            break

        inlier_global_indices = remaining_idx[inlier_mask]
        inlier_points = roof_xyz[inlier_global_indices]
        inlier_normals = roof_normals[inlier_global_indices]
        inlier_opacity = np.clip(roof_opacity[inlier_global_indices], 0.1, 1.0)

        coeffs = weighted_plane_fit(inlier_points, inlier_opacity)
        planes.append(
            RoofPlane(
                plane_id=plane_id,
                coefficients=coeffs,
                point_indices=inlier_global_indices.copy(),
                inlier_indices=inlier_global_indices.copy(),
                centroid_xy=inlier_points[:, :2].mean(axis=0),
                z_range=(float(np.min(inlier_points[:, 2])), float(np.max(inlier_points[:, 2]))),
                mean_normal=normalize_vectors(inlier_normals).mean(axis=0),
                weight_sum=float(np.sum(inlier_opacity)),
            )
        )

        remaining_idx = remaining_idx[~inlier_mask]

    return planes


def choose_ground_z(
    ground_mode: str,
    wall_xyz: np.ndarray,
    all_xyz: np.ndarray,
) -> float:
    """确定建筑底面高程。"""
    if ground_mode == "wall_min":
        return float(np.min(wall_xyz[:, 2]))
    if ground_mode == "global_min":
        return float(np.min(all_xyz[:, 2]))
    if ground_mode == "zero":
        return 0.0
    raise ValueError(f"未知 ground_z_mode: {ground_mode}")


def map_footprint_vertices_to_roof(
    footprint_2d: np.ndarray,
    roof_planes: Sequence[RoofPlane],
    roof_xyz: np.ndarray,
    z_margin: float,
) -> np.ndarray:
    """
    将 2D 足迹顶点映射到屋顶高程。

    策略：
    1. 每个顶点对所有屋顶平面求 z 值；
    2. 保留落在该平面观测高程范围附近的“有效候选”；
    3. 在有效候选中优先选择 XY 重心最近的屋顶平面；
    4. 若没有有效候选，则回退到“所有平面预测值里的最小值”。

    这样做的原因：
    - “最低有效 z”能避免顶面穿出实际包络之外；
    - “最近邻坡面”又能让双坡屋顶在不同角点更贴近各自坡面。
    """
    all_z = []
    chosen_plane_ids = []

    global_roof_min = float(np.min(roof_xyz[:, 2]))
    global_roof_max = float(np.max(roof_xyz[:, 2]))

    for vertex_xy in footprint_2d:
        candidates = []
        for plane in roof_planes:
            predicted_z = float(
                plane_predict(plane.coefficients, vertex_xy[None, :])[0]
            )
            z_min, z_max = plane.z_range
            valid = (z_min - z_margin) <= predicted_z <= (z_max + z_margin)
            distance_xy = float(np.linalg.norm(vertex_xy - plane.centroid_xy))
            candidates.append((plane, predicted_z, valid, distance_xy))

        valid_candidates = [item for item in candidates if item[2]]
        if valid_candidates:
            valid_candidates.sort(key=lambda item: (item[3], item[1]))
            chosen_plane, chosen_z, _, _ = valid_candidates[0]
        else:
            # 退化时，按用户建议取“最低的有效 z”思想的保守回退版。
            fallback = min(candidates, key=lambda item: item[1])
            chosen_plane, chosen_z, _, _ = fallback

        chosen_z = float(np.clip(chosen_z, global_roof_min, global_roof_max + z_margin))
        all_z.append(chosen_z)
        chosen_plane_ids.append(chosen_plane.plane_id)

    return np.asarray(all_z, dtype=np.float64)


def triangulate_convex_polygon(indices: Sequence[int], reverse: bool = False) -> List[List[int]]:
    """
    对凸多边形用扇形法三角化。

    这里足迹默认来自 minimum_rotated_rectangle，因此是凸四边形；
    即便未来改成一般凸足迹，扇形三角化依然成立。
    """
    indices = list(indices)
    faces = []
    for i in range(1, len(indices) - 1):
        tri = [indices[0], indices[i], indices[i + 1]]
        if reverse:
            tri = [tri[0], tri[2], tri[1]]
        faces.append(tri)
    return faces


def build_watertight_mesh(
    footprint_2d: np.ndarray,
    bottom_z: float,
    top_z: np.ndarray,
) -> trimesh.Trimesh:
    """
    基于同一足迹的上下轮廓构造水密 2.5D 网格。

    顶点组织：
    - 前 N 个：底面顶点
    - 后 N 个：顶面顶点

    面组织：
    - 底面：朝下
    - 顶面：朝上
    - 侧墙：每条边形成一个四边形，再拆成两个三角形
    """
    n = footprint_2d.shape[0]
    if n < 3:
        raise ValueError("足迹顶点数不足，无法构建网格。")

    bottom_vertices = np.column_stack(
        [footprint_2d[:, 0], footprint_2d[:, 1], np.full(n, bottom_z, dtype=np.float64)]
    )
    top_vertices = np.column_stack([footprint_2d[:, 0], footprint_2d[:, 1], top_z])
    vertices = np.vstack([bottom_vertices, top_vertices])

    bottom_indices = list(range(n))
    top_indices = list(range(n, 2 * n))

    faces: List[List[int]] = []
    faces.extend(triangulate_convex_polygon(bottom_indices, reverse=True))
    faces.extend(triangulate_convex_polygon(top_indices, reverse=False))

    for i in range(n):
        j = (i + 1) % n
        bi, bj = bottom_indices[i], bottom_indices[j]
        ti, tj = top_indices[i], top_indices[j]

        # 对 CCW 足迹，侧墙外法向对应如下三角拆分。
        faces.append([bi, bj, tj])
        faces.append([bi, tj, ti])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)

    # 修正极少数情况下的面朝向异常。
    trimesh.repair.fix_normals(mesh, multibody=False)
    return mesh


def summarize_roof_planes(roof_planes: Sequence[RoofPlane]) -> str:
    lines = []
    for plane in roof_planes:
        a, b, c = plane.coefficients.tolist()
        z_min, z_max = plane.z_range
        lines.append(
            "  - Plane #{pid}: z = {a:.4f}x + {b:.4f}y + {c:.4f}, "
            "points={pts}, inliers={inliers}, z_range=[{z0:.3f}, {z1:.3f}]".format(
                pid=plane.plane_id,
                a=a,
                b=b,
                c=c,
                pts=len(plane.point_indices),
                inliers=len(plane.inlier_indices),
                z0=z_min,
                z1=z_max,
            )
        )
    return "\n".join(lines)


def export_individual_building_meshes(
    results: Sequence[BuildingLOD2Result],
    merged_output_path: str,
) -> None:
    """将每栋建筑单独导出到与合并 OBJ 相邻的目录。"""
    output_abs = os.path.abspath(merged_output_path)
    stem = os.path.splitext(os.path.basename(output_abs))[0]
    out_dir = os.path.join(os.path.dirname(output_abs), f"{stem}_buildings")
    os.makedirs(out_dir, exist_ok=True)

    for result in results:
        path = os.path.join(out_dir, f"building_{result.building_id:03d}.obj")
        result.mesh.export(path)


def reconstruct_single_building(
    building_id: int,
    wall_xyz: np.ndarray,
    all_roof_xyz: np.ndarray,
    all_roof_normals: np.ndarray,
    all_roof_opacity: np.ndarray,
    all_xyz: np.ndarray,
    args: argparse.Namespace,
) -> Optional[BuildingLOD2Result]:
    """
    对单栋建筑执行完整的 LOD2 生成流程。
    """
    footprint_polygon, footprint_2d = build_footprint_polygon(wall_xyz)
    roof_xyz, roof_normals, roof_opacity = filter_roof_points_by_footprint(
        roof_xyz=all_roof_xyz,
        roof_normals=all_roof_normals,
        roof_opacity=all_roof_opacity,
        footprint_polygon=footprint_polygon,
        buffer_distance=args.roof_footprint_buffer,
    )
    if roof_xyz.shape[0] < 100:
        return None

    roof_labels = cluster_roof_planes(
        roof_xyz=roof_xyz,
        roof_normals=roof_normals,
        max_kmeans_clusters=args.max_kmeans_clusters,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
    )

    roof_planes = fit_roof_planes(
        roof_xyz=roof_xyz,
        roof_normals=roof_normals,
        roof_opacity=roof_opacity,
        labels=roof_labels,
        opacity_threshold=args.opacity_threshold,
        ransac_iters=args.ransac_iters,
        ransac_threshold=args.ransac_threshold,
    )

    support_ratio = (
        max(len(plane.inlier_indices) for plane in roof_planes) / max(roof_xyz.shape[0], 1)
    )
    if len(roof_planes) == 1 or support_ratio < 0.18:
        sequential_planes = fit_roof_planes_sequential_ransac(
            roof_xyz=roof_xyz,
            roof_normals=roof_normals,
            roof_opacity=roof_opacity,
            ransac_iters=args.ransac_iters,
            ransac_threshold=args.ransac_threshold,
            min_inliers=args.sequential_ransac_min_inliers,
            max_planes=args.max_roof_planes,
        )
        if sequential_planes:
            roof_planes = sequential_planes

    bottom_z = choose_ground_z(
        ground_mode=args.ground_z_mode,
        wall_xyz=wall_xyz,
        all_xyz=all_xyz,
    )
    top_z = map_footprint_vertices_to_roof(
        footprint_2d=footprint_2d,
        roof_planes=roof_planes,
        roof_xyz=roof_xyz,
        z_margin=args.z_margin,
    )
    top_z = np.maximum(top_z, bottom_z + 0.1)

    mesh = build_watertight_mesh(
        footprint_2d=footprint_2d,
        bottom_z=bottom_z,
        top_z=top_z,
    )

    return BuildingLOD2Result(
        building_id=building_id,
        wall_point_count=wall_xyz.shape[0],
        roof_point_count=roof_xyz.shape[0],
        footprint_area=float(footprint_polygon.area),
        bottom_z=bottom_z,
        top_z=top_z,
        footprint_2d=footprint_2d,
        roof_planes=roof_planes,
        mesh=mesh,
    )


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output)

    data = load_semantic_point_cloud(args.input)
    split = split_roof_and_wall_points(data)
    wall_clusters = extract_building_wall_clusters(
        wall_xyz=split["wall_xyz"],
        dbscan_eps=args.wall_dbscan_eps,
        dbscan_min_samples=args.wall_dbscan_min_samples,
        min_cluster_size=args.min_wall_cluster_size,
    )
    if args.building_mode == "largest":
        wall_clusters = wall_clusters[:1]

    results: List[BuildingLOD2Result] = []
    for building_id, wall_xyz in enumerate(wall_clusters):
        result = reconstruct_single_building(
            building_id=building_id,
            wall_xyz=wall_xyz,
            all_roof_xyz=split["roof_xyz"],
            all_roof_normals=split["roof_normals"],
            all_roof_opacity=split["roof_opacity"],
            all_xyz=split["all_xyz"],
            args=args,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise ValueError("未能从当前点云中生成任何有效建筑 LOD2。")

    merged_mesh = trimesh.util.concatenate([result.mesh for result in results])
    merged_mesh.export(args.output)

    if args.save_individual_buildings:
        export_individual_building_meshes(results, args.output)

    print("LOD2 白膜生成完成")
    print(f"输入点云: {args.input}")
    print(f"输出模型: {args.output}")
    print(f"原始墙面点数: {split['wall_xyz'].shape[0]}")
    print(f"原始屋顶点数: {split['roof_xyz'].shape[0]}")
    print(f"检测到建筑簇数量: {len(wall_clusters)}")
    print(f"成功重建建筑数量: {len(results)}")
    print(f"合并网格是否水密: {merged_mesh.is_watertight}")
    print(f"合并网格顶点数: {len(merged_mesh.vertices)}, 面数: {len(merged_mesh.faces)}")

    for result in results:
        print(
            f"Building #{result.building_id}: "
            f"wall_points={result.wall_point_count}, "
            f"roof_points={result.roof_point_count}, "
            f"footprint_area={result.footprint_area:.3f}, "
            f"bottom_z={result.bottom_z:.3f}, "
            f"roof_planes={len(result.roof_planes)}, "
            f"watertight={result.mesh.is_watertight}"
        )


if __name__ == "__main__":
    main()
