#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_buildings.py

用途：
1. 对“纯净建筑 2DGS / 3DGS 高斯点云”做基于空间位置的 DBSCAN 聚类，得到 Building ID。
2. 按 Building ID 提取单栋建筑。
3. 基于法向量 z 分量，把单栋建筑拆分为 roof / wall / optional bottom 三类。

设计原则：
- 输入输出均基于 plyfile 的结构化数组直接筛选，尽可能原样保留 3DGS 的所有属性字段。
- 不假设 PLY 中只包含 xyz/normal，而是允许包含 f_dc / f_rest / opacity / scale / rot / semantic 等复杂字段。
- 脚本独立运行，不依赖训练时的 torch 模型对象。

依赖：
    pip install numpy scikit-learn plyfile

可选依赖：
    pip install open3d
本脚本当前核心流程不依赖 open3d，仅保留为后续几何调试扩展。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
try:
    from plyfile import PlyData, PlyElement
except ImportError as exc:
    raise ImportError(
        "缺少依赖 plyfile，请先安装：pip install plyfile"
    ) from exc
from sklearn.cluster import DBSCAN


REQUIRED_XYZ_FIELDS = ("x", "y", "z")
REQUIRED_NORMAL_FIELDS = ("nx", "ny", "nz")


@dataclass
class ClusterResult:
    """保存聚类后的核心信息。"""

    raw_labels: np.ndarray
    valid_building_ids: List[int]
    cluster_sizes: Dict[int, int]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="对纯净建筑 2DGS/3DGS 高斯点云进行建筑实例提取与 roof/wall 分类。"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入高斯点云 PLY 文件路径。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录。默认使用输入 PLY 同级目录下的 extracted_buildings。",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=2.0,
        help="DBSCAN 邻域半径（单位通常为米）。建筑彼此有明显间隔时，1.0~3.0 通常较合理。",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=30,
        help="DBSCAN 最小邻域样本数。点越密可以适当调大。",
    )
    parser.add_argument(
        "--target_id",
        type=int,
        default=None,
        help="仅提取指定 Building ID。若不设置，则遍历输出全部建筑。",
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=0.25,
        help="基于法向量 nz 的 roof/wall 分类阈值。默认 0.25。",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=200,
        help="聚类后再次过滤过小类簇，小于该值的类簇会被视为噪点。",
    )
    parser.add_argument(
        "--save_cluster_labels",
        action="store_true",
        help="额外保存建筑实例标签：building_labels.npy、building_clusters.ply、cluster_summary.json。",
    )
    parser.add_argument(
        "--save_bottom",
        action="store_true",
        help="额外保存 nz < -z_threshold 的 bottom/noise 点为 building_{id}_bottom.ply。",
    )
    return parser.parse_args()


def ensure_fields_exist(vertex_data: np.ndarray, fields: Sequence[str], field_group_name: str) -> None:
    """检查结构化数组中是否包含必要字段。"""
    missing = [name for name in fields if name not in vertex_data.dtype.names]
    if missing:
        raise ValueError(
            f"输入 PLY 缺少 {field_group_name} 所需字段: {missing}。"
            f"当前字段包括: {list(vertex_data.dtype.names)}"
        )


def load_vertex_data(ply_path: str) -> Tuple[PlyData, np.ndarray]:
    """
    读取 PLY，并返回原始 PlyData 与 vertex 结构化数组。

    这里不把字段拆散重建，而是保留完整结构化 dtype，
    后续筛选时直接用布尔索引子集化，以确保所有高斯属性完整继承。
    """
    ply = PlyData.read(ply_path)
    try:
        vertex_data = ply["vertex"].data
    except KeyError as exc:
        raise ValueError(f"PLY 文件中不存在 'vertex' 元素: {ply_path}")
    
    ensure_fields_exist(vertex_data, REQUIRED_XYZ_FIELDS, "xyz")
    ensure_fields_exist(vertex_data, REQUIRED_NORMAL_FIELDS, "normal")
    return ply, vertex_data


def structured_xyz(vertex_data: np.ndarray) -> np.ndarray:
    """从结构化数组提取 xyz，形状为 [N, 3]。"""
    xyz = np.stack(
        [
            np.asarray(vertex_data["x"], dtype=np.float32),
            np.asarray(vertex_data["y"], dtype=np.float32),
            np.asarray(vertex_data["z"], dtype=np.float32),
        ],
        axis=1,
    )
    return xyz


def structured_normals(vertex_data: np.ndarray) -> np.ndarray:
    """从结构化数组提取法向量，形状为 [N, 3]。"""
    normals = np.stack(
        [
            np.asarray(vertex_data["nx"], dtype=np.float32),
            np.asarray(vertex_data["ny"], dtype=np.float32),
            np.asarray(vertex_data["nz"], dtype=np.float32),
        ],
        axis=1,
    )
    return normals


def remap_dbscan_labels(raw_labels: np.ndarray, min_cluster_size: int) -> ClusterResult:
    """
    将 DBSCAN 的原始标签重新映射为连续的 Building ID。

    说明：
    - DBSCAN 的噪点标签为 -1。
    - 某些类簇虽然不是 -1，但如果点数极少，也常常是边缘碎块或残留噪点。
      因此这里再加一层 min_cluster_size 过滤，更符合“建筑实例”的语义。
    """
    raw_labels = np.asarray(raw_labels, dtype=np.int32)
    unique_labels, counts = np.unique(raw_labels, return_counts=True)

    kept_labels: List[int] = []
    cluster_sizes: Dict[int, int] = {}
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        if label == -1:
            continue
        if count < min_cluster_size:
            continue
        kept_labels.append(label)
        cluster_sizes[label] = count

    kept_labels = sorted(kept_labels)

    remapped = np.full_like(raw_labels, fill_value=-1)
    valid_building_ids: List[int] = []
    for building_id, old_label in enumerate(kept_labels):
        remapped[raw_labels == old_label] = building_id
        valid_building_ids.append(building_id)

    # 将 cluster_sizes 的键也改成最终导出的 Building ID，便于后续输出统计信息。
    cluster_sizes_by_building_id: Dict[int, int] = {}
    for building_id, old_label in enumerate(kept_labels):
        cluster_sizes_by_building_id[building_id] = cluster_sizes[old_label]

    return ClusterResult(
        raw_labels=remapped,
        valid_building_ids=valid_building_ids,
        cluster_sizes=cluster_sizes_by_building_id,
    )


def cluster_buildings(
    xyz: np.ndarray,
    eps: float,
    min_samples: int,
    min_cluster_size: int,
) -> ClusterResult:
    """
    使用 DBSCAN 对建筑高斯进行空间聚类。

    参数建议：
    - eps:
      建筑之间物理间隔较清晰时，常见可从 1.0~3.0 米开始试。
      若同栋建筑被切碎，适当增大；若相邻建筑被并到一起，适当减小。
    - min_samples:
      点越密、噪声越少时可适当调大。
    - min_cluster_size:
      用于去除 DBSCAN 后的小碎块。
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels = dbscan.fit_predict(xyz)
    return remap_dbscan_labels(raw_labels, min_cluster_size=min_cluster_size)


def clone_ply_with_vertex_data(original_ply: PlyData, new_vertex_data: np.ndarray) -> PlyData:
    """
    用新的 vertex 数据重建一个 PlyData。

    为了最大程度兼容复杂 3DGS PLY，这里沿用原始 vertex 的 dtype，
    只替换数据内容，不手工重排字段。

    说明：
    - 标准 3DGS 导出的 PLY 基本只有一个 vertex element。
    - 若未来存在额外 element（极少见），这里会一并保留非 vertex element。
    """
    new_elements: List[PlyElement] = []
    for element in original_ply.elements:
        if element.name == "vertex":
            new_elements.append(PlyElement.describe(new_vertex_data, "vertex"))
        else:
            new_elements.append(PlyElement.describe(element.data, element.name))
    return PlyData(
        new_elements,
        text=original_ply.text,
        byte_order=original_ply.byte_order,
        comments=list(original_ply.comments),
        obj_info=list(original_ply.obj_info),
    )


def write_subset_ply(original_ply: PlyData, vertex_data: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    """将满足 mask 的点原样筛选后写出为新的 PLY。"""
    selected = vertex_data[mask]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_ply = clone_ply_with_vertex_data(original_ply, selected)
    new_ply.write(out_path)


def append_building_id_field(vertex_data: np.ndarray, building_labels: np.ndarray) -> np.ndarray:
    """
    复制原始结构化数组，并追加一个 int32 的 building_id 字段。

    这样可以直接在一个 PLY 里查看整场景实例分割结果，同时仍保留全部原始高斯属性。
    """
    if vertex_data.shape[0] != building_labels.shape[0]:
        raise ValueError("vertex_data 与 building_labels 长度不一致。")

    old_dtype_descr = list(vertex_data.dtype.descr)
    new_dtype = np.dtype(old_dtype_descr + [("building_id", "<i4")])
    out = np.empty(vertex_data.shape[0], dtype=new_dtype)

    for name in vertex_data.dtype.names:
        out[name] = vertex_data[name]
    out["building_id"] = building_labels.astype(np.int32)
    return out


def save_cluster_artifacts(
    original_ply: PlyData,
    vertex_data: np.ndarray,
    building_labels: np.ndarray,
    cluster_sizes: Dict[int, int],
    output_dir: str,
) -> None:
    """保存建筑聚类标签相关的调试/可视化文件。"""
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "building_labels.npy"), building_labels.astype(np.int32))

    summary = {
        "num_points_total": int(vertex_data.shape[0]),
        "num_points_assigned": int(np.sum(building_labels >= 0)),
        "num_points_noise": int(np.sum(building_labels < 0)),
        "num_buildings": int(len(cluster_sizes)),
        "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
    }
    with open(os.path.join(output_dir, "cluster_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    vertex_with_id = append_building_id_field(vertex_data, building_labels)
    cluster_ply = clone_ply_with_vertex_data(original_ply, vertex_with_id)
    cluster_ply.write(os.path.join(output_dir, "building_clusters.ply"))


def classify_roof_wall_bottom(normals: np.ndarray, z_threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    根据法向量 z 分量做物理分类。

    规则：
    - roof:   nz >  z_threshold
    - wall:  |nz| <= z_threshold
    - bottom: nz < -z_threshold
    """
    nz = normals[:, 2]
    roof_mask = nz > z_threshold
    wall_mask = np.abs(nz) <= z_threshold
    bottom_mask = nz < -z_threshold
    return roof_mask, wall_mask, bottom_mask


def export_single_building_parts(
    original_ply: PlyData,
    vertex_data: np.ndarray,
    building_id: int,
    building_mask: np.ndarray,
    z_threshold: float,
    output_dir: str,
    save_bottom: bool,
) -> Dict[str, int]:
    """
    导出单栋建筑的 roof / wall / optional bottom 三类 PLY。

    这里先对单栋建筑做一次筛选，再在局部上根据 normal.z 分类，
    从而输出完整属性保留的部件级点云。
    """
    building_vertices = vertex_data[building_mask]
    building_normals = structured_normals(building_vertices)

    roof_local_mask, wall_local_mask, bottom_local_mask = classify_roof_wall_bottom(
        building_normals, z_threshold=z_threshold
    )

    os.makedirs(output_dir, exist_ok=True)

    roof_path = os.path.join(output_dir, f"building_{building_id}_roof.ply")
    wall_path = os.path.join(output_dir, f"building_{building_id}_wall.ply")
    write_subset_ply(
        original_ply=original_ply,
        vertex_data=building_vertices,
        mask=roof_local_mask,
        out_path=roof_path,
    )
    write_subset_ply(
        original_ply=original_ply,
        vertex_data=building_vertices,
        mask=wall_local_mask,
        out_path=wall_path,
    )

    if save_bottom:
        bottom_path = os.path.join(output_dir, f"building_{building_id}_bottom.ply")
        write_subset_ply(
            original_ply=original_ply,
            vertex_data=building_vertices,
            mask=bottom_local_mask,
            out_path=bottom_path,
        )

    return {
        "building_id": int(building_id),
        "num_points": int(building_vertices.shape[0]),
        "roof_points": int(np.sum(roof_local_mask)),
        "wall_points": int(np.sum(wall_local_mask)),
        "bottom_points": int(np.sum(bottom_local_mask)),
    }


def resolve_output_dir(input_path: str, output_dir: Optional[str]) -> str:
    """生成输出目录。"""
    if output_dir is not None:
        return output_dir
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), "extracted_buildings")


def validate_target_id(target_id: Optional[int], valid_building_ids: Iterable[int]) -> None:
    """检查目标 Building ID 是否有效。"""
    if target_id is None:
        return
    valid_set = set(valid_building_ids)
    if target_id not in valid_set:
        raise ValueError(
            f"指定的 --target_id={target_id} 不存在。"
            f"当前有效 Building ID: {sorted(valid_set)}"
        )


def main() -> None:
    args = parse_args()

    input_path = os.path.abspath(args.input)
    output_dir = os.path.abspath(resolve_output_dir(input_path, args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 读取输入 PLY: {input_path}")
    ply, vertex_data = load_vertex_data(input_path)

    xyz = structured_xyz(vertex_data)
    normals = structured_normals(vertex_data)

    print(f"[INFO] 点总数: {xyz.shape[0]}")
    print(
        f"[INFO] 开始 DBSCAN 聚类: eps={args.eps}, "
        f"min_samples={args.min_samples}, min_cluster_size={args.min_cluster_size}"
    )
    cluster_result = cluster_buildings(
        xyz=xyz,
        eps=args.eps,
        min_samples=args.min_samples,
        min_cluster_size=args.min_cluster_size,
    )

    building_labels = cluster_result.raw_labels
    num_noise = int(np.sum(building_labels < 0))
    num_assigned = int(np.sum(building_labels >= 0))

    print(f"[INFO] 聚类完成: 有效建筑数量={len(cluster_result.valid_building_ids)}")
    print(f"[INFO] 被标为噪点/碎块的点数: {num_noise}")
    print(f"[INFO] 被分配到建筑实例的点数: {num_assigned}")

    for building_id in cluster_result.valid_building_ids:
        print(f"  - Building {building_id}: {cluster_result.cluster_sizes[building_id]} points")

    if args.save_cluster_labels:
        print("[INFO] 保存聚类标签与统计信息...")
        save_cluster_artifacts(
            original_ply=ply,
            vertex_data=vertex_data,
            building_labels=building_labels,
            cluster_sizes=cluster_result.cluster_sizes,
            output_dir=output_dir,
        )

    validate_target_id(args.target_id, cluster_result.valid_building_ids)

    # 仅为了显式检查 normals 存在且可用，避免后面 roof/wall 分类时出现隐式问题。
    if normals.shape[1] != 3:
        raise ValueError("法向量数组维度异常，无法执行 roof/wall 分类。")

    target_building_ids = (
        [args.target_id] if args.target_id is not None else cluster_result.valid_building_ids
    )

    stats: List[Dict[str, int]] = []
    print(f"[INFO] 开始导出建筑部件，z_threshold={args.z_threshold}")
    for building_id in target_building_ids:
        building_mask = building_labels == building_id
        building_stats = export_single_building_parts(
            original_ply=ply,
            vertex_data=vertex_data,
            building_id=building_id,
            building_mask=building_mask,
            z_threshold=args.z_threshold,
            output_dir=output_dir,
            save_bottom=args.save_bottom,
        )
        stats.append(building_stats)
        print(
            f"[INFO] Building {building_id} 导出完成: "
            f"total={building_stats['num_points']}, "
            f"roof={building_stats['roof_points']}, "
            f"wall={building_stats['wall_points']}, "
            f"bottom={building_stats['bottom_points']}"
        )

    with open(os.path.join(output_dir, "export_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": input_path,
                "eps": args.eps,
                "min_samples": args.min_samples,
                "min_cluster_size": args.min_cluster_size,
                "z_threshold": args.z_threshold,
                "target_id": args.target_id,
                "num_buildings_exported": len(target_building_ids),
                "buildings": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] 全部完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
