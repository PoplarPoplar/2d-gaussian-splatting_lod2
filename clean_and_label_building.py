#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_and_label_building.py

用途：
1. 读取 2DGS / 3DGS 高斯点云 PLY。
2. 解码高斯的底层参数（法向量、尺度、透明度）。
3. 基于尺度与透明度执行去噪，拔除巨大拉长的 spiky artifacts。
4. 基于归一化法向量的 z 分量，对建筑表面进行 Roof / Wall / Bottom 打标签。
5. 导出为一个精简后的标准 PLY：只保留后续几何处理中最关键的字段。

注意：
- 当前目标文件是 2DGS 点云，实际 header 中只包含 `scale_0` 和 `scale_1`，
  没有 `scale_2`。因此本脚本会自动读取所有存在的 `scale_*` 字段，
  同时兼容 2DGS（2 个尺度）和 3DGS（3 个尺度）。
- 输出 PLY 只包含：
  x, y, z, nx, ny, nz, opacity, surface_label
"""

from __future__ import annotations

import argparse
import os
from typing import List, Sequence, Tuple

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError as exc:
    raise ImportError("缺少依赖 plyfile，请先安装：pip install plyfile") from exc


# =========================
# 可调全局配置
# =========================
INPUT_PLY_PATH = (
    "/home/liu/code/2d-gaussian-splatting_lod2/output/110kv_test/"
    "point_cloud/iteration_30000/point_cloud.ply"
)
OUTPUT_PLY_PATH = (
    "/home/liu/code/2d-gaussian-splatting_lod2/output/110kv_test/"
    "point_cloud/iteration_30000/building_cleaned.ply"
)

# 尺度百分位阈值：例如 95 表示剔除 S_max 排名前 5% 的极度膨胀高斯。
SCALE_PERCENTILE = 95.0

# 物理透明度阈值：Sigmoid 解码后低于该值的点视为不可信噪声。
OPACITY_THRESHOLD = 0.1

# 法向量 z 分量阈值：约等于 15 度容差。
Z_THRESHOLD = 0.25

# 对于 nz < -Z_THRESHOLD 的点：
# True  表示保留并标记为 2（例如屋檐底面 / 朝下表面）
# False 表示直接剔除
KEEP_DOWNWARD_SURFACES = True


REQUIRED_POSITION_FIELDS = ("x", "y", "z")
REQUIRED_NORMAL_FIELDS = ("nx", "ny", "nz")
REQUIRED_MIN_SCALE_FIELDS = ("scale_0", "scale_1")
REQUIRED_OPACITY_FIELD = "opacity"


def parse_args() -> argparse.Namespace:
    """解析命令行参数，默认值直接来自顶部全局配置。"""
    parser = argparse.ArgumentParser(
        description="对纯净建筑 2DGS/3DGS 高斯点云进行去噪并输出表面标签。"
    )
    parser.add_argument("--input", type=str, default=INPUT_PLY_PATH, help="输入 PLY 文件路径。")
    parser.add_argument("--output", type=str, default=OUTPUT_PLY_PATH, help="输出 PLY 文件路径。")
    parser.add_argument(
        "--scale_percentile",
        type=float,
        default=SCALE_PERCENTILE,
        help="S_max 的百分位阈值，例如 95 表示剔除最膨胀的前 5%%。",
    )
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=OPACITY_THRESHOLD,
        help="物理透明度阈值，低于该值的点会被剔除。",
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=Z_THRESHOLD,
        help="基于归一化法向量 nz 的分类阈值。",
    )
    parser.add_argument(
        "--drop_downward",
        action="store_true",
        help="若设置，则直接剔除 nz < -z_threshold 的点，而不是标记为 2。",
    )
    return parser.parse_args()


def ensure_fields_exist(vertex_data: np.ndarray, fields: Sequence[str], group_name: str) -> None:
    """检查 PLY 顶点结构化数组中是否包含必要字段。"""
    missing = [field for field in fields if field not in vertex_data.dtype.names]
    if missing:
        raise ValueError(
            f"输入 PLY 缺少 {group_name} 所需字段: {missing}。\n"
            f"当前可用字段: {list(vertex_data.dtype.names)}"
        )


def get_scale_field_names(vertex_data: np.ndarray) -> List[str]:
    """
    获取所有 scale_* 字段名，并按索引排序。

    这样做的原因：
    - 3DGS 常见为 scale_0, scale_1, scale_2
    - 当前 2DGS 文件实际只有 scale_0, scale_1
    因此这里不能把尺度字段写死为 3 个。
    """
    scale_names = [name for name in vertex_data.dtype.names if name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda name: int(name.split("_")[-1]))

    # 至少要求 2DGS 的两个主尺度存在。
    missing_min_fields = [field for field in REQUIRED_MIN_SCALE_FIELDS if field not in scale_names]
    if missing_min_fields:
        raise ValueError(
            f"输入 PLY 缺少最基本的尺度字段: {missing_min_fields}。\n"
            f"当前可用字段: {list(vertex_data.dtype.names)}"
        )
    return scale_names


def load_vertex_data(ply_path: str) -> np.ndarray:
    """
    读取 PLY 中的 vertex 结构化数组。

    plyfile 会把 PLY 的属性列解析为 numpy 结构化数组，
    每个字段都可以通过 `vertex_data['x']` 这种方式按列访问。
    """
    ply = PlyData.read(ply_path)
    if "vertex" not in ply:
        raise ValueError(f"PLY 文件中不存在 'vertex' 元素: {ply_path}")
    vertex_data = ply["vertex"].data

    ensure_fields_exist(vertex_data, REQUIRED_POSITION_FIELDS, "坐标")
    ensure_fields_exist(vertex_data, REQUIRED_NORMAL_FIELDS, "法向量")
    ensure_fields_exist(vertex_data, (REQUIRED_OPACITY_FIELD,), "透明度")
    get_scale_field_names(vertex_data)
    return vertex_data


def decode_positions(vertex_data: np.ndarray) -> np.ndarray:
    """从结构化数组中提取 xyz，输出形状为 [N, 3]。"""
    return np.stack(
        [
            np.asarray(vertex_data["x"], dtype=np.float32),
            np.asarray(vertex_data["y"], dtype=np.float32),
            np.asarray(vertex_data["z"], dtype=np.float32),
        ],
        axis=1,
    )


def decode_normals(vertex_data: np.ndarray) -> np.ndarray:
    """
    读取显式法向量 nx, ny, nz，并执行 L2 归一化。

    某些点可能出现零向量或接近零向量，因此要对分母做下限保护。
    """
    normals = np.stack(
        [
            np.asarray(vertex_data["nx"], dtype=np.float32),
            np.asarray(vertex_data["ny"], dtype=np.float32),
            np.asarray(vertex_data["nz"], dtype=np.float32),
        ],
        axis=1,
    )
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return normals / norm


def decode_scales(vertex_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    读取所有 scale_* 字段，并从对数尺度空间解码到真实物理尺度空间。

    训练时高斯通常存的是 log(scale)，因此这里要执行 exp()。
    返回：
    - scales_log:  原始对数尺度（仅供调试可追踪）
    - scales_phys: 物理解码后的尺度
    """
    scale_names = get_scale_field_names(vertex_data)
    scales_log = np.stack(
        [np.asarray(vertex_data[name], dtype=np.float32) for name in scale_names],
        axis=1,
    )
    scales_phys = np.exp(scales_log).astype(np.float32)
    return scales_phys, scale_names


def decode_opacity(vertex_data: np.ndarray) -> np.ndarray:
    """
    读取 opacity，并使用 Sigmoid 解码到 [0, 1] 的物理透明度空间。

    这里的原始 opacity 是训练参数空间的 logit，不是物理透明度本身。
    """
    opacity_logits = np.asarray(vertex_data["opacity"], dtype=np.float32)
    opacity_phys = 1.0 / (1.0 + np.exp(-opacity_logits))
    return opacity_phys.astype(np.float32)


def build_denoise_mask(
    scales_phys: np.ndarray,
    opacity_phys: np.ndarray,
    scale_percentile: float,
    opacity_threshold: float,
) -> Tuple[np.ndarray, float]:
    """
    构造尺度与透明度联合去噪掩码。

    规则：
    1. 对每个高斯求 S_max = max(scale_i)
    2. 用百分位阈值剔除过度膨胀的高斯
    3. 同时剔除物理透明度过低的高斯
    """
    s_max = np.max(scales_phys, axis=1)
    scale_cutoff = float(np.percentile(s_max, scale_percentile))

    scale_mask = s_max <= scale_cutoff
    opacity_mask = opacity_phys >= opacity_threshold
    keep_mask = scale_mask & opacity_mask
    return keep_mask, scale_cutoff


def classify_surfaces(
    normals: np.ndarray,
    z_threshold: float,
    keep_downward_surfaces: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于归一化法向量的 nz 分量做分类。

    标签定义：
    - 0: Roof，nz > z_threshold
    - 1: Wall，abs(nz) <= z_threshold
    - 2: Bottom / downward，nz < -z_threshold（可选择保留或删除）

    返回：
    - surface_label: 整型标签数组
    - keep_mask:     若选择删除 downward surfaces，则这里会进一步给出筛选掩码
    """
    nz = normals[:, 2]
    surface_label = np.full(normals.shape[0], 1, dtype=np.int32)

    roof_mask = nz > z_threshold
    wall_mask = np.abs(nz) <= z_threshold
    downward_mask = nz < -z_threshold

    surface_label[roof_mask] = 0
    surface_label[wall_mask] = 1
    surface_label[downward_mask] = 2

    if keep_downward_surfaces:
        keep_mask = np.ones(normals.shape[0], dtype=bool)
    else:
        keep_mask = ~downward_mask

    return surface_label, keep_mask


def apply_mask(*arrays: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    对多个数组施加同一个布尔掩码，确保所有属性严格对齐。

    这是点云清洗脚本里最容易出错的地方之一：
    一旦 xyz / normals / opacity / label 其中某个数组漏掉同一层 mask，
    后续导出的属性就会发生错位。
    """
    return tuple(array[mask] for array in arrays)


def export_cleaned_ply(
    output_path: str,
    xyz: np.ndarray,
    normals: np.ndarray,
    opacity_phys: np.ndarray,
    surface_label: np.ndarray,
) -> None:
    """
    将清洗后的点云编码为标准 PLY。

    导出的 vertex dtype 必须显式定义，否则 plyfile 无法知道每一列的
    字段名与数值类型。
    """
    vertex_dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("nx", np.float32),
            ("ny", np.float32),
            ("nz", np.float32),
            ("opacity", np.float32),
            ("surface_label", np.int32),
        ]
    )

    vertex_data = np.empty(xyz.shape[0], dtype=vertex_dtype)

    # 这里按列回填结构化数组，等价于重新定义 PLY 中的属性列。
    vertex_data["x"] = xyz[:, 0].astype(np.float32)
    vertex_data["y"] = xyz[:, 1].astype(np.float32)
    vertex_data["z"] = xyz[:, 2].astype(np.float32)
    vertex_data["nx"] = normals[:, 0].astype(np.float32)
    vertex_data["ny"] = normals[:, 1].astype(np.float32)
    vertex_data["nz"] = normals[:, 2].astype(np.float32)
    vertex_data["opacity"] = opacity_phys.astype(np.float32)
    vertex_data["surface_label"] = surface_label.astype(np.int32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=False)
    ply.write(output_path)


def main() -> None:
    args = parse_args()

    keep_downward_surfaces = not args.drop_downward

    vertex_data = load_vertex_data(args.input)

    # 步骤 1：底层属性解码
    xyz = decode_positions(vertex_data)
    normals = decode_normals(vertex_data)
    scales_phys, scale_names = decode_scales(vertex_data)
    opacity_phys = decode_opacity(vertex_data)

    # 步骤 2：尺度与形状去噪（拔除尖刺）
    denoise_mask, scale_cutoff = build_denoise_mask(
        scales_phys=scales_phys,
        opacity_phys=opacity_phys,
        scale_percentile=args.scale_percentile,
        opacity_threshold=args.opacity_threshold,
    )
    xyz, normals, opacity_phys = apply_mask(
        xyz, normals, opacity_phys, mask=denoise_mask
    )

    # 步骤 3：基于 z 轴法向量做建筑部件分类
    surface_label, downward_keep_mask = classify_surfaces(
        normals=normals,
        z_threshold=args.z_threshold,
        keep_downward_surfaces=keep_downward_surfaces,
    )
    xyz, normals, opacity_phys, surface_label = apply_mask(
        xyz, normals, opacity_phys, surface_label, mask=downward_keep_mask
    )

    # 步骤 4：整合并导出单一 PLY
    export_cleaned_ply(
        output_path=args.output,
        xyz=xyz,
        normals=normals,
        opacity_phys=opacity_phys,
        surface_label=surface_label,
    )

    total_points = int(len(vertex_data))
    after_denoise_points = int(np.count_nonzero(denoise_mask))
    final_points = int(xyz.shape[0])
    removed_by_denoise = total_points - after_denoise_points
    removed_downward = after_denoise_points - final_points

    roof_count = int(np.count_nonzero(surface_label == 0))
    wall_count = int(np.count_nonzero(surface_label == 1))
    bottom_count = int(np.count_nonzero(surface_label == 2))

    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"检测到尺度字段: {scale_names}")
    print(f"原始点数: {total_points}")
    print(
        "去噪阈值: "
        f"scale_percentile={args.scale_percentile}, "
        f"scale_cutoff={scale_cutoff:.6f}, "
        f"opacity_threshold={args.opacity_threshold}"
    )
    print(f"去噪后点数: {after_denoise_points} (移除 {removed_by_denoise})")
    print(
        f"下向表面处理: {'保留并标记为 2' if keep_downward_surfaces else '直接剔除'}"
    )
    print(f"最终导出点数: {final_points} (额外移除 {removed_downward})")
    print(
        f"标签统计: roof=0 -> {roof_count}, wall=1 -> {wall_count}, bottom=2 -> {bottom_count}"
    )


if __name__ == "__main__":
    main()
