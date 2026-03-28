#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAIN_PY="$REPO_ROOT/train.py"

DATA_ROOT="/media/liu/my_pssd/program/data_milo_run/wo_f"
OUTPUT_ROOT="/media/liu/my_pssd/program/白模结果/2DGS_OUTPUT"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "未找到训练脚本: $TRAIN_PY"
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "数据目录不存在: $DATA_ROOT"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

failed=0

while IFS= read -r scene_dir; do
  scene_name="$(basename "$scene_dir")"
  out_dir="$OUTPUT_ROOT/$scene_name"

  echo "=============================="
  echo "开始训练: $scene_name"
  echo "数据路径: $scene_dir"
  echo "输出路径: $out_dir"

  if ! python "$TRAIN_PY" \
    -s "$scene_dir" \
    -r 8 \
    --save_iterations 3000 \
    --enable_semantic_training \
    -m "$out_dir"; then
    echo "训练失败: $scene_name"
    failed=1
  else
    echo "训练完成: $scene_name"
  fi
done < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)

echo "=============================="
if [[ $failed -ne 0 ]]; then
  echo "批量训练结束：存在失败场景。"
  exit 1
fi

echo "批量训练结束：全部成功。"