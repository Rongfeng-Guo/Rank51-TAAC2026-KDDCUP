#!/usr/bin/env bash
# setup_taac_local.sh
# 用法：
#   bash setup_taac_local.sh /path/to/official_baseline
#
# 可选：
#   TAAC_TORCH=cpu bash setup_taac_local.sh /path/to/baseline
#   TAAC_TORCH=cu126 bash setup_taac_local.sh /path/to/baseline
#
# 作用：
#   1) 创建本地 Python 3.10+ 虚拟环境
#   2) 安装官方 baseline 需要的主要依赖
#   3) 下载 Hugging Face demo 数据
#   4) 生成官方代码需要的 schema.json
#   5) 重写 demo parquet 为多个 Row Group，避免 1000 行单 Row Group 导致 train split 为空

set -euo pipefail

PROJECT_DIR="${1:-$(pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"

VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv-taac}"
RAW_DIR="${RAW_DIR:-${PROJECT_DIR}/data/demo_1000_raw}"
READY_DIR="${READY_DIR:-${PROJECT_DIR}/data/demo_1000_ready}"
HF_REPO="${HF_REPO:-TAAC2026/data_sample_1000}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/5] Project dir: ${PROJECT_DIR}"
echo "[1/5] Venv dir:    ${VENV_DIR}"
echo "[1/5] Raw data:    ${RAW_DIR}"
echo "[1/5] Ready data:  ${READY_DIR}"

if [[ ! -f "${PROJECT_DIR}/train.py" || ! -f "${PROJECT_DIR}/run.sh" ]]; then
  echo "ERROR: ${PROJECT_DIR} 下没有找到 train.py/run.sh。请把官方 baseline 目录作为第一个参数传入。"
  exit 1
fi

echo "[2/5] Create virtual environment"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Install PyTorch"
TORCH_CHOICE="${TAAC_TORCH:-auto}"

if [[ "${TORCH_CHOICE}" == "cpu" ]]; then
  python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
elif [[ "${TORCH_CHOICE}" == "cu126" ]]; then
  python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi detected; installing torch 2.7.1 cu126 wheel."
    python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
  else
    echo "No nvidia-smi detected; installing CPU torch wheel."
    python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "[4/5] Install Python packages"
python -m pip install \
  numpy==2.2.5 \
  pandas==2.3.3 \
  pyarrow==23.0.1 \
  scikit-learn==1.7.2 \
  scipy==1.15.3 \
  tqdm==4.67.3 \
  tensorboard==2.19.0 \
  huggingface_hub \
  hf_xet \
  datasets

echo "[5/5] Download HF demo dataset and prepare local data directory"
mkdir -p "${RAW_DIR}" "${READY_DIR}"

python - "${HF_REPO}" "${RAW_DIR}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
raw_dir = Path(sys.argv[2])
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(raw_dir),
    local_dir_use_symlinks=False,
)
print(f"Downloaded {repo_id} to {raw_dir}")
PY

python - "${RAW_DIR}" "${READY_DIR}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

raw_dir = Path(sys.argv[1])
ready_dir = Path(sys.argv[2])
ready_dir.mkdir(parents=True, exist_ok=True)

parquets = sorted(raw_dir.rglob("*.parquet"))
if not parquets:
    raise FileNotFoundError(f"No parquet files found under {raw_dir}")

tables = [pq.read_table(str(p)) for p in parquets]
try:
    table = pa.concat_tables(tables, promote_options="default")
except TypeError:
    table = pa.concat_tables(tables, promote=True)

# 只保留一份训练可见 parquet，且写成多个 Row Group。
# 官方 get_pcvr_data 按 Row Group 切 train/valid；如果只有 1 个 RG，会导致 train=0。
out_parquet = ready_dir / "demo_1000_rg.parquet"
row_group_size = max(1, math.ceil(table.num_rows / 4))
pq.write_table(table, out_parquet, row_group_size=row_group_size)

names = set(table.column_names)

def fid_from_name(name: str) -> int:
    m = re.search(r"_(\d+)$", name)
    if not m:
        raise ValueError(f"Cannot parse fid from {name}")
    return int(m.group(1))

def is_list_type(t):
    return pa.types.is_list(t) or pa.types.is_large_list(t)

def get_array(name: str):
    return table[name].combine_chunks()

def list_len_and_values(arr):
    offsets = np.asarray(arr.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    if len(offsets) <= 1:
        max_len = 0
    else:
        diffs = offsets[1:] - offsets[:-1]
        max_len = int(diffs.max()) if len(diffs) else 0
    vals = arr.values.to_numpy(zero_copy_only=False)
    return max_len, vals

def safe_int_vocab_and_dim(name: str, force_dim=None):
    arr = get_array(name)
    typ = arr.type
    if is_list_type(typ):
        max_len, vals = list_len_and_values(arr)
        dim = max(1, int(max_len if force_dim is None else force_dim))
    else:
        vals = arr.to_numpy(zero_copy_only=False)
        dim = 1 if force_dim is None else int(force_dim)

    vals = np.asarray(vals)
    if vals.size == 0:
        max_v = 0
    else:
        # object array 里可能有 None；统一过滤。
        clean = []
        for x in vals.reshape(-1):
            if x is None:
                continue
            try:
                if np.isnan(x):
                    continue
            except TypeError:
                pass
            try:
                clean.append(int(x))
            except Exception:
                pass
        max_v = max(clean) if clean else 0
    # <=0 会在 dataset.py 中被当 padding；vocab 只需要覆盖正 id。
    vocab = max(1, int(max_v) + 1)
    return [fid_from_name(name), vocab, dim]

def safe_dense_dim(name: str):
    arr = get_array(name)
    if not is_list_type(arr.type):
        return [fid_from_name(name), 1]
    max_len, _ = list_len_and_values(arr)
    return [fid_from_name(name), max(1, int(max_len))]

def collect(prefix: str, fids):
    cols = []
    for fid in fids:
        name = f"{prefix}_{fid}"
        if name in names:
            cols.append(name)
        else:
            raise KeyError(f"Missing column: {name}")
    return cols

# 按比赛说明的字段集合构造 schema；顺序按 README 中的列顺序。
user_int_fids = [1, 3, 4, 15] + list(range(48, 61)) + [62, 63, 64, 65, 66, 80, 82, 86, 89, 90, 91] + list(range(92, 110))
item_int_fids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 81, 83, 84, 85]
user_dense_fids = [61, 62, 63, 64, 65, 66, 87, 89, 90, 91]

schema = {
    "user_int": [safe_int_vocab_and_dim(c) for c in collect("user_int_feats", user_int_fids)],
    "item_int": [safe_int_vocab_and_dim(c) for c in collect("item_int_feats", item_int_fids)],
    "user_dense": [safe_dense_dim(c) for c in collect("user_dense_feats", user_dense_fids)],
    "seq": {}
}

seq_defs = {
    # 这些 ts_fid 根据 demo 中明显的秒级时间戳列推断；用于计算 time_bucket，不作为 side-info embedding。
    "seq_a": ("domain_a_seq", list(range(38, 47)), 39),
    "seq_b": ("domain_b_seq", list(range(67, 80)) + [88], 67),
    "seq_c": ("domain_c_seq", list(range(27, 38)) + [47], 27),
    "seq_d": ("domain_d_seq", list(range(17, 27)), 17),
}
for domain, (prefix, fids, ts_fid) in seq_defs.items():
    features = []
    for col in collect(prefix, fids):
        fid, vocab, _dim = safe_int_vocab_and_dim(col)
        features.append([fid, vocab])
    schema["seq"][domain] = {
        "prefix": prefix,
        "ts_fid": ts_fid,
        "features": features,
    }

schema_path = ready_dir / "schema.json"
with schema_path.open("w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2)

print(f"Wrote {out_parquet}")
print(f"Wrote {schema_path}")
print(f"Rows: {table.num_rows}; row_group_size={row_group_size}")
print("Schema summary:")
print("  user_int:", len(schema["user_int"]))
print("  item_int:", len(schema["item_int"]))
print("  user_dense:", len(schema["user_dense"]))
print("  seq domains:", list(schema["seq"].keys()))
PY

cat > "${PROJECT_DIR}/run_demo_local.sh" <<EOF
set -euo pipefail
cd "${PROJECT_DIR}"
source "${VENV_DIR}/bin/activate"

export TRAIN_DATA_PATH="${READY_DIR}"
export TRAIN_CKPT_PATH="${PROJECT_DIR}/outputs/demo_ckpt"
export TRAIN_LOG_PATH="${PROJECT_DIR}/outputs/demo_logs"
export TRAIN_TF_EVENTS_PATH="${PROJECT_DIR}/outputs/demo_tb"

mkdir -p "\${TRAIN_CKPT_PATH}" "\${TRAIN_LOG_PATH}" "\${TRAIN_TF_EVENTS_PATH}"

bash run.sh \\
  --batch_size 16 \\
  --num_epochs 1 \\
  --num_workers 0 \\
  --buffer_batches 1 \\
  --valid_ratio 0.25 \\
  --seq_max_lens seq_a:64,seq_b:64,seq_c:64,seq_d:64 \\
  --reinit_sparse_after_epoch 999
EOF
chmod +x "${PROJECT_DIR}/run_demo_local.sh"

echo
echo "Done."
echo
echo "下一步运行："
echo "  cd ${PROJECT_DIR}"
echo "  ./run_demo_local.sh"
echo
echo "数据目录：${READY_DIR}"
echo "虚拟环境：${VENV_DIR}"
