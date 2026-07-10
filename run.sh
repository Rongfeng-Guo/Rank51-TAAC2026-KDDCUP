#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- 当前启用配置：RankMixer NS 分词器（无需 ns_groups.json） ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- 备选配置：基于 ns_groups.json 的 GroupNSTokenizer ----
# 该配置使用 ns_groups.json 的分组结果（7 个 user 组 + 4 个 item 组）。
# 当 d_model=64 且 num_ns=12（7 user_int + 1 user_dense + 4 item_int）时，
# 只有 num_queries=1 能满足 d_model % T == 0（T = num_queries*4 + num_ns）。
# 如需切换，请注释上方命令并取消下方命令注释。
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
