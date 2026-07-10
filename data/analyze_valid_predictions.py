from __future__ import annotations
"""验证集预测误差分析脚本。

基于 valid_predictions 与 prepared valid 对齐后，输出：
1. 按标签的错分统计
2. 按时间桶的错分统计
3. 按活跃度桶的错分统计
4. 高置信错分样本清单

接口契约（输入）：
- valid_predictions.parquet：需包含 JOIN_KEYS + 预测概率列
- prepared valid.parquet：需包含 JOIN_KEYS + 各域长度代表列

接口契约（输出）：
- label/time/activity 三类聚合 CSV
- high_confidence_errors.csv（高置信错分样本）
- summary.json（关键统计与输出索引）
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


JOIN_KEYS: tuple[str, ...] = (
    "user_id",
    "item_id",
    "label_type",
    "label_time",
    "timestamp",
)

DOMAIN_LENGTH_REPRESENTATIVES: dict[str, str] = {
    "domain_a": "domain_a_seq_38_len",
    "domain_b": "domain_b_seq_67_len",
    "domain_c": "domain_c_seq_27_len",
    "domain_d": "domain_d_seq_17_len",
}


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze validation prediction errors by label, time bucket, and user activity."
        )
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("outputs") / "training_runs" / "latest" / "valid_predictions.parquet",
        help="Path to valid_predictions.parquet produced by the training script.",
    )
    parser.add_argument(
        "--valid-prepared-path",
        type=Path,
        default=Path("outputs") / "prepared_balanced_preset" / "valid.parquet",
        help="Path to the prepared validation split used to derive activity signals.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "training_runs" / "latest" / "error_analysis",
        help="Directory used to store grouped error summaries and sample tables.",
    )
    parser.add_argument(
        "--time-buckets",
        type=int,
        default=4,
        help="Number of quantile buckets used for label_time based error analysis.",
    )
    parser.add_argument(
        "--activity-buckets",
        type=int,
        default=4,
        help="Number of quantile buckets used for history-length based activity analysis.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=25,
        help="Number of highest-confidence wrong samples exported to the sample table.",
    )
    return parser.parse_args()


def _validate_columns(frame: pd.DataFrame, required_columns: set[str], name: str) -> None:
    """检查输入表是否包含分析所需字段。

    Args:
        frame: 待校验数据表。
        required_columns: 必需字段集合。
        name: 表名标识，用于错误提示。

    Raises:
        ValueError: 当缺少必需字段时抛出。
    """
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {name}: {', '.join(missing)}")


def _with_bucket_labels(
    values: pd.Series,
    bucket_count: int,
    prefix: str,
) -> pd.Series:
    """将连续变量按分位数分桶并生成可读桶标签。

    Args:
        values: 连续变量序列。
        bucket_count: 桶数量。
        prefix: 桶名前缀。

    Returns:
        pd.Series: 分桶标签序列（如 time_1、time_2）。

    Raises:
        ValueError: 当 bucket_count 小于 1 时抛出。
    """
    if bucket_count < 1:
        raise ValueError(f"bucket_count must be at least 1, got {bucket_count}")
    if values.nunique(dropna=False) <= 1:
        return pd.Series([f"{prefix}_1"] * len(values), index=values.index, dtype="object")

    ranked = values.rank(method="first")
    bucket_codes = pd.qcut(ranked, q=min(bucket_count, len(values)), labels=False, duplicates="drop")
    return bucket_codes.map(lambda code: f"{prefix}_{int(code) + 1}").astype("object")


def _build_label_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """构建按标签聚合的错分统计。"""
    summary = (
        frame.groupby("label_type", dropna=False)
        .agg(
            rows=("label_type", "size"),
            error_count=("is_error", "sum"),
            avg_confidence=("predicted_confidence", "mean"),
            avg_true_label_prob=("true_label_probability", "mean"),
            avg_margin=("confidence_margin", "mean"),
        )
        .reset_index()
        .sort_values("label_type")
    )
    summary["error_rate"] = summary["error_count"] / summary["rows"]
    return summary


def _build_time_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """构建按时间桶聚合的错分统计。"""
    summary = (
        frame.groupby("time_bucket", dropna=False)
        .agg(
            rows=("time_bucket", "size"),
            error_count=("is_error", "sum"),
            avg_confidence=("predicted_confidence", "mean"),
            avg_margin=("confidence_margin", "mean"),
            time_min=("label_time", "min"),
            time_max=("label_time", "max"),
        )
        .reset_index()
        .sort_values("time_bucket")
    )
    summary["error_rate"] = summary["error_count"] / summary["rows"]
    return summary


def _build_activity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """构建按活跃度桶聚合的错分统计。"""
    summary = (
        frame.groupby("activity_bucket", dropna=False)
        .agg(
            rows=("activity_bucket", "size"),
            error_count=("is_error", "sum"),
            avg_confidence=("predicted_confidence", "mean"),
            avg_margin=("confidence_margin", "mean"),
            avg_total_history_length=("total_history_length", "mean"),
            activity_min=("total_history_length", "min"),
            activity_max=("total_history_length", "max"),
        )
        .reset_index()
        .sort_values("activity_bucket")
    )
    summary["error_rate"] = summary["error_count"] / summary["rows"]
    return summary


def _build_high_confidence_errors(frame: pd.DataFrame, top_errors: int) -> pd.DataFrame:
    """提取高置信且预测错误的样本。"""
    error_frame = frame.loc[frame["is_error"]].copy()
    if error_frame.empty:
        return error_frame
    return error_frame.sort_values(
        ["predicted_confidence", "confidence_margin"],
        ascending=[False, False],
    ).head(top_errors)


def _json_default(value: Any) -> Any:
    """将 numpy 标量转换为可 JSON 序列化对象。"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    return value


def main() -> None:
    """执行完整误差分析并写出 CSV/JSON 结果。

    Raises:
        ValueError: 当输入文件缺少关键字段时由校验函数抛出。
    """
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(args.predictions_path)
    prepared_valid = pd.read_parquet(args.valid_prepared_path)

    _validate_columns(
        predictions,
        set(JOIN_KEYS) | {"predicted_label", "prob_label_1", "prob_label_2"},
        "predictions",
    )
    _validate_columns(
        prepared_valid,
        set(JOIN_KEYS) | set(DOMAIN_LENGTH_REPRESENTATIVES.values()),
        "prepared_valid",
    )

    activity_frame = prepared_valid[list(JOIN_KEYS) + list(DOMAIN_LENGTH_REPRESENTATIVES.values())].copy()
    merged = predictions.merge(activity_frame, on=list(JOIN_KEYS), how="left", validate="one_to_one")

    for domain_name, column_name in DOMAIN_LENGTH_REPRESENTATIVES.items():
        merged[f"{domain_name}_history_length"] = merged[column_name].fillna(0).astype(int)

    merged["total_history_length"] = merged[
        [f"{domain_name}_history_length" for domain_name in DOMAIN_LENGTH_REPRESENTATIVES]
    ].sum(axis=1)
    merged["is_error"] = merged["predicted_label"] != merged["label_type"]
    merged["predicted_confidence"] = merged.apply(
        lambda row: row[f"prob_label_{int(row['predicted_label'])}"],
        axis=1,
    )
    merged["true_label_probability"] = merged.apply(
        lambda row: row[f"prob_label_{int(row['label_type'])}"],
        axis=1,
    )
    merged["confidence_margin"] = (
        merged[["prob_label_1", "prob_label_2"]].max(axis=1)
        - merged[["prob_label_1", "prob_label_2"]].min(axis=1)
    )
    merged["time_bucket"] = _with_bucket_labels(
        merged["label_time"],
        bucket_count=args.time_buckets,
        prefix="time",
    )
    merged["activity_bucket"] = _with_bucket_labels(
        merged["total_history_length"],
        bucket_count=args.activity_buckets,
        prefix="activity",
    )

    label_summary = _build_label_summary(merged)
    time_summary = _build_time_summary(merged)
    activity_summary = _build_activity_summary(merged)
    high_confidence_errors = _build_high_confidence_errors(merged, args.top_errors)

    label_summary.to_csv(args.output_dir / "label_error_summary.csv", index=False)
    time_summary.to_csv(args.output_dir / "time_bucket_error_summary.csv", index=False)
    activity_summary.to_csv(args.output_dir / "activity_bucket_error_summary.csv", index=False)
    high_confidence_errors.to_csv(args.output_dir / "high_confidence_errors.csv", index=False)

    summary_payload = {
        "predictions_path": str(args.predictions_path),
        "valid_prepared_path": str(args.valid_prepared_path),
        "rows": int(len(merged)),
        "error_rows": int(merged["is_error"].sum()),
        "error_rate": float(merged["is_error"].mean()),
        "time_bucket_count": int(time_summary["time_bucket"].nunique()),
        "activity_bucket_count": int(activity_summary["activity_bucket"].nunique()),
        "activity_definition": {
            "type": "sum_of_domain_history_lengths",
            "representative_columns": DOMAIN_LENGTH_REPRESENTATIVES,
        },
        "outputs": {
            "label_summary": str(args.output_dir / "label_error_summary.csv"),
            "time_summary": str(args.output_dir / "time_bucket_error_summary.csv"),
            "activity_summary": str(args.output_dir / "activity_bucket_error_summary.csv"),
            "high_confidence_errors": str(args.output_dir / "high_confidence_errors.csv"),
        },
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()