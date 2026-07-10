from __future__ import annotations
"""TAAC 数据预处理主入口。

本脚本负责将原始 parquet 数据转换为训练可直接消费的产物，包含：
1. 数据摘要与完整性校验
2. 清洗与序列张量化
3. 训练/验证切分
4. 报告与数据文件落盘

设计目标：命令行一键执行、参数可覆盖、输出结构稳定。

接口契约（输入）：
- 原始数据：demo_1000.parquet（或 --dataset-path 指定路径）
- 关键字段：user_id/item_id/label_type/label_time/timestamp
- 可选配置：sequence-length-preset、sequence-length-overrides

接口契约（输出）：
- 数据文件：cleaned.parquet、prepared.parquet、train.parquet、valid.parquet
- 报告文件：summary/audit/cleaning/sequence_tensorization/model_input_spec/split_summary
- 目录约定：<output-dir>/reports/*.json
"""

import argparse
import json
from pathlib import Path

from taac_dataset import (
    audit_dataset,
    clean_dataset,
    build_model_input_spec,
    load_dataset,
    get_sequence_length_presets,
    prepare_training_frame,
    split_by_time,
    summarize_sequence_tensorization,
    summarize_dataset,
    summarize_splits,
    validate_summary,
    write_dataframe,
    write_json_file,
    write_splits,
)


def _parse_sequence_length_overrides(raw_value: str | None) -> dict[str, int] | None:
    """解析序列长度覆盖配置。

    支持三种输入：
    - JSON 字符串
    - PowerShell 常见的 key:value / key=value 对
    - JSON 文件路径

    Args:
        raw_value: 原始命令行参数值。

    Returns:
        dict[str, int] | None: 解析后的覆盖字典；若未提供则返回 None。

    Raises:
        ValueError: 当输入格式非法或无法解析为对象时抛出。
    """
    if raw_value is None:
        return None

    candidate = raw_value.strip()
    if not candidate:
        return None

    if candidate.startswith("'") and candidate.endswith("'"):
        candidate = candidate[1:-1]

    if candidate.startswith("{"):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            normalized_pairs = candidate.strip("{} ")
            if not normalized_pairs:
                return {}

            payload = {}
            for item in normalized_pairs.split(","):
                pair = item.strip()
                if not pair:
                    continue
                if ":" in pair:
                    key, value = pair.split(":", 1)
                elif "=" in pair:
                    key, value = pair.split("=", 1)
                else:
                    raise ValueError(
                        "sequence-length-overrides entries must use key:value or key=value syntax"
                    )
                payload[key.strip().strip('"\'')] = int(value.strip())
    else:
        config_path = Path(candidate)
        if not config_path.exists():
            raise ValueError(
                "sequence-length-overrides must be a JSON object string or an existing JSON file path"
            )
        payload = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("sequence-length-overrides must resolve to a JSON object")

    return {str(key): value for key, value in payload.items()}


def _resolve_sequence_length_configuration(
    preset_name: str | None,
    manual_overrides: dict[str, int] | None,
) -> dict[str, int] | None:
    """将预设与手工覆盖合并为最终序列长度配置。

    Args:
        preset_name: 预设名称。
        manual_overrides: 手工覆盖项。

    Returns:
        dict[str, int] | None: 合并后的配置；若无配置则返回 None。

    Raises:
        ValueError: 当预设名称不存在时抛出。
    """
    presets = get_sequence_length_presets()
    if preset_name is not None and preset_name not in presets:
        raise ValueError(
            f"Unknown sequence-length-preset: {preset_name}. "
            f"Available presets: {', '.join(sorted(presets))}"
        )

    merged_overrides: dict[str, int] = {}
    if preset_name is not None:
        merged_overrides.update(presets[preset_name])
    if manual_overrides:
        merged_overrides.update(manual_overrides)
    return merged_overrides or None


def parse_args() -> argparse.Namespace:
    """定义并解析 CLI 参数。"""
    available_presets = sorted(get_sequence_length_presets())
    parser = argparse.ArgumentParser(
        description="Analyze, clean, prepare, and split the TAAC parquet dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("demo_1000.parquet"),
        help="Path to the source parquet dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "prepared",
        help="Directory used to store reports and processed parquet files.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Fraction of rows reserved for validation after sorting by time.",
    )
    parser.add_argument(
        "--time-column",
        default="label_time",
        help="Timestamp column used to order rows before splitting.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=50,
        help="Max length kept for domain sequence features in the prepared frame.",
    )
    parser.add_argument(
        "--sequence-length-preset",
        choices=available_presets,
        help="Named preset used to initialize per-domain sequence target lengths before manual overrides.",
    )
    parser.add_argument(
        "--sequence-length-overrides",
        help=(
            "JSON object or JSON file path used to override sequence target lengths by "
            "column name, prefix like domain_a_seq, or domain name like domain_a."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary and audit information without writing processed outputs.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Write cleaned and prepared outputs but skip train/valid split generation.",
    )
    return parser.parse_args()


def _write_reports(
    output_dir: Path,
    summary_payload: dict,
    feature_groups_payload: dict,
    audit_payload: dict,
    cleaning_payload: dict,
    sequence_payload: dict,
    model_input_spec_payload: dict,
    split_payload: dict | None,
) -> dict[str, Path]:
    """将各类摘要信息写入 reports 目录并返回路径映射。

    Args:
        output_dir: 产物根目录。
        summary_payload: 数据摘要。
        feature_groups_payload: 特征分组摘要。
        audit_payload: 审计结果。
        cleaning_payload: 清洗结果。
        sequence_payload: 序列张量化摘要。
        model_input_spec_payload: 输入契约规格。
        split_payload: 切分摘要（可选）。

    Returns:
        dict[str, Path]: 报告名到文件路径的映射。
    """
    reports_dir = output_dir / "reports"
    report_paths = {
        "summary": write_json_file(summary_payload, reports_dir / "summary.json"),
        "feature_groups": write_json_file(
            feature_groups_payload,
            reports_dir / "feature_groups.json",
        ),
        "audit": write_json_file(audit_payload, reports_dir / "audit.json"),
        "cleaning": write_json_file(cleaning_payload, reports_dir / "cleaning.json"),
        "sequence_tensorization": write_json_file(
            sequence_payload,
            reports_dir / "sequence_tensorization.json",
        ),
        "model_input_spec": write_json_file(
            model_input_spec_payload,
            reports_dir / "model_input_spec.json",
        ),
    }
    if split_payload is not None:
        report_paths["splits"] = write_json_file(
            split_payload,
            reports_dir / "split_summary.json",
        )
    return report_paths


def _print_report(title: str, payload: dict) -> None:
    """标准化打印单个报告块，便于日志检索。"""
    print(f"[{title}]")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _print_overview(
    summary_payload: dict,
    audit_payload: dict,
    sequence_payload: dict,
) -> None:
    """打印高层运行概览（行列数、审计信息、序列配置）。"""
    print(
        "Loaded dataset: "
        f"rows={summary_payload['rows']}, columns={summary_payload['columns']}"
    )
    if sequence_payload.get("sequence_length_preset"):
        print(f"Sequence preset: {sequence_payload['sequence_length_preset']}")
    print(
        "Audit overview: "
        f"unique_users={audit_payload['unique_users']}, "
        f"unique_items={audit_payload['unique_items']}, "
        f"duplicate_rows={audit_payload['duplicate_rows']}"
    )
    print(f"Label distribution: {audit_payload['label_distribution']}")
    first_sequence = next(iter(sequence_payload["sequence_columns"].values()), None)
    if first_sequence is not None:
        print(
            "Sequence tensorization: "
            f"columns={sequence_payload['sequence_group_count']}, "
            f"target_length={first_sequence['target_length']}, "
            f"overrides={len(sequence_payload['sequence_length_overrides'])}"
        )


def main() -> None:
    """执行完整预处理流程。

    当启用 --summary-only 时，只进行读取、审计和摘要输出；
    否则会继续写出 cleaned/prepared/split 数据及相关报告。

    Raises:
        ValueError: 当参数配置非法（如序列覆盖或切分参数）时由下游函数抛出。
    """
    args = parse_args()
    manual_sequence_length_overrides = _parse_sequence_length_overrides(
        args.sequence_length_overrides
    )
    sequence_length_overrides = _resolve_sequence_length_configuration(
        args.sequence_length_preset,
        manual_sequence_length_overrides,
    )
    df = load_dataset(args.dataset_path)

    summary = summarize_dataset(df)
    validate_summary(summary)

    audit = audit_dataset(df)
    summary_payload = summary.to_dict()
    audit_payload = audit.to_dict()
    feature_groups_payload = summary.feature_groups.as_dict()
    sequence_payload = summarize_sequence_tensorization(
        df,
        max_sequence_length=args.max_sequence_length,
        sequence_length_overrides=sequence_length_overrides,
        sequence_length_preset=args.sequence_length_preset,
    )

    if args.summary_only:
        cleaned_df, _ = clean_dataset(df, time_column=args.time_column)
        prepared_df = prepare_training_frame(
            cleaned_df,
            max_sequence_length=args.max_sequence_length,
            sequence_length_overrides=sequence_length_overrides,
        )
        model_input_spec_payload = build_model_input_spec(
            cleaned_df,
            prepared_df,
            sequence_payload,
            sequence_length_preset=args.sequence_length_preset,
        )
        _print_report("summary", summary_payload)
        _print_report("audit", audit_payload)
        _print_report("sequence_tensorization", sequence_payload)
        _print_report("model_input_spec", model_input_spec_payload)
        return

    _print_overview(summary_payload, audit_payload, sequence_payload)

    cleaned_df, cleaning_report = clean_dataset(df, time_column=args.time_column)
    prepared_df = prepare_training_frame(
        cleaned_df,
        max_sequence_length=args.max_sequence_length,
        sequence_length_overrides=sequence_length_overrides,
    )
    model_input_spec_payload = build_model_input_spec(
        cleaned_df,
        prepared_df,
        sequence_payload,
        sequence_length_preset=args.sequence_length_preset,
    )

    cleaned_path = write_dataframe(cleaned_df, args.output_dir / "cleaned.parquet")
    prepared_path = write_dataframe(prepared_df, args.output_dir / "prepared.parquet")

    split_payload: dict | None = None
    train_path: Path | None = None
    valid_path: Path | None = None
    if not args.skip_split:
        train_df, valid_df = split_by_time(
            prepared_df,
            valid_ratio=args.valid_ratio,
            time_column=args.time_column,
        )
        train_path, valid_path = write_splits(train_df, valid_df, args.output_dir)
        split_payload = summarize_splits(
            train_df,
            valid_df,
            time_column=args.time_column,
        )

    report_paths = _write_reports(
        args.output_dir,
        summary_payload=summary_payload,
        feature_groups_payload=feature_groups_payload,
        audit_payload=audit_payload,
        cleaning_payload=cleaning_report.to_dict(),
        sequence_payload=sequence_payload,
        model_input_spec_payload=model_input_spec_payload,
        split_payload=split_payload,
    )

    print(f"Wrote cleaned dataset to: {cleaned_path}")
    print(f"Wrote prepared dataset to: {prepared_path}")
    if train_path is not None and valid_path is not None:
        print(f"Wrote train split to: {train_path}")
        print(f"Wrote valid split to: {valid_path}")
    for name, path in report_paths.items():
        print(f"Wrote {name} report to: {path}")


if __name__ == "__main__":
    main()
