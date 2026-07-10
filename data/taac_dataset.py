from __future__ import annotations
"""TAAC 数据处理核心库。

本模块提供从原始数据到训练输入的完整底层能力：
1. 特征分组推断与数据审计
2. 清洗、去重、序列归一化
3. 序列张量化（tensor/mask/length）
4. 高缺失 user_int 特征增强（缺失指示列）
5. 时序切分与报告落盘

接口契约（输入）：
- 原始 DataFrame：需包含 REQUIRED_ID_LABEL_COLUMNS
- 序列列命名：按 FEATURE_GROUP_PREFIXES 前缀约定推断分组
- 可选配置：max_sequence_length、sequence_length_overrides

接口契约（输出）：
- 清洗结果：cleaned DataFrame + CleaningReport
- 训练结果：prepared DataFrame（含 *_len/*_tensor/*_mask 及 *_is_missing）
- 输入规范：model_input_spec（identifiers/feature_layout/prepared_layout）
- 切分结果：train/valid DataFrame 与 split 摘要
"""

import json
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REQUIRED_ID_LABEL_COLUMNS: tuple[str, ...] = (
    "user_id",
    "item_id",
    "label_type",
    "label_time",
    "timestamp",
)

FEATURE_GROUP_PREFIXES: dict[str, str] = {
    "user_int": "user_int_feats_",
    "user_dense": "user_dense_feats_",
    "item_int": "item_int_feats_",
    "domain_a_seq": "domain_a_seq_",
    "domain_b_seq": "domain_b_seq_",
    "domain_c_seq": "domain_c_seq_",
    "domain_d_seq": "domain_d_seq_",
}

EXPECTED_GROUP_COUNTS: dict[str, int] = {
    "id_and_label": 5,
    "user_int": 46,
    "user_dense": 10,
    "item_int": 14,
    "domain_a_seq": 9,
    "domain_b_seq": 14,
    "domain_c_seq": 12,
    "domain_d_seq": 10,
}

SEQUENCE_LENGTH_PRESETS: dict[str, dict[str, int]] = {
    "compact": {
        "domain_a_seq": 128,
        "domain_b_seq": 128,
        "domain_c_seq": 256,
        "domain_d_seq": 256,
    },
    "balanced": {
        "domain_a_seq": 256,
        "domain_b_seq": 256,
        "domain_c_seq": 512,
        "domain_d_seq": 1024,
    },
    "long_context": {
        "domain_a_seq": 512,
        "domain_b_seq": 512,
        "domain_c_seq": 1024,
        "domain_d_seq": 1536,
    },
}

SEQUENCE_LENGTH_PRESET_DESCRIPTIONS: dict[str, str] = {
    "compact": "Lower-memory preset for quick experiments with shorter history windows.",
    "balanced": "Recommended preset that relaxes truncation for all four behavior domains.",
    "long_context": "Long-history preset for offline experiments that can afford larger sequence windows.",
}

HIGH_MISSING_USER_INT_FID_RANGE: tuple[int, int] = (83, 103)
HIGH_MISSING_USER_INT_NULL_RATE_THRESHOLD: float = 0.5
MISSING_INDICATOR_SUFFIX: str = "_is_missing"


@dataclass(frozen=True)
class FeatureGroups:
    id_and_label: tuple[str, ...]
    user_int: tuple[str, ...]
    user_dense: tuple[str, ...]
    item_int: tuple[str, ...]
    domain_a_seq: tuple[str, ...]
    domain_b_seq: tuple[str, ...]
    domain_c_seq: tuple[str, ...]
    domain_d_seq: tuple[str, ...]

    @property
    def domain_sequence(self) -> tuple[str, ...]:
        return (
            self.domain_a_seq
            + self.domain_b_seq
            + self.domain_c_seq
            + self.domain_d_seq
        )

    @property
    def all_features(self) -> tuple[str, ...]:
        return (
            self.user_int
            + self.user_dense
            + self.item_int
            + self.domain_sequence
        )

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "id_and_label": list(self.id_and_label),
            "user_int": list(self.user_int),
            "user_dense": list(self.user_dense),
            "item_int": list(self.item_int),
            "domain_a_seq": list(self.domain_a_seq),
            "domain_b_seq": list(self.domain_b_seq),
            "domain_c_seq": list(self.domain_c_seq),
            "domain_d_seq": list(self.domain_d_seq),
        }


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    feature_groups: FeatureGroups
    null_counts: dict[str, int]
    count_matches: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "feature_groups": {
                name: len(columns)
                for name, columns in self.feature_groups.as_dict().items()
            },
            "null_counts": self.null_counts,
            "count_matches": self.count_matches,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class DatasetAudit:
    unique_users: int
    unique_items: int
    duplicate_rows: int
    label_distribution: dict[str, int]
    time_ranges: dict[str, dict[str, int | float | None]]
    null_rates: dict[str, float]
    constant_columns: tuple[str, ...]
    scalar_feature_ranges: dict[str, dict[str, int | float | None]]
    list_length_stats: dict[str, dict[str, int | float | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unique_users": self.unique_users,
            "unique_items": self.unique_items,
            "duplicate_rows": self.duplicate_rows,
            "label_distribution": self.label_distribution,
            "time_ranges": self.time_ranges,
            "null_rates": self.null_rates,
            "constant_columns": list(self.constant_columns),
            "scalar_feature_ranges": self.scalar_feature_ranges,
            "list_length_stats": self.list_length_stats,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    output_rows: int
    dropped_null_rows: int
    dropped_duplicate_rows: int
    normalized_list_columns: tuple[str, ...]
    sorted_by: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "dropped_null_rows": self.dropped_null_rows,
            "dropped_duplicate_rows": self.dropped_duplicate_rows,
            "normalized_list_columns": list(self.normalized_list_columns),
            "sorted_by": self.sorted_by,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _to_python_scalar(value: Any) -> Any:
    """将 numpy/pandas 标量转换为 Python 原生类型。"""
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _is_missing_value(value: Any) -> bool:
    """判断值是否可视为缺失。"""
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, bool) else False


def _is_sequence_value(value: Any) -> bool:
    """判断值是否应按序列处理。"""
    if _is_missing_value(value):
        return False
    if isinstance(value, (str, bytes, bytearray, dict)):
        return False
    if isinstance(value, SequenceABC):
        return True
    return hasattr(value, "tolist") and not pd.api.types.is_scalar(value)


def _normalize_sequence_value(value: Any) -> list[Any]:
    """将任意输入规范化为 Python list 序列。"""
    if _is_missing_value(value):
        return []
    if hasattr(value, "tolist") and not isinstance(value, list):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [_to_python_scalar(item) for item in value]
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_python_scalar(item) for item in list(value)]
    return [_to_python_scalar(value)]


def _is_constant_column(series: pd.Series) -> bool:
    """判断列是否为常量列。"""
    seen: set[Any] = set()
    for value in series:
        normalized = _hashable_value(value)
        seen.add(normalized)
        if len(seen) > 1:
            return False
    return True


def _hashable_value(value: Any) -> Any:
    """将值映射为可哈希表示，便于集合比较。"""
    if _is_sequence_value(value):
        return tuple(_normalize_sequence_value(value))
    return _to_python_scalar(value)


def _describe_numeric_series(series: pd.Series) -> dict[str, int | float | None]:
    """返回数值列的最小值与最大值。"""
    if series.empty:
        return {"min": None, "max": None}
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        return {"min": None, "max": None}
    return {
        "min": _to_python_scalar(numeric.min()),
        "max": _to_python_scalar(numeric.max()),
    }


def _describe_lengths(series: pd.Series) -> dict[str, int | float | None]:
    """返回序列长度分布统计。"""
    if series.empty:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "empty_ratio": 0.0,
        }

    lengths = series.map(lambda value: len(_normalize_sequence_value(value)))
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 4),
        "p50": round(float(lengths.quantile(0.50)), 4),
        "p95": round(float(lengths.quantile(0.95)), 4),
        "p99": round(float(lengths.quantile(0.99)), 4),
        "empty_ratio": round(float((lengths == 0).mean()), 6),
    }


def _resolve_target_length(lengths: pd.Series, max_sequence_length: int | None) -> int:
    """解析目标序列长度：优先使用配置值，否则采用观测最大值。"""
    if max_sequence_length is not None:
        if max_sequence_length < 1:
            raise ValueError(
                f"max_sequence_length must be at least 1, got {max_sequence_length}"
            )
        return max_sequence_length

    if lengths.empty:
        return 1
    return max(int(lengths.max()), 1)


def _normalize_sequence_length_overrides(
    sequence_length_overrides: Mapping[str, int] | None,
) -> dict[str, int]:
    """校验并规范化序列长度覆盖配置。"""
    if sequence_length_overrides is None:
        return {}

    normalized_overrides: dict[str, int] = {}
    for key, value in sequence_length_overrides.items():
        if not isinstance(key, str) or not key:
            raise ValueError("sequence_length_overrides keys must be non-empty strings")
        if not isinstance(value, int):
            raise ValueError(
                "sequence_length_overrides values must be integers, "
                f"got {type(value).__name__} for {key}"
            )
        if value < 1:
            raise ValueError(
                "sequence_length_overrides values must be at least 1, "
                f"got {value} for {key}"
            )
        normalized_overrides[key] = value

    return normalized_overrides


def get_sequence_length_presets() -> dict[str, dict[str, int]]:
    """返回内置序列长度预设。"""
    return {
        preset_name: dict(overrides)
        for preset_name, overrides in SEQUENCE_LENGTH_PRESETS.items()
    }


def get_sequence_length_preset_descriptions() -> dict[str, str]:
    """返回序列长度预设说明。"""
    return dict(SEQUENCE_LENGTH_PRESET_DESCRIPTIONS)


def _resolve_sequence_target_length(
    column_name: str,
    lengths: pd.Series,
    max_sequence_length: int | None,
    sequence_length_overrides: Mapping[str, int] | None = None,
) -> tuple[int, str]:
    """按列名/前缀/域名三级优先级解析序列目标长度。"""
    normalized_overrides = _normalize_sequence_length_overrides(sequence_length_overrides)
    sequence_prefix = column_name.rsplit("_", 1)[0]
    domain_prefix = column_name.split("_seq_")[0] if "_seq_" in column_name else sequence_prefix

    for override_key in (column_name, sequence_prefix, domain_prefix):
        if override_key in normalized_overrides:
            return normalized_overrides[override_key], override_key

    return _resolve_target_length(lengths, max_sequence_length), "default"


def _infer_sequence_element_dtype(series: pd.Series) -> str:
    """推断序列元素类型（int/float/bool/...）。"""
    for value in series:
        normalized = _normalize_sequence_value(value)
        for item in normalized:
            if item is None:
                continue
            if isinstance(item, bool):
                return "bool"
            if isinstance(item, float):
                return "float"
            if isinstance(item, int):
                return "int"
            return type(item).__name__
    return "unknown"


def _padding_value_for_dtype(element_dtype: str) -> int | float:
    """根据元素类型返回默认 padding 值。"""
    if element_dtype == "float":
        return 0.0
    return 0


def _infer_sequence_pad_value(values: Sequence[Any]) -> int | float:
    """根据序列值类型推断 padding 值类型。"""
    for value in values:
        if isinstance(value, float):
            return 0.0
    return 0


def _tensorize_sequence(
    values: Sequence[Any],
    target_length: int,
) -> tuple[list[Any], list[int], int]:
    """将序列截断/补齐为定长并生成 mask 与有效长度。"""
    normalized_values = list(values)
    truncated_values = normalized_values[-target_length:]
    effective_length = len(truncated_values)
    pad_value = _infer_sequence_pad_value(truncated_values)
    pad_count = max(target_length - effective_length, 0)

    tensor_values = truncated_values + [pad_value] * pad_count
    mask_values = [1] * effective_length + [0] * pad_count
    return tensor_values, mask_values, effective_length


def _duplicate_mask(
    df: pd.DataFrame,
    key_columns: Sequence[str] = REQUIRED_ID_LABEL_COLUMNS,
) -> pd.Series:
    """基于关键字段计算重复行掩码。"""
    available_keys = [column for column in key_columns if column in df.columns]
    if not available_keys:
        raise ValueError("No duplicate-key columns are available in the dataframe")
    return df.duplicated(subset=available_keys)


def _maybe_time_ranges(df: pd.DataFrame) -> dict[str, dict[str, int | float | None]]:
    """提取可用时间列的取值范围。"""
    time_ranges: dict[str, dict[str, int | float | None]] = {}
    for column in ("label_time", "timestamp"):
        if column in df.columns:
            time_ranges[column] = _describe_numeric_series(df[column])
    return time_ranges


def detect_list_columns(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """识别候选列中应按 list 处理的列。"""
    selected_columns = columns if columns is not None else df.columns.tolist()
    list_columns: list[str] = []
    for column in selected_columns:
        sample = df[column].dropna().head(10)
        if any(_is_sequence_value(value) for value in sample):
            list_columns.append(column)
    return tuple(list_columns)


def infer_feature_groups(columns: Sequence[str]) -> FeatureGroups:
    """根据列名前缀推断特征分组。"""
    column_set = set(columns)
    missing = [column for column in REQUIRED_ID_LABEL_COLUMNS if column not in column_set]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_text}")

    def group_by_prefix(prefix: str) -> tuple[str, ...]:
        return tuple(sorted(column for column in columns if column.startswith(prefix)))

    return FeatureGroups(
        id_and_label=tuple(column for column in columns if column in REQUIRED_ID_LABEL_COLUMNS),
        user_int=group_by_prefix(FEATURE_GROUP_PREFIXES["user_int"]),
        user_dense=group_by_prefix(FEATURE_GROUP_PREFIXES["user_dense"]),
        item_int=group_by_prefix(FEATURE_GROUP_PREFIXES["item_int"]),
        domain_a_seq=group_by_prefix(FEATURE_GROUP_PREFIXES["domain_a_seq"]),
        domain_b_seq=group_by_prefix(FEATURE_GROUP_PREFIXES["domain_b_seq"]),
        domain_c_seq=group_by_prefix(FEATURE_GROUP_PREFIXES["domain_c_seq"]),
        domain_d_seq=group_by_prefix(FEATURE_GROUP_PREFIXES["domain_d_seq"]),
    )


def _parse_feature_suffix(column_name: str, prefix: str) -> int | None:
    """解析特征名后缀编号（例如 user_int_feats_99 -> 99）。"""
    if not column_name.startswith(prefix):
        return None
    suffix = column_name[len(prefix):]
    return int(suffix) if suffix.isdigit() else None


def _select_sparse_user_int_scalar_columns(
    df: pd.DataFrame,
    user_int_columns: Sequence[str],
    list_columns: Sequence[str],
) -> list[str]:
    """选择高缺失 user_int 标量列，用于缺失指示增强。"""
    selected: list[str] = []
    list_column_set = set(list_columns)
    prefix = FEATURE_GROUP_PREFIXES["user_int"]
    lower, upper = HIGH_MISSING_USER_INT_FID_RANGE

    for column in user_int_columns:
        if column in list_column_set:
            continue

        suffix = _parse_feature_suffix(column, prefix)
        in_target_range = suffix is not None and lower <= suffix <= upper
        null_rate = float(df[column].isna().mean())

        if in_target_range or null_rate >= HIGH_MISSING_USER_INT_NULL_RATE_THRESHOLD:
            selected.append(column)

    return selected


def _apply_sparse_user_int_missing_logic(
    prepared: pd.DataFrame,
    feature_groups: FeatureGroups,
    list_columns: Sequence[str],
) -> pd.DataFrame:
    """对高缺失 user_int 列生成缺失指示列并填充值。"""
    sparse_columns = _select_sparse_user_int_scalar_columns(
        prepared,
        feature_groups.user_int,
        list_columns,
    )
    for column in sparse_columns:
        indicator_column = f"{column}{MISSING_INDICATOR_SUFFIX}"
        prepared[indicator_column] = prepared[column].isna().astype("int8")
        prepared[column] = prepared[column].fillna(0)
    return prepared


def _collect_missing_indicator_columns(
    prepared_df: pd.DataFrame,
    base_scalar_columns: Sequence[str],
) -> list[str]:
    """收集 prepared 数据中可加入 scalar 输入的缺失指示列。"""
    base_scalar_set = set(base_scalar_columns)
    indicators: list[str] = []
    for column in prepared_df.columns:
        if not column.endswith(MISSING_INDICATOR_SUFFIX):
            continue
        source_column = column[: -len(MISSING_INDICATOR_SUFFIX)]
        if source_column in base_scalar_set:
            indicators.append(column)
    return sorted(indicators)


def load_dataset(dataset_path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """读取 parquet 数据集。"""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None)


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    """生成数据集摘要（行列数、分组计数、关键列缺失）。"""
    feature_groups = infer_feature_groups(df.columns.tolist())
    null_counts = {
        column: int(df[column].isna().sum())
        for column in REQUIRED_ID_LABEL_COLUMNS
    }

    observed_counts = {
        name: len(columns)
        for name, columns in feature_groups.as_dict().items()
    }
    count_matches = {
        name: observed_counts[name] == expected_count
        for name, expected_count in EXPECTED_GROUP_COUNTS.items()
    }

    return DatasetSummary(
        rows=len(df),
        columns=df.shape[1],
        feature_groups=feature_groups,
        null_counts=null_counts,
        count_matches=count_matches,
    )


def audit_dataset(df: pd.DataFrame) -> DatasetAudit:
    """执行数据审计（唯一值、重复、缺失率、值域、常量列等）。"""
    feature_groups = infer_feature_groups(df.columns.tolist())
    list_columns = detect_list_columns(df, feature_groups.all_features)
    scalar_columns = [
        column for column in feature_groups.all_features if column not in list_columns
    ]

    scalar_feature_ranges = {
        column: _describe_numeric_series(df[column])
        for column in scalar_columns
        if pd.api.types.is_numeric_dtype(df[column])
    }
    list_length_stats = {
        column: _describe_lengths(df[column])
        for column in list_columns
    }
    null_rates = {
        column: round(float(df[column].isna().mean()), 6)
        for column in df.columns
    }
    constant_columns = tuple(
        sorted(column for column in df.columns if _is_constant_column(df[column]))
    )
    label_distribution = {
        str(_to_python_scalar(label)): int(count)
        for label, count in df["label_type"].value_counts(dropna=False).sort_index().items()
    }
    return DatasetAudit(
        unique_users=int(df["user_id"].nunique(dropna=True)),
        unique_items=int(df["item_id"].nunique(dropna=True)),
        duplicate_rows=int(_duplicate_mask(df).sum()),
        label_distribution=label_distribution,
        time_ranges=_maybe_time_ranges(df),
        null_rates=null_rates,
        constant_columns=constant_columns,
        scalar_feature_ranges=scalar_feature_ranges,
        list_length_stats=list_length_stats,
    )


def validate_summary(summary: DatasetSummary) -> None:
    """校验摘要结果是否满足基础数据质量要求。"""
    if any(count > 0 for count in summary.null_counts.values()):
        raise ValueError(
            "Required ID/label columns contain null values: "
            f"{summary.null_counts}"
        )

    mismatched_groups = [
        name for name, is_match in summary.count_matches.items() if not is_match
    ]
    if mismatched_groups:
        mismatched_text = ", ".join(mismatched_groups)
        raise ValueError(
            "Observed feature counts do not match README expectations for groups: "
            f"{mismatched_text}"
        )


def clean_dataset(
    df: pd.DataFrame,
    time_column: str = "label_time",
) -> tuple[pd.DataFrame, CleaningReport]:
    """执行数据清洗：去关键列空值、去重、序列归一化、按时间排序。

    Args:
        df: 原始输入数据表。
        time_column: 排序时间列名。

    Returns:
        tuple[pd.DataFrame, CleaningReport]:
            清洗后的数据表与清洗统计报告。
    """
    feature_groups = infer_feature_groups(df.columns.tolist())
    list_columns = detect_list_columns(df, feature_groups.all_features)

    cleaned = df.copy()
    input_rows = len(cleaned)

    null_mask = cleaned[list(REQUIRED_ID_LABEL_COLUMNS)].isna().any(axis=1)
    dropped_null_rows = int(null_mask.sum())
    cleaned = cleaned.loc[~null_mask].copy()

    duplicate_mask = _duplicate_mask(cleaned)
    dropped_duplicate_rows = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()

    for column in list_columns:
        cleaned[column] = cleaned[column].map(_normalize_sequence_value)

    sorted_by: str | None = None
    if time_column in cleaned.columns:
        cleaned = cleaned.sort_values(time_column, kind="mergesort").reset_index(drop=True)
        sorted_by = time_column

    return cleaned, CleaningReport(
        input_rows=input_rows,
        output_rows=len(cleaned),
        dropped_null_rows=dropped_null_rows,
        dropped_duplicate_rows=dropped_duplicate_rows,
        normalized_list_columns=list_columns,
        sorted_by=sorted_by,
    )


def prepare_training_frame(
    df: pd.DataFrame,
    max_sequence_length: int | None = None,
    sequence_length_overrides: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """构建训练数据表。

    主要动作：
    1. 应用高缺失 user_int 缺失增强
    2. 对 list 特征补充长度列
    3. 对序列特征补充 tensor/mask/length 三元组

    Args:
        df: 清洗后的输入数据表。
        max_sequence_length: 序列默认目标长度。
        sequence_length_overrides: 按列/域覆盖的目标长度配置。

    Returns:
        pd.DataFrame: 可直接供训练输入适配层消费的 prepared 数据表。
    """
    feature_groups = infer_feature_groups(df.columns.tolist())
    list_columns = detect_list_columns(df, feature_groups.all_features)
    prepared = df.copy()
    # 先处理高缺失 user_int 列：补齐值并显式输出缺失指示列，
    # 让下游模型能区分“真实 0”与“缺失后填 0”。
    prepared = _apply_sparse_user_int_missing_logic(
        prepared,
        feature_groups,
        list_columns,
    )

    length_columns: dict[str, pd.Series] = {}
    tensor_columns: dict[str, pd.Series] = {}
    mask_columns: dict[str, pd.Series] = {}
    for column in list_columns:
        normalized = prepared[column].map(_normalize_sequence_value)
        prepared[column] = normalized
        if column in feature_groups.domain_sequence:
            lengths = normalized.map(len)
            # 序列列需要按配置解析目标长度，用于截断与补齐。
            target_length, _ = _resolve_sequence_target_length(
                column,
                lengths,
                max_sequence_length,
                sequence_length_overrides=sequence_length_overrides,
            )
            tensorized = normalized.map(
                lambda values: _tensorize_sequence(values, target_length)
            )
            # 三元组拆包：values 定长序列、mask 有效位、len 原始有效长度。
            tensor_columns[f"{column}_tensor"] = tensorized.map(lambda item: item[0])
            mask_columns[f"{column}_mask"] = tensorized.map(lambda item: item[1])
            length_columns[f"{column}_len"] = tensorized.map(lambda item: item[2])
        else:
            # 非 domain 序列的 list 特征仅记录长度，不额外做定长张量化。
            length_columns[f"{column}_len"] = normalized.map(len)

    if length_columns:
        # 将新增衍生列一次性拼接，保持索引对齐并避免逐列插入带来的性能损耗。
        prepared = pd.concat(
            [
                prepared,
                pd.DataFrame(tensor_columns, index=prepared.index),
                pd.DataFrame(mask_columns, index=prepared.index),
                pd.DataFrame(length_columns, index=prepared.index),
            ],
            axis=1,
        )

    return prepared


def summarize_sequence_tensorization(
    df: pd.DataFrame,
    max_sequence_length: int | None = None,
    sequence_length_overrides: Mapping[str, int] | None = None,
    sequence_length_preset: str | None = None,
) -> dict[str, Any]:
    """汇总序列张量化配置与统计信息。"""
    feature_groups = infer_feature_groups(df.columns.tolist())
    sequence_columns = feature_groups.domain_sequence
    sequence_payload: dict[str, Any] = {}
    normalized_overrides = _normalize_sequence_length_overrides(sequence_length_overrides)

    for column in sequence_columns:
        normalized = df[column].map(_normalize_sequence_value)
        lengths = normalized.map(len)
        target_length, resolved_from = _resolve_sequence_target_length(
            column,
            lengths,
            max_sequence_length,
            sequence_length_overrides=normalized_overrides,
        )
        sequence_payload[column] = {
            "tensor_column": f"{column}_tensor",
            "mask_column": f"{column}_mask",
            "length_column": f"{column}_len",
            "target_length": target_length,
            "resolved_from": resolved_from,
            "observed_max_length": int(lengths.max()) if not lengths.empty else 0,
            "observed_mean_length": round(float(lengths.mean()), 4) if not lengths.empty else 0.0,
            "truncated_rows": int((lengths > target_length).sum()),
            "empty_rows": int((lengths == 0).sum()),
        }

    return {
        "sequence_group_count": len(sequence_columns),
        "sequence_length_preset": sequence_length_preset,
        "default_target_length": max_sequence_length,
        "sequence_length_overrides": normalized_overrides,
        "available_presets": get_sequence_length_presets(),
        "preset_descriptions": get_sequence_length_preset_descriptions(),
        "sequence_columns": sequence_payload,
    }


def build_model_input_spec(
    df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    sequence_tensorization: Mapping[str, Any],
    sequence_length_preset: str | None = None,
) -> dict[str, Any]:
    """构建模型输入规范。

    输出包含 identifiers、scalar/ragged/sequence 布局、
    以及 prepared 数据中新增列的结构说明。

    Args:
        df: 清洗后原始特征数据。
        prepared_df: prepare_training_frame 输出结果。
        sequence_tensorization: 序列张量化摘要配置。
        sequence_length_preset: 使用的长度预设名称。

    Returns:
        dict[str, Any]: model_input_spec 契约字典。
    """
    feature_groups = infer_feature_groups(df.columns.tolist())
    list_columns = detect_list_columns(df, feature_groups.all_features)
    sequence_columns = set(feature_groups.domain_sequence)
    scalar_columns = [
        column for column in feature_groups.all_features if column not in list_columns
    ]
    # 将 prepare_training_frame 生成的缺失指示列并入 scalar 输入契约。
    scalar_columns += _collect_missing_indicator_columns(prepared_df, scalar_columns)
    ragged_columns = [
        column for column in list_columns if column not in sequence_columns
    ]

    ragged_feature_inputs = {
        column: {
            "element_dtype": _infer_sequence_element_dtype(df[column]),
            "length_column": f"{column}_len",
        }
        for column in ragged_columns
    }

    sequence_inputs = {}
    for column in feature_groups.domain_sequence:
        payload = sequence_tensorization["sequence_columns"][column]
        element_dtype = _infer_sequence_element_dtype(df[column])
        # sequence_inputs 描述的是“模型读取方式”，不是数据内容本身。
        # 因此这里同时记录列名映射、padding 语义和 mask 约定，
        # 便于不同训练脚本统一解析。
        sequence_inputs[column] = {
            "source_column": column,
            "tensor_column": payload["tensor_column"],
            "mask_column": payload["mask_column"],
            "length_column": payload["length_column"],
            "target_length": payload["target_length"],
            "resolved_from": payload.get("resolved_from", "default"),
            "element_dtype": element_dtype,
            "padding_value": _padding_value_for_dtype(element_dtype),
            "padding_side": "right",
            "truncation_strategy": "keep_tail",
            "mask_dtype": "int8",
            "mask_semantics": {
                "valid": 1,
                "padding": 0,
            },
        }

    return {
        "sequence_length_configuration": {
            "preset": sequence_length_preset,
            "default_target_length": sequence_tensorization.get("default_target_length"),
            "overrides": sequence_tensorization.get("sequence_length_overrides", {}),
            "available_presets": sequence_tensorization.get("available_presets", {}),
            "preset_descriptions": sequence_tensorization.get("preset_descriptions", {}),
        },
        "identifiers": {
            "id_columns": ["user_id", "item_id"],
            "label_column": "label_type",
            "time_columns": [column for column in ("label_time", "timestamp") if column in df.columns],
        },
        "feature_layout": {
            "scalar_columns": scalar_columns,
            "ragged_feature_inputs": ragged_feature_inputs,
            "sequence_inputs": sequence_inputs,
        },
        "prepared_layout": {
            "rows": len(prepared_df),
            "columns": prepared_df.shape[1],
            "sequence_tensor_columns": [payload["tensor_column"] for payload in sequence_inputs.values()],
            "sequence_mask_columns": [payload["mask_column"] for payload in sequence_inputs.values()],
            "sequence_length_columns": [payload["length_column"] for payload in sequence_inputs.values()],
        },
    }


def split_by_time(
    df: pd.DataFrame,
    valid_ratio: float = 0.2,
    time_column: str = "label_time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间升序切分训练集与验证集。

    Args:
        df: 待切分数据表。
        valid_ratio: 验证集比例。
        time_column: 时间排序列名。

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 训练集与验证集。

    Raises:
        ValueError: 当比例、时间列、标签列或样本量不满足约束时抛出。
    """
    if not 0 < valid_ratio < 1:
        raise ValueError(f"valid_ratio must be between 0 and 1, got {valid_ratio}")
    if time_column not in df.columns:
        raise ValueError(f"Unknown time column: {time_column}")
    if "label_type" not in df.columns:
        raise ValueError("label_type column is required for supervised training")
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to create train/valid splits")

    ordered = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    split_index = int(len(ordered) * (1 - valid_ratio))
    split_index = min(max(split_index, 1), len(ordered) - 1)

    train_df = ordered.iloc[:split_index].reset_index(drop=True)
    valid_df = ordered.iloc[split_index:].reset_index(drop=True)
    return train_df, valid_df


def summarize_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    time_column: str = "label_time",
) -> dict[str, Any]:
    """汇总 train/valid 切分统计信息。"""
    def build_split_stats(frame: pd.DataFrame) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "rows": len(frame),
            "label_distribution": {
                str(_to_python_scalar(label)): int(count)
                for label, count in frame["label_type"].value_counts(dropna=False).sort_index().items()
            },
        }
        if time_column in frame.columns:
            stats["time_range"] = _describe_numeric_series(frame[time_column])
        return stats

    return {
        "train": build_split_stats(train_df),
        "valid": build_split_stats(valid_df),
    }


def write_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """将 train/valid 数据写入 parquet 文件。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.parquet"
    valid_path = output_path / "valid.parquet"
    train_df.to_parquet(train_path, index=False)
    valid_df.to_parquet(valid_path, index=False)
    return train_path, valid_path


def write_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """将 DataFrame 写入 parquet。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_json_file(payload: dict[str, Any], output_path: str | Path) -> Path:
    """将字典写入 JSON 文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
