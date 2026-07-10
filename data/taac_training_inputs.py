from __future__ import annotations
"""训练输入适配层（Pandas/NumPy 版本）。

职责：
1. 读取 prepared 数据与 model_input_spec
2. 将单条样本映射为统一结构
3. 将样本列表拼接为批次（batch）
4. 提供简易 Dataset 迭代接口

接口契约（输入）：
- prepared parquet：包含原始特征与派生列（*_len, *_tensor, *_mask 等）
- model_input_spec.json：定义 identifiers、scalar、ragged、sequence 布局

接口契约（输出）：
- 单样本结构：identifiers/scalar_features/ragged_features/sequence_features
- 批次结构：上述字段按 batch 维拼接为 NumPy 容器
- 下游消费者：taac_torch_inputs.py、inspect_training_batch.py
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd


def _to_python_scalar(value: Any) -> Any:
    """将 numpy / pandas 标量转换为 Python 原生标量。"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def load_model_input_spec(spec_path: str | Path) -> dict[str, Any]:
    """读取模型输入规范 JSON。

    Args:
        spec_path: model_input_spec.json 路径。

    Returns:
        dict[str, Any]: 解析后的输入规范字典。
    """
    path = Path(spec_path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_prepared_frame(prepared_path: str | Path) -> pd.DataFrame:
    """读取预处理后的 parquet 数据。"""
    return pd.read_parquet(Path(prepared_path))


def _build_identifier_sample(row: pd.Series, spec: Mapping[str, Any]) -> dict[str, Any]:
    """提取样本的标识字段（用户/物料/标签/时间）。"""
    identifier_spec = spec["identifiers"]
    sample = {
        column: _to_python_scalar(row[column])
        for column in identifier_spec["id_columns"]
    }
    sample[identifier_spec["label_column"]] = _to_python_scalar(
        row[identifier_spec["label_column"]]
    )
    for column in identifier_spec["time_columns"]:
        sample[column] = _to_python_scalar(row[column])
    return sample


def _build_scalar_sample(row: pd.Series, spec: Mapping[str, Any]) -> dict[str, Any]:
    """提取样本的标量特征。"""
    return {
        column: _to_python_scalar(row[column])
        for column in spec["feature_layout"]["scalar_columns"]
    }


def _build_ragged_sample(row: pd.Series, spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """提取样本的变长（ragged）特征及长度信息。"""
    ragged_payload: dict[str, dict[str, Any]] = {}
    ragged_spec = spec["feature_layout"]["ragged_feature_inputs"]
    for column, payload in ragged_spec.items():
        ragged_payload[column] = {
            "values": list(row[column]),
            "length": int(row[payload["length_column"]]),
            "element_dtype": payload["element_dtype"],
        }
    return ragged_payload


def _build_sequence_sample(row: pd.Series, spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """提取样本的序列张量输入（values/mask/length）。"""
    sequence_payload: dict[str, dict[str, Any]] = {}
    sequence_spec = spec["feature_layout"]["sequence_inputs"]
    for column, payload in sequence_spec.items():
        sequence_payload[column] = {
            "values": list(row[payload["tensor_column"]]),
            "mask": list(row[payload["mask_column"]]),
            "length": int(row[payload["length_column"]]),
            "target_length": int(payload["target_length"]),
            "element_dtype": payload["element_dtype"],
            "padding_value": payload["padding_value"],
        }
    return sequence_payload


def build_model_input_sample(
    row: pd.Series,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """将一行 DataFrame 转换为模型输入样本结构。

    Args:
        row: prepared 数据单行。
        spec: model_input_spec 配置。

    Returns:
        dict[str, Any]: 标准化样本结构。
    """
    return {
        "identifiers": _build_identifier_sample(row, spec),
        "scalar_features": _build_scalar_sample(row, spec),
        "ragged_features": _build_ragged_sample(row, spec),
        "sequence_features": _build_sequence_sample(row, spec),
    }


def _numpy_array(values: Sequence[Any]) -> np.ndarray:
    """统一构造 numpy 数组并保证标量已 Python 化。"""
    return np.asarray([_to_python_scalar(value) for value in values])


def collate_model_input_batch(
    samples: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """将样本列表拼接为 NumPy 批次结构。

    Args:
        samples: 样本序列。
        spec: model_input_spec 配置。

    Returns:
        dict[str, Any]: 按批次组织的 NumPy 容器结构。

    Raises:
        ValueError: 当 samples 为空时抛出。
    """
    if not samples:
        raise ValueError("samples must not be empty")

    identifiers = {
        column: _numpy_array([sample["identifiers"][column] for sample in samples])
        for column in spec["identifiers"]["id_columns"]
    }
    identifiers[spec["identifiers"]["label_column"]] = _numpy_array(
        [sample["identifiers"][spec["identifiers"]["label_column"]] for sample in samples]
    )
    for column in spec["identifiers"]["time_columns"]:
        identifiers[column] = _numpy_array(
            [sample["identifiers"][column] for sample in samples]
        )

    scalar_features = {
        column: _numpy_array([sample["scalar_features"][column] for sample in samples])
        for column in spec["feature_layout"]["scalar_columns"]
    }

    ragged_features: dict[str, dict[str, Any]] = {}
    for column in spec["feature_layout"]["ragged_feature_inputs"]:
        ragged_features[column] = {
            "values": [sample["ragged_features"][column]["values"] for sample in samples],
            "lengths": _numpy_array(
                [sample["ragged_features"][column]["length"] for sample in samples]
            ),
        }

    sequence_features: dict[str, dict[str, Any]] = {}
    for column, payload in spec["feature_layout"]["sequence_inputs"].items():
        dtype = np.float32 if payload["element_dtype"] == "float" else np.int64
        sequence_features[column] = {
            "values": np.asarray(
                [sample["sequence_features"][column]["values"] for sample in samples],
                dtype=dtype,
            ),
            "mask": np.asarray(
                [sample["sequence_features"][column]["mask"] for sample in samples],
                dtype=np.int8,
            ),
            "lengths": _numpy_array(
                [sample["sequence_features"][column]["length"] for sample in samples]
            ),
        }

    return {
        "batch_size": len(samples),
        "identifiers": identifiers,
        "scalar_features": scalar_features,
        "ragged_features": ragged_features,
        "sequence_features": sequence_features,
    }


def summarize_model_input_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """输出批次结构摘要，便于快速自检。"""
    first_sequence_name = next(iter(batch["sequence_features"]))
    first_sequence = batch["sequence_features"][first_sequence_name]
    first_ragged_name = next(iter(batch["ragged_features"]))
    first_ragged = batch["ragged_features"][first_ragged_name]

    return {
        "batch_size": int(batch["batch_size"]),
        "identifier_columns": sorted(batch["identifiers"].keys()),
        "scalar_feature_count": len(batch["scalar_features"]),
        "ragged_feature_count": len(batch["ragged_features"]),
        "sequence_feature_count": len(batch["sequence_features"]),
        "first_sequence_feature": {
            "name": first_sequence_name,
            "values_shape": list(first_sequence["values"].shape),
            "mask_shape": list(first_sequence["mask"].shape),
            "lengths_shape": list(first_sequence["lengths"].shape),
        },
        "first_ragged_feature": {
            "name": first_ragged_name,
            "batch_item_count": len(first_ragged["values"]),
            "lengths_shape": list(first_ragged["lengths"].shape),
        },
    }


@dataclass
class PreparedModelInputDataset:
    """基于 prepared parquet 的轻量 Dataset 封装。"""
    frame: pd.DataFrame
    spec: dict[str, Any]

    @classmethod
    def from_paths(
        cls,
        prepared_path: str | Path,
        spec_path: str | Path,
    ) -> PreparedModelInputDataset:
        """从文件路径初始化 Dataset。"""
        return cls(
            frame=load_prepared_frame(prepared_path),
            spec=load_model_input_spec(spec_path),
        )

    def __len__(self) -> int:
        """返回样本总数。"""
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按索引读取并构造单样本输入。"""
        row = self.frame.iloc[index]
        return build_model_input_sample(row, self.spec)

    def iter_batches(
        self,
        batch_size: int,
        drop_last: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """按批次迭代样本，输出 NumPy 批次结构。

        Args:
            batch_size: 批大小。
            drop_last: 是否丢弃最后一个不足 batch 的尾批次。

        Yields:
            Iterator[dict[str, Any]]: 每次产出一个批次结构。

        Raises:
            ValueError: 当 batch_size 小于 1 时抛出。
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            if drop_last and stop - start < batch_size:
                break
            samples = [self[index] for index in range(start, stop)]
            yield collate_model_input_batch(samples, self.spec)
