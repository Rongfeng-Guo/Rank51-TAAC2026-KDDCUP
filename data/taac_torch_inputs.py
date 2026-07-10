from __future__ import annotations
"""PyTorch 训练输入桥接层。

本模块将 NumPy/Python 结构化 batch 转换为 Torch 张量，
并提供 dense/sequence 特征构建、设备迁移、标签提取等通用能力。

接口契约（输入）：
- 来自 taac_training_inputs 的批次结构（identifiers/scalar/ragged/sequence）
- sequence payload 需包含 values/mask/lengths，且维度按 spec 对齐

接口契约（输出）：
- Torch 批次结构：可直接送入训练循环
- dense 特征：标量值 + 变长长度 + 序列长度
- sequence 摘要特征：mean/std/max/last/coverage 拼接向量
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from taac_training_inputs import (
    PreparedModelInputDataset,
    build_model_input_sample,
    collate_model_input_batch,
    load_model_input_spec,
    load_prepared_frame,
)


def _to_tensor(value: Any, dtype: torch.dtype | None = None) -> Tensor:
    """将任意数组/标量安全转换为 Torch Tensor。"""
    if isinstance(value, Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def torchify_model_input_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """将 NumPy 批次转换为 Torch 批次。

    Args:
        batch: 来自 taac_training_inputs 的批次字典。

    Returns:
        dict[str, Any]: 结构等价但张量字段转换为 torch.Tensor 的批次。
    """
    identifiers = {
        name: _to_tensor(values)
        for name, values in batch["identifiers"].items()
    }
    scalar_features = {
        name: _to_tensor(values, dtype=torch.float32)
        for name, values in batch["scalar_features"].items()
    }

    ragged_features = {
        name: {
            "values": payload["values"],
            "lengths": _to_tensor(payload["lengths"], dtype=torch.long),
        }
        for name, payload in batch["ragged_features"].items()
    }

    sequence_features = {
        name: {
            "values": _to_tensor(payload["values"]),
            "mask": _to_tensor(payload["mask"], dtype=torch.float32),
            "lengths": _to_tensor(payload["lengths"], dtype=torch.long),
        }
        for name, payload in batch["sequence_features"].items()
    }

    return {
        "batch_size": batch["batch_size"],
        "identifiers": identifiers,
        "scalar_features": scalar_features,
        "ragged_features": ragged_features,
        "sequence_features": sequence_features,
    }


def build_torch_collate_fn(
    spec: Mapping[str, Any],
) -> Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]]:
    """构造 DataLoader 可用的 collate_fn。

    Args:
        spec: model_input_spec 配置。

    Returns:
        Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]]: 可直接传给 DataLoader 的拼批函数。
    """

    def collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch = collate_model_input_batch(samples, spec)
        return torchify_model_input_batch(batch)

    return collate


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """将批次中所有张量迁移到指定设备（CPU/GPU）。

    Args:
        batch: torch 化后的批次结构。
        device: 目标设备。

    Returns:
        dict[str, Any]: 迁移后的批次副本。
    """
    moved = {
        "batch_size": batch["batch_size"],
        "identifiers": {},
        "scalar_features": {},
        "ragged_features": {},
        "sequence_features": {},
    }

    moved["identifiers"] = {
        name: value.to(device) if isinstance(value, Tensor) else value
        for name, value in batch["identifiers"].items()
    }
    moved["scalar_features"] = {
        name: value.to(device)
        for name, value in batch["scalar_features"].items()
    }
    moved["ragged_features"] = {
        name: {
            "values": payload["values"],
            "lengths": payload["lengths"].to(device),
        }
        for name, payload in batch["ragged_features"].items()
    }
    moved["sequence_features"] = {
        name: {
            "values": payload["values"].to(device),
            "mask": payload["mask"].to(device),
            "lengths": payload["lengths"].to(device),
        }
        for name, payload in batch["sequence_features"].items()
    }
    return moved


def build_dense_feature_tensor(batch: Mapping[str, Any]) -> Tensor:
    """构建 dense 输入：标量值 + ragged 长度 + sequence 长度。

    Args:
        batch: torch 化批次结构。

    Returns:
        Tensor: dense 特征张量，形状为 [batch_size, dense_dim]。
    """
    # 采用排序后的键顺序拼接，保证不同运行间特征维度一致。
    scalar_tensor = torch.stack(
        [values.float() for _, values in sorted(batch["scalar_features"].items())],
        dim=1,
    )
    scalar_tensor = torch.nan_to_num(scalar_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    ragged_length_tensor = torch.stack(
        [
            payload["lengths"].float()
            for _, payload in sorted(batch["ragged_features"].items())
        ],
        dim=1,
    )

    sequence_length_tensor = torch.stack(
        [
            payload["lengths"].float()
            for _, payload in sorted(batch["sequence_features"].items())
        ],
        dim=1,
    )

    dense_tensor = torch.cat(
        [scalar_tensor, ragged_length_tensor, sequence_length_tensor],
        dim=1,
    )
    return torch.nan_to_num(dense_tensor, nan=0.0, posinf=0.0, neginf=0.0)


def build_sequence_summary_tensor(batch: Mapping[str, Any]) -> Tensor:
    """构建序列摘要特征。

    每个序列提取 mean/std/max/last/coverage 五类统计，
    并做数值稳定处理，最终拼接成定长向量。

    Args:
        batch: torch 化批次结构，需包含 sequence_features。

    Returns:
        Tensor: 序列摘要张量，形状为 [batch_size, summary_dim]。
    """
    summary_tensors: list[Tensor] = []

    for _, payload in sorted(batch["sequence_features"].items()):
        # 先做数值清洗，再做符号-对数变换，压缩长尾并保留方向信息。
        values = torch.nan_to_num(
            payload["values"].float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        mask = payload["mask"].float()
        transformed = torch.sign(values) * torch.log1p(torch.abs(values))
        masked_values = transformed * mask

        # mean/std 基于有效位（mask=1）统计，避免 padding 干扰。
        lengths = payload["lengths"].float().clamp_min(1.0)
        raw_lengths = payload["lengths"].float()
        mean = masked_values.sum(dim=1) / lengths
        square_mean = masked_values.square().sum(dim=1) / lengths
        variance = torch.clamp(square_mean - mean.square(), min=0.0)
        std = torch.sqrt(variance)

        # max 通过将 padding 位填为 -inf 实现，之后再回填 0。
        masked_for_max = torch.where(
            mask > 0,
            transformed,
            torch.full_like(transformed, float("-inf")),
        )
        max_value = masked_for_max.max(dim=1).values
        max_value = torch.where(
            torch.isfinite(max_value),
            max_value,
            torch.zeros_like(max_value),
        )

        # last 取最后一个有效 token；空序列回退为 0。
        last_indices = payload["lengths"].clamp_min(1).long() - 1
        last_value = transformed.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        last_value = torch.where(
            raw_lengths > 0,
            last_value,
            torch.zeros_like(last_value),
        )

        # coverage 表示有效长度占目标长度比例，是序列完整度信号。
        coverage = raw_lengths / float(values.shape[1])
        summary_tensors.extend([mean, std, max_value, last_value, coverage])

    summary_tensor = torch.stack(summary_tensors, dim=1)
    return torch.nan_to_num(summary_tensor, nan=0.0, posinf=0.0, neginf=0.0)


def build_model_feature_tensor(
    batch: Mapping[str, Any],
    feature_mode: str = "dense-plus-sequence",
) -> Tensor:
    """根据 feature_mode 组合模型输入特征。

    Args:
        batch: torch 化批次结构。
        feature_mode: 特征模式，支持 dense-only/dense-plus-sequence。

    Returns:
        Tensor: 模型最终输入特征张量。

    Raises:
        ValueError: 当 feature_mode 非法时抛出。
    """
    dense_tensor = build_dense_feature_tensor(batch)
    if feature_mode == "dense-only":
        return dense_tensor
    if feature_mode != "dense-plus-sequence":
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    sequence_summary_tensor = build_sequence_summary_tensor(batch)
    return torch.cat([dense_tensor, sequence_summary_tensor], dim=1)


def get_label_tensor(batch: Mapping[str, Any]) -> Tensor:
    """提取标签张量，并将 label_type 映射到从 0 开始的类别索引。"""
    return batch["identifiers"]["label_type"].long() - 1


def summarize_torch_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """输出 Torch batch 关键形状信息，便于调试。"""
    dense_features = build_dense_feature_tensor(batch)
    sequence_summary = build_sequence_summary_tensor(batch)
    first_sequence_name = next(iter(batch["sequence_features"]))
    first_sequence = batch["sequence_features"][first_sequence_name]
    return {
        "batch_size": int(batch["batch_size"]),
        "dense_features_shape": list(dense_features.shape),
        "sequence_summary_shape": list(sequence_summary.shape),
        "dense_plus_sequence_shape": list(
            build_model_feature_tensor(batch, feature_mode="dense-plus-sequence").shape
        ),
        "label_shape": list(get_label_tensor(batch).shape),
        "first_sequence_name": first_sequence_name,
        "first_sequence_values_shape": list(first_sequence["values"].shape),
        "first_sequence_mask_shape": list(first_sequence["mask"].shape),
    }


@dataclass
class TorchPreparedDataset(Dataset[dict[str, Any]]):
    """面向 PyTorch 的 prepared 数据集封装。"""
    base_dataset: PreparedModelInputDataset

    @classmethod
    def from_paths(
        cls,
        prepared_path: str | Path,
        spec_path: str | Path,
    ) -> TorchPreparedDataset:
        """从 prepared parquet 与 spec 路径初始化数据集。"""
        return cls(
            base_dataset=PreparedModelInputDataset(
                frame=load_prepared_frame(prepared_path),
                spec=load_model_input_spec(spec_path),
            )
        )

    @property
    def spec(self) -> dict[str, Any]:
        """返回模型输入规范字典。"""
        return self.base_dataset.spec

    def __len__(self) -> int:
        """返回样本总数。"""
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按索引读取单条样本并转为模型输入结构。"""
        row = self.base_dataset.frame.iloc[index]
        return build_model_input_sample(row, self.base_dataset.spec)

    def iter_torch_batches(
        self,
        batch_size: int,
        drop_last: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """按批次迭代并直接返回 Torch 张量结构。"""
        collate_fn = build_torch_collate_fn(self.spec)
        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            if drop_last and stop - start < batch_size:
                break
            samples = [self[index] for index in range(start, stop)]
            yield collate_fn(samples)
