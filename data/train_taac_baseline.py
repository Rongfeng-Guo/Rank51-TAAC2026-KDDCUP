from __future__ import annotations
"""TAAC 基线训练脚本。

功能覆盖：
1. MLP 与序列编码模型训练
2. 类别不平衡处理（采样/加权）
3. 验证集阈值搜索与指标落盘
4. 模型、指标、预测结果输出

接口契约（输入）：
- train/valid prepared parquet（需与 spec 对齐）
- model_input_spec.json（字段布局契约）
- 可选训练策略：imbalance-strategy、feature-mode、model-type

接口契约（输出）：
- best_model.pt：最佳验证损失对应的模型权重与训练状态
- metrics.json：训练日志、类别不平衡配置、阈值搜索结果
- valid_predictions.parquet：概率输出 + 最优阈值推断标签
"""

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from taac_torch_inputs import (
    TorchPreparedDataset,
    build_dense_feature_tensor,
    build_model_feature_tensor,
    build_torch_collate_fn,
    get_label_tensor,
    move_batch_to_device,
    summarize_torch_batch,
)


PredictionRow: TypeAlias = dict[str, object]
BinaryMetrics: TypeAlias = dict[str, float | int]


class FeatureFusionTAACBaseline(nn.Module):
    """基于融合特征的 MLP 基线模型。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向计算，输出二分类 logits。"""
        return self.network(features)


class NumericSequenceEncoder(nn.Module):
    """数值序列编码器。

    对原始序列先做符号-对数变换，再经 token MLP 投影，
    最终聚合 mean/max/last/coverage 并映射到固定维度。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.token_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.summary_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 1),
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """对单个序列特征进行编码，返回固定长度表示。"""
        normalized_values = torch.sign(values.float()) * torch.log1p(torch.abs(values.float()))
        token_states = self.token_projection(normalized_values.unsqueeze(-1))

        mask = mask.float()
        mask_expanded = mask.unsqueeze(-1)
        lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        mean_state = (token_states * mask_expanded).sum(dim=1) / lengths
        masked_token_states = token_states.masked_fill(mask_expanded == 0, float("-inf"))
        max_state = masked_token_states.max(dim=1).values
        max_state = torch.where(
            torch.isfinite(max_state),
            max_state,
            torch.zeros_like(max_state),
        )

        last_indices = mask.sum(dim=1).long().clamp_min(1) - 1
        gather_index = last_indices.view(-1, 1, 1).expand(-1, 1, token_states.size(-1))
        last_state = token_states.gather(1, gather_index).squeeze(1)
        last_state = torch.where(
            mask.sum(dim=1, keepdim=True) > 0,
            last_state,
            torch.zeros_like(last_state),
        )

        coverage = (mask.sum(dim=1, keepdim=True) / mask.size(1)).float()
        summary = torch.cat([mean_state, max_state, last_state, coverage], dim=1)
        return self.summary_projection(summary)


class SequenceAwareTAACModel(nn.Module):
    """序列感知模型：dense 塔 + 多域序列聚合 + 分类器。"""

    def __init__(
        self,
        dense_input_dim: int,
        sequence_feature_names: list[str],
        hidden_dim: int,
        output_dim: int,
        sequence_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.domain_feature_names = self._group_domain_features(sequence_feature_names)
        self.domain_names = tuple(sorted(self.domain_feature_names))
        self.dense_tower = nn.Sequential(
            nn.LayerNorm(dense_input_dim),
            nn.Linear(dense_input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.sequence_encoder = NumericSequenceEncoder(sequence_hidden_dim)
        fusion_dim = hidden_dim + len(self.domain_names) * sequence_hidden_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _group_domain_features(sequence_feature_names: list[str]) -> dict[str, list[str]]:
        """按域名将序列特征分组。"""
        grouped: dict[str, list[str]] = {}
        for feature_name in sorted(sequence_feature_names):
            domain_name = feature_name.split("_seq_")[0]
            grouped.setdefault(domain_name, []).append(feature_name)
        return grouped

    def forward(
        self,
        dense_features: torch.Tensor,
        sequence_features: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """融合 dense 表示与各域序列表示，输出分类 logits。"""
        dense_representation = self.dense_tower(dense_features)

        domain_representations = []
        for domain_name in self.domain_names:
            encoded_sequences = []
            for feature_name in self.domain_feature_names[domain_name]:
                payload = sequence_features[feature_name]
                encoded_sequences.append(
                    self.sequence_encoder(payload["values"], payload["mask"])
                )
            domain_representations.append(torch.stack(encoded_sequences, dim=1).mean(dim=1))

        fused_representation = torch.cat(
            [dense_representation] + domain_representations,
            dim=1,
        )
        return self.classifier(fused_representation)


def parse_args() -> argparse.Namespace:
    """定义并解析训练脚本参数。

    Returns:
        argparse.Namespace: 训练配置参数集合。
    """
    parser = argparse.ArgumentParser(
        description="Train a minimal dense baseline on TAAC prepared splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "training_runs" / "latest",
        help="Directory used to store metrics, checkpoints, and validation predictions.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("outputs") / "prepared_balanced_preset" / "train.parquet",
        help="Path to the prepared training parquet split.",
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=Path("outputs") / "prepared_balanced_preset" / "valid.parquet",
        help="Path to the prepared validation parquet split.",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=Path("outputs") / "prepared_balanced_preset" / "reports" / "model_input_spec.json",
        help="Path to the model_input_spec.json file.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--model-type",
        choices=["mlp", "sequence-encoder"],
        default="sequence-encoder",
        help="Model family used for training.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["dense-only", "dense-plus-sequence"],
        default="dense-plus-sequence",
        help="Feature composition used before the MLP baseline.",
    )
    parser.add_argument(
        "--sequence-hidden-dim",
        type=int,
        default=32,
        help="Hidden size used by the numeric sequence encoder when model-type=sequence-encoder.",
    )
    parser.add_argument(
        "--imbalance-strategy",
        choices=["none", "class-weight", "sampler", "both"],
        default="both",
        help="Imbalance handling strategy for label_type.",
    )
    parser.add_argument(
        "--threshold-search-min",
        type=float,
        default=0.05,
        help="Minimum probability threshold for label=2 threshold search.",
    )
    parser.add_argument(
        "--threshold-search-max",
        type=float,
        default=0.95,
        help="Maximum probability threshold for label=2 threshold search.",
    )
    parser.add_argument(
        "--threshold-search-steps",
        type=int,
        default=91,
        help="Number of points in threshold search grid (inclusive).",
    )
    return parser.parse_args()


def _compute_binary_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
) -> BinaryMetrics:
    """计算二分类核心指标（Precision/Recall/F1 与混淆矩阵项）。"""
    tp = fp = tn = fn = 0
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_positive = true_label == 2
        predicted_positive = predicted_label == 2
        if true_positive and predicted_positive:
            tp += 1
        elif not true_positive and predicted_positive:
            fp += 1
        elif true_positive and not predicted_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": (tp + fp) / max(tp + fp + tn + fn, 1),
    }


def _threshold_metrics_from_predictions(
    prediction_rows: list[PredictionRow],
    threshold: float,
) -> BinaryMetrics:
    """在给定阈值下计算正类指标。"""
    true_labels = [int(row["label_type"]) for row in prediction_rows]
    predicted_labels = [2 if float(row["prob_label_2"]) >= threshold else 1 for row in prediction_rows]
    metrics = _compute_binary_metrics(true_labels, predicted_labels)
    metrics["threshold"] = threshold
    return metrics


def _search_best_threshold(
    prediction_rows: list[PredictionRow],
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> BinaryMetrics:
    """在阈值网格上搜索 F1 最优点（并以 recall 作为并列优先级）。

    Args:
        prediction_rows: 验证集预测明细，每条记录需包含 label_type 与 prob_label_2。
        threshold_min: 阈值搜索下界（含）。
        threshold_max: 阈值搜索上界（含）。
        threshold_steps: 网格点数，至少为 2。

    Returns:
        BinaryMetrics: 最优阈值对应指标，包含 threshold/precision/recall/f1 等字段。

    Raises:
        ValueError: 当阈值范围非法、步数不足或无法计算指标时抛出。
    """
    if threshold_steps < 2:
        raise ValueError(f"threshold-search-steps must be at least 2, got {threshold_steps}")
    if not 0.0 <= threshold_min < threshold_max <= 1.0:
        raise ValueError(
            "threshold-search range must satisfy 0 <= min < max <= 1, "
            f"got min={threshold_min}, max={threshold_max}"
        )

    best_metrics: BinaryMetrics | None = None
    step_size = (threshold_max - threshold_min) / (threshold_steps - 1)
    for index in range(threshold_steps):
        threshold = threshold_min + index * step_size
        metrics = _threshold_metrics_from_predictions(prediction_rows, threshold)
        if best_metrics is None:
            best_metrics = metrics
            continue
        if metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
        elif metrics["f1"] == best_metrics["f1"] and metrics["recall"] > best_metrics["recall"]:
            # F1 相同场景优先召回，适配“不要漏掉正样本”的推荐任务目标。
            best_metrics = metrics

    if best_metrics is None:
        raise ValueError("No threshold metrics could be computed")
    return best_metrics


def _build_class_weights_and_sampler(
    train_dataset: TorchPreparedDataset,
    strategy: str,
) -> tuple[torch.Tensor | None, WeightedRandomSampler | None, dict[str, Any]]:
    """根据标签分布构建类别权重与采样器。

    Args:
        train_dataset: 训练集对象（要求包含 label_type 列）。
        strategy: 不平衡策略，可选 none/class-weight/sampler/both。

    Returns:
        tuple[torch.Tensor | None, WeightedRandomSampler | None, dict[str, Any]]:
            - criterion 权重（用于 CrossEntropyLoss）
            - 采样器（用于 DataLoader）
            - 调试信息（类别计数、类别权重、策略开关）
    """
    labels = train_dataset.base_dataset.frame["label_type"].astype(int) - 1
    # label_type 原始从 1 开始，这里映射到 [0, 1] 以适配 CrossEntropyLoss。
    label_tensor = torch.as_tensor(labels.to_numpy(), dtype=torch.long)
    class_counts = torch.bincount(label_tensor, minlength=2)
    class_counts_float = class_counts.float().clamp_min(1.0)
    total = float(class_counts.sum().item())
    # 采用反频率权重：样本少的类别权重大，缓解不平衡偏置。
    class_weights = total / (class_counts_float * len(class_counts_float))

    sampler: WeightedRandomSampler | None = None
    if strategy in {"sampler", "both"}:
        sample_weights = class_weights[label_tensor].double()
        # replacement=True 允许少数类样本在一个 epoch 内被重复采到。
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    criterion_weights = class_weights if strategy in {"class-weight", "both"} else None
    debug_payload = {
        "strategy": strategy,
        "class_counts": {
            "label_1": int(class_counts[0].item()),
            "label_2": int(class_counts[1].item()),
        },
        "class_weights": {
            "label_1": round(float(class_weights[0].item()), 6),
            "label_2": round(float(class_weights[1].item()), 6),
        },
        "uses_sampler": sampler is not None,
        "uses_class_weight": criterion_weights is not None,
    }
    return criterion_weights, sampler, debug_payload


def build_logits(
    model: nn.Module,
    batch: dict[str, dict[str, torch.Tensor]],
    model_type: str,
    feature_mode: str,
) -> torch.Tensor:
    """按模型类型组织输入并返回 logits。

    Args:
        model: 待前向推理模型。
        batch: 已 torch 化并完成设备迁移的批次。
        model_type: 模型类型（mlp 或 sequence-encoder）。
        feature_mode: 特征组合模式（仅对 mlp 生效）。

    Returns:
        torch.Tensor: 分类 logits，形状为 [batch_size, 2]。

    Raises:
        ValueError: 当 model_type 非法时抛出。
    """
    if model_type == "mlp":
        features = build_model_feature_tensor(batch, feature_mode=feature_mode)
        return model(features)
    if model_type == "sequence-encoder":
        dense_features = build_dense_feature_tensor(batch)
        return model(dense_features, batch["sequence_features"])
    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    model_type: str,
    feature_mode: str,
    collect_predictions: bool = False,
) -> tuple[float, float, list[PredictionRow] | None]:
    """在验证集上评估损失、准确率，并可选导出预测明细。

    Args:
        model: 待评估模型。
        loader: 验证集 DataLoader。
        device: 评估设备。
        criterion: 损失函数。
        model_type: 模型类型。
        feature_mode: 特征模式。
        collect_predictions: 是否额外返回逐样本预测明细。

    Returns:
        tuple[float, float, list[PredictionRow] | None]:
            平均损失、准确率、可选预测明细。
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    prediction_rows: list[PredictionRow] | None = [] if collect_predictions else None
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = get_label_tensor(batch)
            logits = build_logits(
                model,
                batch,
                model_type=model_type,
                feature_mode=feature_mode,
            )
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

            if prediction_rows is not None:
                probabilities = torch.softmax(logits, dim=1)
                predicted_labels = logits.argmax(dim=1) + 1
                for index in range(batch_size):
                    prediction_rows.append(
                        {
                            "user_id": int(batch["identifiers"]["user_id"][index].detach().cpu().item()),
                            "item_id": int(batch["identifiers"]["item_id"][index].detach().cpu().item()),
                            "label_type": int(batch["identifiers"]["label_type"][index].detach().cpu().item()),
                            "label_time": int(batch["identifiers"]["label_time"][index].detach().cpu().item()),
                            "timestamp": int(batch["identifiers"]["timestamp"][index].detach().cpu().item()),
                            "predicted_label": int(predicted_labels[index].detach().cpu().item()),
                            "prob_label_1": float(probabilities[index, 0].detach().cpu().item()),
                            "prob_label_2": float(probabilities[index, 1].detach().cpu().item()),
                        }
                    )

    return total_loss / total_examples, total_correct / total_examples, prediction_rows


def main() -> None:
    """执行完整训练流程并输出模型/指标/预测文件。

    流程包含：
    1. 参数解析与数据集构建
    2. 不平衡策略配置与训练循环
    3. 验证集阈值评估
    4. checkpoint/metrics/predictions 落盘
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TorchPreparedDataset.from_paths(args.train_path, args.spec_path)
    valid_dataset = TorchPreparedDataset.from_paths(args.valid_path, args.spec_path)
    collate_fn = build_torch_collate_fn(train_dataset.spec)

    criterion_weights, train_sampler, imbalance_debug = _build_class_weights_and_sampler(
        train_dataset,
        strategy=args.imbalance_strategy,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # sampler 与 shuffle 互斥：启用 sampler 后由采样器决定顺序。
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    sample_batch = next(iter(train_loader))
    dense_input_dim = build_dense_feature_tensor(sample_batch).shape[1]
    input_dim = build_model_feature_tensor(
        sample_batch,
        feature_mode=args.feature_mode,
    ).shape[1]
    output_dim = 2

    if args.model_type == "mlp":
        model = FeatureFusionTAACBaseline(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
        ).to(device)
    else:
        sequence_feature_names = list(
            train_dataset.spec["feature_layout"]["sequence_inputs"].keys()
        )
        model = SequenceAwareTAACModel(
            dense_input_dim=dense_input_dim,
            sequence_feature_names=sequence_feature_names,
            hidden_dim=args.hidden_dim,
            output_dim=output_dim,
            sequence_hidden_dim=args.sequence_hidden_dim,
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(
        weight=criterion_weights.to(device) if criterion_weights is not None else None
    )

    training_log: list[dict[str, float | int | dict[str, object]]] = []
    best_valid_loss = float("inf")
    best_epoch = 0
    best_predictions: list[PredictionRow] | None = None
    best_threshold_summary: dict[str, BinaryMetrics] | None = None
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_examples = 0
        total_correct = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            labels = get_label_tensor(batch)

            optimizer.zero_grad(set_to_none=True)
            logits = build_logits(
                model,
                batch,
                model_type=args.model_type,
                feature_mode=args.feature_mode,
            )
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = running_loss / total_examples
        train_accuracy = total_correct / total_examples
        valid_loss, valid_accuracy, valid_predictions = evaluate(
            model,
            valid_loader,
            device,
            criterion,
            model_type=args.model_type,
            feature_mode=args.feature_mode,
            collect_predictions=True,
        )

        # 同时记录固定阈值(0.5)与搜索最优阈值下的正类指标，
        # 便于后续按业务目标（召回/精确）做阈值选择。
        default_threshold_metrics = _threshold_metrics_from_predictions(
            valid_predictions or [],
            threshold=0.5,
        )
        best_threshold_metrics = _search_best_threshold(
            valid_predictions or [],
            threshold_min=args.threshold_search_min,
            threshold_max=args.threshold_search_max,
            threshold_steps=args.threshold_search_steps,
        )

        training_log.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "valid_loss": round(valid_loss, 6),
                "valid_accuracy": round(valid_accuracy, 6),
                "valid_positive_metrics_at_0_5": {
                    "precision": round(float(default_threshold_metrics["precision"]), 6),
                    "recall": round(float(default_threshold_metrics["recall"]), 6),
                    "f1": round(float(default_threshold_metrics["f1"]), 6),
                },
                "valid_best_threshold": round(float(best_threshold_metrics["threshold"]), 6),
                "valid_positive_metrics_at_best_threshold": {
                    "precision": round(float(best_threshold_metrics["precision"]), 6),
                    "recall": round(float(best_threshold_metrics["recall"]), 6),
                    "f1": round(float(best_threshold_metrics["f1"]), 6),
                },
            }
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_predictions = valid_predictions
            # 只保留“最佳验证损失对应轮次”的阈值摘要，
            # 保证最终落盘指标与最佳 checkpoint 对齐。
            best_threshold_summary = {
                "default_threshold_metrics": default_threshold_metrics,
                "best_threshold_metrics": best_threshold_metrics,
            }
            torch.save(
                {
                    "model_type": args.model_type,
                    "feature_mode": args.feature_mode,
                    "sequence_hidden_dim": args.sequence_hidden_dim,
                    "input_dim": dense_input_dim if args.model_type == "sequence-encoder" else input_dim,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_valid_loss": best_valid_loss,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )

    metrics_path = args.output_dir / "metrics.json"
    predictions_path = args.output_dir / "valid_predictions.parquet"
    if best_predictions is not None:
        export_rows = pd.DataFrame(best_predictions)
        threshold = 0.5
        if best_threshold_summary is not None:
            threshold = float(best_threshold_summary["best_threshold_metrics"]["threshold"])
        # 将最优阈值和阈值重算后的预测标签写回 parquet，
        # 供误差分析或后处理脚本直接复用。
        export_rows["best_threshold"] = threshold
        export_rows["predicted_label_at_best_threshold"] = (
            (export_rows["prob_label_2"] >= threshold).astype("int64") + 1
        )
        export_rows.to_parquet(predictions_path, index=False)

    result = {
        "device": str(device),
        "model_type": args.model_type,
        "feature_mode": args.feature_mode,
        "input_dim": int(input_dim if args.model_type == "mlp" else dense_input_dim),
        "dense_input_dim": int(dense_input_dim),
        "output_dim": output_dim,
        "sequence_hidden_dim": args.sequence_hidden_dim,
        "imbalance_handling": imbalance_debug,
        "best_epoch": best_epoch,
        "best_valid_loss": round(best_valid_loss, 6),
        "best_threshold_summary": best_threshold_summary,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "batch_preview": summarize_torch_batch(sample_batch),
        "training_log": training_log,
    }
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()