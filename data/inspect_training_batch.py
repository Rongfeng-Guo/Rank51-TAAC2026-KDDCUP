from __future__ import annotations
"""训练输入快速体检脚本。

用于在不启动完整训练的前提下，快速验证 prepared 数据与 spec
是否能正确构造样本与 batch。
"""

import argparse
import json
from pathlib import Path

from taac_training_inputs import PreparedModelInputDataset, summarize_model_input_batch


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Inspect model-ready samples and batches from a prepared TAAC parquet split."
    )
    parser.add_argument(
        "--prepared-path",
        type=Path,
        default=Path("outputs") / "prepared_balanced_preset" / "train.parquet",
        help="Path to a prepared train/valid parquet split.",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=Path("outputs") / "prepared_balanced_preset" / "reports" / "model_input_spec.json",
        help="Path to the model_input_spec.json file for the prepared split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used for the inspection preview.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Dataset sample index used for the per-sample preview.",
    )
    return parser.parse_args()


def main() -> None:
    """执行样本与 batch 的结构体检并打印结果。"""
    args = parse_args()
    dataset = PreparedModelInputDataset.from_paths(args.prepared_path, args.spec_path)
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError(
            f"sample-index must be between 0 and {len(dataset) - 1}, got {args.sample_index}"
        )

    sample = dataset[args.sample_index]
    batch = next(dataset.iter_batches(args.batch_size))
    batch_summary = summarize_model_input_batch(batch)

    preview = {
        "dataset_rows": len(dataset),
        "sample_index": args.sample_index,
        "sample_identifiers": sample["identifiers"],
        "sample_scalar_feature_count": len(sample["scalar_features"]),
        "sample_ragged_feature_count": len(sample["ragged_features"]),
        "sample_sequence_feature_count": len(sample["sequence_features"]),
        "batch_summary": batch_summary,
    }
    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()