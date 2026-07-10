# TAAC 数据处理代码说明（Code README）

本文档面向当前目录下的数据处理与建模代码，重点说明：
- 数据处理思路（从原始 parquet 到可训练输入）
- 各脚本职责与调用关系
- 常用运行命令
- 输出目录和关键产物说明

## 1. 目录目标与总体流程

当前目录以 `demo_1000.parquet` 为输入，围绕“可复用的离线训练管线”组织代码。

总体流程：
1. 数据审计与清洗：校验必需列、统计缺失/重复、序列特征规范化。
2. 训练样本构建：生成标量特征、变长特征、定长序列张量（含 mask 和长度）。
3. 高缺失特征增强：对高缺失 user_int 标量列生成缺失指示列（`*_is_missing`）并统一填充值。
4. 时序切分：按 `label_time` 排序后切分 train/valid。
5. 训练输入适配：将 prepared parquet 转为 numpy / torch 可消费 batch。
6. 基线训练：支持 MLP 和序列编码模型，支持类别不平衡处理（采样/加权）与阈值搜索。
7. 误差分析：基于 `valid_predictions.parquet` 进行按标签/时间/活跃度分组分析。
8. Notebook 可视化：在 `notebooks/taac_eda.ipynb` 中汇总报告与图表。

## 2. 数据处理思路

### 2.1 数据分层设计

代码将特征划分为四类输入层：
- 标识与监督信号：`user_id`、`item_id`、`label_type`、`label_time`、`timestamp`
- 标量特征（scalar）：直接作为数值输入
- 变长特征（ragged）：保留原始列表与长度
- 行为序列（sequence）：转为定长 tensor + mask + length

这样做的目的：
- 兼容当前 demo 与未来完整数据集（通过列名前缀自动识别特征组）
- 对不同模型提供稳定输入接口（同一份 prepared 数据可服务多种模型）

### 2.2 序列处理策略

序列特征通过统一规则处理：
- 截断/填充到目标长度（支持预设和手动覆盖）
- 同步生成 mask 与真实长度列
- 在训练时进一步做数值稳定处理（如 `nan_to_num`）

当前提供三个长度预设（见 `taac_dataset.py`）：
- `compact`：更省内存，适合快速实验
- `balanced`：默认推荐，截断与信息保留折中
- `long_context`：保留更长历史，适合离线重实验

### 2.3 切分与评估口径

- 默认按 `label_time` 排序后按比例切分验证集（近时段作为 valid）
- 训练脚本输出验证集预测概率，后续误差分析复用同一口径进行分桶统计

### 2.4 高缺失列处理策略（已实现）

针对高缺失 user_int 标量特征，当前在 `prepare_training_frame` 中已启用：
- 自动识别高缺失列（包含 fid 区间 `83~103` 及缺失率阈值逻辑）
- 生成缺失指示列：`<feature>_is_missing`
- 原始列统一 `fillna(0)`，并由缺失指示列保留“缺失即信息”

该策略会反映到 `reports/model_input_spec.json` 的 `feature_layout.scalar_columns` 中，供训练端直接消费。

## 3. 核心文件说明

### 3.1 预处理与数据规范

- `taac_dataset.py`
  - 数据加载、特征分组、审计、清洗、序列张量化、切分与报告生成核心实现。
  - 关键能力：
    - 必需列校验（ID/Label/时间）
    - 序列长度预设与覆盖
    - 高缺失 user_int 缺失指示列生成（`*_is_missing`）
    - `model_input_spec.json` 生成（训练输入契约）

- `prepare_dataset.py`
  - 预处理总入口（CLI）。
  - 负责串联：加载 -> 审计 -> 清洗 -> prepared 构建 -> train/valid 切分 -> 报告落盘。
  - 支持：
    - `--summary-only` 只看报告不写数据
    - `--sequence-length-preset` 与 `--sequence-length-overrides`

### 3.2 训练输入适配

- `taac_training_inputs.py`
  - 将 prepared parquet + spec 转为样本字典和 batch（numpy）。
  - 提供 `PreparedModelInputDataset` 以统一迭代接口。

- `taac_torch_inputs.py`
  - 将 numpy batch 转换为 torch batch。
  - 提供 dense 特征、序列摘要特征构建，以及设备迁移工具。
  - 包含 NaN/Inf 防护，保障训练稳定性。

- `inspect_training_batch.py`
  - 小工具脚本，用于快速检查样本结构和 batch 形状是否符合预期。

### 3.3 训练与误差分析

- `train_taac_baseline.py`
  - 基线训练入口。
  - 支持模型：
    - `mlp`：纯特征融合 MLP
    - `sequence-encoder`：包含数值序列编码器（默认）
  - 支持类别不平衡处理：
    - 类别加权损失（`CrossEntropyLoss(weight=...)`）
    - `WeightedRandomSampler` 训练采样
    - 验证集阈值搜索（针对 `prob_label_2`，输出最优阈值下指标）
  - 主要输出到 `outputs/training_runs/latest`：
    - `best_model.pt`
    - `metrics.json`
    - `valid_predictions.parquet`
      - 包含 `best_threshold` 与 `predicted_label_at_best_threshold`

- `analyze_valid_predictions.py`
  - 读取验证预测并与 prepared valid 对齐，计算错分分析。
  - 产物写入 `outputs/training_runs/latest/error_analysis`：
    - `label_error_summary.csv`
    - `time_bucket_error_summary.csv`
    - `activity_bucket_error_summary.csv`
    - `high_confidence_errors.csv`
    - `summary.json`

### 3.4 Notebook

- `notebooks/taac_eda.ipynb`
  - 用于展示预处理报告、序列分布、训练输入概览和误差分析可视化。
  - 当前已包含误差分析结果的读取与图表展示单元。

## 4. 常用命令

### 4.1 安装依赖

```bash
pip install -r requirements.txt
```

### 4.2 仅查看数据摘要

```bash
python prepare_dataset.py --summary-only
```

### 4.3 生成 balanced 预处理产物（推荐）

```bash
python prepare_dataset.py --output-dir outputs/prepared_balanced_preset --sequence-length-preset balanced
```

### 4.4 检查训练输入 batch

```bash
python inspect_training_batch.py --prepared-path outputs/prepared_balanced_preset/train.parquet --spec-path outputs/prepared_balanced_preset/reports/model_input_spec.json
```

### 4.5 训练基线模型

```bash
python train_taac_baseline.py --train-path outputs/prepared_balanced_preset/train.parquet --valid-path outputs/prepared_balanced_preset/valid.parquet --spec-path outputs/prepared_balanced_preset/reports/model_input_spec.json
```

可选：启用不平衡策略与阈值搜索（默认已开启 `both`）

```bash
python train_taac_baseline.py \
  --train-path outputs/prepared_balanced_preset/train.parquet \
  --valid-path outputs/prepared_balanced_preset/valid.parquet \
  --spec-path outputs/prepared_balanced_preset/reports/model_input_spec.json \
  --imbalance-strategy both \
  --threshold-search-min 0.05 \
  --threshold-search-max 0.95 \
  --threshold-search-steps 91
```

### 4.6 运行验证误差分析

```bash
python analyze_valid_predictions.py --predictions-path outputs/training_runs/latest/valid_predictions.parquet --valid-prepared-path outputs/prepared_balanced_preset/valid.parquet
```

## 5. 输出目录约定

- `outputs/prepared_*`
  - `cleaned.parquet`：清洗后数据
  - `prepared.parquet`：训练友好格式（含长度列、序列张量列）
    - 包含高缺失 user_int 的缺失指示列 `*_is_missing`
  - `train.parquet` / `valid.parquet`：时序切分结果
  - `reports/*.json`：摘要、审计、清洗、切分、输入规范报告
    - `model_input_spec.json` 的 `feature_layout.scalar_columns` 已包含 `*_is_missing`

- `outputs/training_runs/latest`
  - 训练指标、最佳模型、验证预测
    - `metrics.json`：包含 `imbalance_handling` 与 `best_threshold_summary`
    - `valid_predictions.parquet`：包含 `best_threshold` 与 `predicted_label_at_best_threshold`
  - `error_analysis/`：误差分析细分报告

## 6. 推荐迭代方向

1. 更精细的序列建模：在 `sequence-encoder` 基础上加入跨域注意力或门控融合。
2. 活跃度定义扩展：除历史长度外，加入时间衰减行为计数等指标。
3. 误差分析联动特征：将高置信错分样本回溯到关键特征分布与异常值。
4. 训练配置化：将模型与特征开关抽离到 YAML/JSON，便于批量实验。

## 7. 版本与依赖

见 `requirements.txt`：
- pandas
- pyarrow
- matplotlib
- numpy
- torch

---
如需将该 README 扩展为“新同学上手手册”（含常见报错与排查清单），可在此文件基础上继续补充。