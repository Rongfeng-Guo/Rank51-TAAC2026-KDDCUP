7:1 的不平衡如果不处理，模型会偏向预测 `label=1`。  
在这个项目里，最实用的是“**加权损失 + 分层采样 + 阈值调优**”三件套。

**推荐方案（按优先级）**
1. 先上类别加权损失（最稳）
2. 再加训练集重采样（提升召回）
3. 最后在验证集做阈值调优（提升业务指标）

---

**1. 加权损失（建议先做）**
在 CrossEntropy 里给少数类更高权重。  
当前分布：`n1=876, n2=124`，可用：
$$
w_i = \frac{N}{K \cdot n_i}
$$
得到大致：
- `w_label1 ≈ 0.57`
- `w_label2 ≈ 4.03`

这样模型错分 `label=2` 的代价更高，通常能显著提高正样本召回。

---

**2. 训练采样：WeightedRandomSampler**
对每个样本按“类别倒数频率”赋采样概率，让 batch 里正负更均衡。  
简单理解：`label=2` 样本被抽到的概率更高。  
注意：
- 用了 sampler 后，DataLoader 不要再 `shuffle=True`
- 建议“轻度重采样”，避免过拟合少数类（可配一个倍率）

---

**3. 阈值调优（别固定 0.5）**
你现在有 `prob_label_2`，在验证集扫阈值（比如 0.1~0.9），按目标指标选最优阈值：
- 如果你更关心“抓住感兴趣用户”：优化 `F1` 或 `Recall@positive`
- 如果你更关心“少误报”：优化 `Precision`

很多时候，仅做阈值优化就能明显改善线上体验。

---

**4. 评估指标要改**
除了 accuracy，一定要看：
1. PR-AUC
2. 正类 Precision / Recall / F1
3. 混淆矩阵
4. 按时间桶的正类召回（防止后段漂移）

---

**在你当前代码里的落地点**
- 训练损失： train_taac_baseline.py 里的 `nn.CrossEntropyLoss()` 改为带 `weight`
- 训练 DataLoader： train_taac_baseline.py 里的 `train_loader` 改为 `WeightedRandomSampler`
- 验证后处理： train_taac_baseline.py 的 `evaluate` 后增加阈值扫描并保存最佳阈值到 `metrics.json`

train_taac_baseline.py，（改造前后正类 Recall/F1）。