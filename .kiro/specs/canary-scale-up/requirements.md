# 需求文档

## 简介

本特性将 canary 样本数量从 10 扩充到 50，以提升隐私审计实验的统计置信度。改动涉及全链路：canary 生成参数化、数据插入策略重构（修复分布不均问题）、偏好数据生成调整、训练重跑、审计重跑、报告更新。同时纳入 agent_sync T26 提出的统计置信度改进和审计指标扩展，包括多 seed 重复试验、Bootstrap CI、序列级提取成功率、Top-k 命中率、ROC_AUC/PR_AUC 等。核心约束来自 agent_sync T27/T29 共识：canary 占比目标区间 0.3%-0.8%、上限不超过 1%（50/10050 ≈ 0.5%，在范围内），canary 应均匀分布在训练数据中。T29 还要求增加训练有效性防回归验证和 Stage2a/Stage2b 差异 sanity check。

## 术语表

- **Canary**: 插入训练数据中的合成唯一序列，用于追踪模型记忆和隐私泄露
- **Canary_Generator**: canary 样本生成模块（`src/canary.py`）
- **Data_Preparer**: 将 canary 插入 wiki 语料的数据准备模块（`src/prepare_data.py`）
- **Preference_Data_Generator**: 生成 DPO 偏好训练数据的模块（`src/prepare_preference_data.py`）
- **DPO**: Direct Preference Optimization，直接偏好优化训练方法
- **SFT**: Supervised Fine-Tuning，监督微调
- **Audit_Pipeline**: 隐私审计流水线，包含 MIA、Extraction、Internal Signals 模块
- **Stage_Attribution_Analyzer**: 跨阶段隐私风险归因分析模块（`src/stage_attribution.py`）
- **Wiki_Corpus**: 从 Wikipedia 采样的 10,000 条训练语料
- **Canary_Ratio**: canary 样本占总训练数据的比例
- **MIA**: Membership Inference Attack，成员推断攻击
- **ROC_AUC**: Receiver Operating Characteristic - Area Under Curve
- **PR_AUC**: Precision-Recall Area Under Curve
- **Bootstrap_CI**: Bootstrap 置信区间，通过重采样估计统计量的不确定性
- **Effect_Size**: 效应量，量化两组之间差异的实际显著性
- **Run_Metadata**: 运行元数据，包含 seed、commit hash、模型路径、时间戳等可复现信息
- **Extraction_Rate**: 序列级提取成功率，通过 greedy decode 生成文本与完整 canary 序列匹配的成功比例
- **Top5_Hit_Rate**: token 级 Top-5 命中率，canary 各 token 在模型预测该位置 top-5 中命中的平均比例
- **Top10_Hit_Rate**: token 级 Top-10 命中率，同上但 k=10
- **Top50_Hit_Rate**: token 级 Top-50 命中率，同上但 k=50

## 需求

### 需求 1：Canary 生成参数化

**用户故事：** 作为研究人员，我希望通过 CLI 参数控制 canary 生成数量，以便灵活调整实验规模而无需修改代码。

#### 验收标准

1. WHEN 用户指定 `--num-canaries N` 参数时，THE Canary_Generator SHALL 生成恰好 N 条 canary 样本
2. WHEN 未指定 `--num-canaries` 参数时，THE Canary_Generator SHALL 默认生成 50 条 canary 样本
3. THE Canary_Generator SHALL 将生成的 canary 样本写入 `data/canary_output.txt`，每行一条
4. WHEN 指定 `--seed` 参数时，THE Canary_Generator SHALL 使用该种子生成可复现的 canary 序列
5. IF `--num-canaries` 参数值小于 1，THEN THE Canary_Generator SHALL 输出错误信息并以非零退出码终止

### 需求 2：Canary 插入策略重构

**用户故事：** 作为研究人员，我希望 canary 在训练数据中均匀分布，以避免分布不均导致的记忆模式偏差。

#### 验收标准

1. THE Data_Preparer SHALL 根据 canary 数量自动计算插入间隔（`interval = total_wiki // num_canaries`）
2. WHEN 插入 canary 时，THE Data_Preparer SHALL 确保所有 canary 均匀分散在 Wiki_Corpus 中，不在末尾堆积
3. THE Data_Preparer SHALL 将合并后的数据写入 `data/wiki_trimmed_with_canary.jsonl`
4. WHEN 插入完成后，THE Data_Preparer SHALL 在日志中输出 canary 数量、wiki 数量、总样本数和 Canary_Ratio
5. IF canary 数量超过 Wiki_Corpus 的 0.8%，THEN THE Data_Preparer SHALL 输出警告信息（目标占比区间 0.3%-0.8%）
6. IF canary 数量超过 Wiki_Corpus 的 1%，THEN THE Data_Preparer SHALL 输出错误信息并以非零退出码终止（硬上限）
7. THE Data_Preparer SHALL 移除硬编码的 `INTERVAL=900` 常量，改为动态计算
8. WHEN 插入完成后，THE Data_Preparer SHALL 确保相邻 canary 之间的最大间距不超过平均间距的 2 倍，且最小间距不低于平均间距的 0.5 倍（允许因整除余数导致的末尾间距偏差）

### 需求 3：偏好数据生成调整

**用户故事：** 作为研究人员，我希望偏好数据中的 canary 对数量随 canary 样本数自动调整，以保持每个 canary 的偏好对比例一致。

#### 验收标准

1. THE Preference_Data_Generator SHALL 根据实际 canary 数量自动计算 canary 偏好对数量（每个 canary 生成 2 对）
2. THE Preference_Data_Generator SHALL 移除硬编码的 `NUM_CANARY_PAIRS = 20` 常量，改为动态计算
3. WHEN 生成含 canary 的偏好数据时，THE Preference_Data_Generator SHALL 确保每个 canary 至少出现在 2 个偏好对中
4. THE Preference_Data_Generator SHALL 确保两个变体（no-canary 和 with-canary）的普通偏好对内容和数量保持一致
5. WHEN 生成完成后，THE Preference_Data_Generator SHALL 在日志中输出 canary 对数量、普通对数量和总对数

### 需求 4：全链路重跑训练

**用户故事：** 作为研究人员，我希望使用扩充后的 50 canary 数据重新训练所有模型，以获得基于更大样本量的实验结果。

#### 验收标准

1. THE SFT_Training_Notebook SHALL 使用包含 50 canary 的 `data/wiki_trimmed_with_canary.jsonl` 重新训练 SFT 模型
2. THE DPO_Training_Notebook SHALL 使用更新后的偏好数据分别训练 DPO-no-canary 和 DPO-with-canary 变体
3. THE DPO_Training_Notebook SHALL 对两个 DPO 变体使用相同的训练超参数
4. WHEN 训练完成后，THE DPO_Training_Notebook SHALL 将模型保存到对应的 `models/` 子目录
5. WHEN DPO 训练完成后，THE DPO_Training_Notebook SHALL 验证训练后模型权重与训练前 SFT 基础模型权重存在差异（训练有效性防回归）
6. THE DPO_Trainer SHALL 支持 `--seed` 参数以支持多 seed 重复试验

### 需求 5：全链路重跑审计

**用户故事：** 作为研究人员，我希望使用新训练的模型重新执行所有隐私审计，以获得基于 50 canary 的统计结果。

#### 验收标准

1. THE Audit_Pipeline SHALL 对 Stage0_Base、Stage1_SFT、Stage2a_DPO_NoCanary、Stage2b_DPO_WithCanary 四个阶段重新执行隐私审计
2. THE Audit_Pipeline SHALL 使用更新后的 50 条 canary 列表作为审计目标
3. WHEN 审计完成后，THE Audit_Pipeline SHALL 将结果保存到 `reports/` 目录下的对应文件
4. THE Stress_Test_Notebook SHALL 使用 50 canary 重新执行压力测试
5. THE Stress_Test_Notebook SHALL 按 compare_tag 和 variant 分开报告方向一致率，不仅看整体均值
6. IF Stage2a_DPO_NoCanary 和 Stage2b_DPO_WithCanary 的所有审计指标完全相等，THEN THE Audit_Pipeline SHALL 输出警告信息（Stage2a/Stage2b 差异 sanity check）

### 需求 6：审计指标扩展（T26 高优项）

**用户故事：** 作为研究人员，我希望审计流水线包含序列级提取成功率、Top-k 命中率和 ROC_AUC/PR_AUC 指标，以增强隐私风险结论的解释力。

#### 验收标准

1. THE Audit_Pipeline SHALL 对每个阶段计算序列级提取成功率（通过 greedy decode 生成文本，与完整 canary 序列进行匹配）
2. THE Audit_Pipeline SHALL 对每个阶段计算 token 级 Top-k 命中率（k=5, 10, 50），衡量 canary 中每个 token 在模型预测该位置 top-k 中的命中比例（对多 token canary 取各 token 命中率的平均值）
3. THE Audit_Pipeline SHALL 新增 ROC_AUC 和 PR_AUC 指标，同时保留现有 MIA_Gap 以确保向后兼容（MIA_Gap 标注为 deprecated，后续版本移除）
4. WHEN 计算 ROC_AUC 时，THE Audit_Pipeline SHALL 基于 canary 样本（正例）和等量随机采样的 non-canary 样本（负例）构建二分类评估
5. THE Audit_Pipeline SHALL 将扩展指标纳入 `reports/privacy_audit_summary.csv` 的新列，同时保留现有核心列（Stage、MIA_Gap、Avg_LogProb、Avg_Rank、Canary_PPL、PPL_Ratio）确保向后兼容
6. WHEN 展示审计结果时，THE Audit_Pipeline SHALL 以并排方式展示四个阶段的所有指标对比
7. WHEN 新增 CSV 列时，THE Stage_Attribution_Analyzer SHALL 同步更新以识别和处理新增指标列

### 需求 7：统计置信度改进（T26 项）

**用户故事：** 作为研究人员，我希望审计结果包含 Bootstrap 置信区间、效应量和多 seed 重复试验支持，以量化指标差异的统计显著性。

#### 验收标准

1. THE Stage_Attribution_Analyzer SHALL 对关键差值（Stage2b - Stage2a）计算 Bootstrap 95% 置信区间，覆盖以下指标：MIA_Gap、Avg_LogProb、Avg_Rank、Canary_PPL、PPL_Ratio、Extraction_Rate、Top5_Hit_Rate、Top10_Hit_Rate、Top50_Hit_Rate、ROC_AUC、PR_AUC
2. THE Stage_Attribution_Analyzer SHALL 对上述各指标的关键差值计算 Effect_Size（Cohen's d）
3. WHEN Bootstrap_CI 跨越零时，THE Stage_Attribution_Analyzer SHALL 在报告中标注该差异不具统计显著性
4. THE Stage_Attribution_Analyzer SHALL 将 Bootstrap_CI 和 Effect_Size 结果纳入归因报告
5. THE Audit_Pipeline SHALL 支持多 seed 重复试验（MVP 阶段 3 个 seed，扩展阶段 5-10 个 seed），并报告各指标的均值±标准差
6. WHEN 执行多 seed 试验时，训练脚本负责按不同 seed 独立执行训练，THE Audit_Pipeline SHALL 支持接收各 seed 对应的模型并独立执行审计流程

### 需求 8：运行元数据与判定标准（T26 项）

**用户故事：** 作为研究人员，我希望每次实验运行自动记录元数据并使用预定义判定标准，以确保可复现性和减少事后解释偏差。

#### 验收标准

1. WHEN 训练脚本执行时，THE DPO_Trainer SHALL 将训练运行元数据（seed、commit hash、模型路径、时间戳、训练超参数）追加写入 `reports/run_metadata.jsonl`
2. WHEN SFT 训练脚本执行时，THE SFT_Training_Notebook SHALL 将 SFT 训练运行元数据（seed、checkpoint 路径、时间戳、训练超参数）追加写入 `reports/run_metadata.jsonl`
3. WHEN 审计脚本执行时，THE Audit_Pipeline SHALL 将审计运行元数据（seed、canary 数量、Canary_Ratio、审计时间戳、模型路径）追加写入 `reports/run_metadata.jsonl`
4. THE Run_Metadata SHALL 包含足够信息以完全复现该次实验运行（覆盖 SFT、DPO、审计全链路）
5. THE Stage_Attribution_Analyzer SHALL 使用预定义判定标准评估结果（方向一致率阈值、CI 是否跨零）
6. THE Stage_Attribution_Analyzer SHALL 在报告中显式标注每个指标是否满足预定义判定标准

### 需求 9：报告与文档更新

**用户故事：** 作为研究人员，我希望报告和文档准确反映 50 canary 的实验配置和结果，以便其他研究者理解和复现实验。

#### 验收标准

1. THE README SHALL 更新 canary 样本数量描述（从 10 改为 50）
2. THE README SHALL 记录 Canary_Ratio（50/10050 ≈ 0.5%）和总样本量
3. THE Research_Report SHALL 更新实验配置部分，反映 50 canary 的设置
4. THE Research_Report SHALL 更新所有审计结果表格和图表，包含新增的扩展指标
5. WHEN 更新文档时，THE README SHALL 说明插入策略从固定间隔改为动态计算间隔

### 需求 10：数据完整性与可复现性

**用户故事：** 作为研究人员，我希望扩充后的实验数据满足统计约束且可完全复现，以确保实验结论的可靠性。

#### 验收标准

1. THE Data_Preparer SHALL 确保 Canary_Ratio 低于 1%（目标区间 0.3%-0.8%，超过 1% 硬失败）
2. THE Canary_Generator SHALL 支持通过固定 seed 生成完全相同的 canary 序列
3. THE Data_Preparer SHALL 支持通过固定 seed 生成完全相同的插入结果
4. WHEN 生成数据时，THE Data_Preparer SHALL 在输出中显式记录 canary 占比、canary 数量和总样本量
5. THE Preference_Data_Generator SHALL 确保移除 canary 对后，两个变体的普通偏好对通过逐行哈希比较完全一致
