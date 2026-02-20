# 需求文档

## 简介

本特性旨在解决当前项目中 DPO 训练阶段的实验设计矛盾：README 声称 DPO 阶段不引入 canary，但 `prepare_preference_data.py` 实际在偏好数据中包含了 20 条 canary 偏好对。为此，将 DPO 阶段拆分为两个实验组（DPO-no-canary 和 DPO-with-canary），分别训练和审计，以对比分析偏好数据中 canary 的存在对隐私风险的影响。

## 术语表

- **Canary**: 插入训练数据中的合成唯一序列，用于追踪模型记忆和隐私泄露
- **DPO**: Direct Preference Optimization，直接偏好优化训练方法
- **DPO-no-canary**: 偏好数据中不包含任何 canary 对的 DPO 训练变体
- **DPO-with-canary**: 偏好数据中包含 canary 偏好对的 DPO 训练变体
- **Preference_Data_Generator**: 生成 DPO 偏好训练数据的模块 (`src/prepare_preference_data.py`)
- **DPO_Trainer**: 执行 DPO 训练的模块 (`src/train_dpo.py`)
- **Stage_Attribution_Analyzer**: 跨阶段隐私风险归因分析模块 (`src/stage_attribution.py`)
- **Audit_Pipeline**: 隐私审计流水线，包含 MIA、Extraction、Internal Signals 模块
- **MIA**: Membership Inference Attack，成员推断攻击
- **PPL**: Perplexity，困惑度

## 需求

### 需求 1：偏好数据生成支持双变体

**用户故事：** 作为研究人员，我希望能分别生成包含和不包含 canary 的偏好数据，以便对 DPO 训练中 canary 的影响进行消融实验。

#### 验收标准

1. WHEN 用户指定 `--no-canary` 标志时，THE Preference_Data_Generator SHALL 生成不包含任何 canary 偏好对的偏好数据文件
2. WHEN 用户指定 `--with-canary` 标志时，THE Preference_Data_Generator SHALL 生成包含 canary 偏好对的偏好数据文件
3. WHEN 生成不含 canary 的偏好数据时，THE Preference_Data_Generator SHALL 将输出保存到 `data/preference_data_no_canary.jsonl`
4. WHEN 生成含 canary 的偏好数据时，THE Preference_Data_Generator SHALL 将输出保存到 `data/preference_data_with_canary.jsonl`
5. THE Preference_Data_Generator SHALL 在两种模式下生成相同数量的普通偏好对（2000 对）
6. WHEN 未指定任何标志时，THE Preference_Data_Generator SHALL 默认生成两个变体的偏好数据文件
7. IF 同时指定 `--no-canary` 和 `--with-canary` 标志，THEN THE Preference_Data_Generator SHALL 输出错误信息并以非零退出码终止执行

### 需求 2：DPO 训练支持双变体

**用户故事：** 作为研究人员，我希望能分别训练不含 canary 和含 canary 的 DPO 模型，以便对比两种训练条件下的隐私风险差异。

#### 验收标准

1. WHEN 用户指定偏好数据路径和输出目录时，THE DPO_Trainer SHALL 使用指定的偏好数据训练模型并保存到指定目录
2. WHEN 训练 DPO-no-canary 变体时，THE DPO_Trainer SHALL 将模型保存到 `models/stage2_dpo_no_canary/`
3. WHEN 训练 DPO-with-canary 变体时，THE DPO_Trainer SHALL 将模型保存到 `models/stage2_dpo_with_canary/`
4. THE DPO_Trainer SHALL 对两个变体使用相同的训练超参数（learning rate、epochs、batch size、beta）
5. THE DPO_Trainer SHALL 对两个变体使用相同的 SFT 基础模型（`models/stage1_sft/`）

### 需求 3：DPO 训练 Notebook 更新

**用户故事：** 作为研究人员，我希望在 Colab notebook 中能依次训练两个 DPO 变体，以便在 GPU 环境下完成消融实验。

#### 验收标准

1. THE DPO_Training_Notebook SHALL 包含训练 DPO-no-canary 变体的代码段
2. THE DPO_Training_Notebook SHALL 包含训练 DPO-with-canary 变体的代码段
3. WHEN 执行训练时，THE DPO_Training_Notebook SHALL 先训练 DPO-no-canary 变体，再训练 DPO-with-canary 变体
4. THE DPO_Training_Notebook SHALL 将训练完成的模型保存到本地输出目录，并可选上传到 Google Drive

### 需求 4：隐私审计支持四阶段评估

**用户故事：** 作为研究人员，我希望审计流水线能评估四个阶段（Base、SFT、DPO-no-canary、DPO-with-canary）的隐私风险，以便全面对比分析。

#### 验收标准

1. THE Audit_Pipeline SHALL 对 Stage0_Base、Stage1_SFT、Stage2a_DPO_NoCanary、Stage2b_DPO_WithCanary 四个阶段分别执行隐私审计
2. WHEN 执行审计时，THE Audit_Pipeline SHALL 对每个阶段计算 MIA Gap、Avg LogProb、Avg Rank、Canary PPL、PPL Ratio 五项指标
3. THE Audit_Pipeline SHALL 将四阶段审计结果保存到 `reports/privacy_audit_summary.csv`
4. WHEN 展示审计结果时，THE Audit_Pipeline SHALL 以并排方式展示 DPO-no-canary 和 DPO-with-canary 的指标对比

### 需求 5：阶段归因分析扩展

**用户故事：** 作为研究人员，我希望阶段归因分析能处理四个阶段的数据，以便量化 canary 在 DPO 偏好数据中的影响。

#### 验收标准

1. THE Stage_Attribution_Analyzer SHALL 计算 Base→SFT、SFT→DPO-no-canary、SFT→DPO-with-canary 三个阶段转换的指标变化
2. THE Stage_Attribution_Analyzer SHALL 计算 Stage2a_DPO_NoCanary 与 Stage2b_DPO_WithCanary 之间的指标差异
3. WHEN 生成归因报告时，THE Stage_Attribution_Analyzer SHALL 包含四阶段的完整归因分数
4. THE Stage_Attribution_Analyzer SHALL 生成包含四阶段对比的可视化图表
5. WHEN 生成报告时，THE Stage_Attribution_Analyzer SHALL 将结果保存到 `reports/attribution_summary.json`

### 需求 6：文档更新

**用户故事：** 作为研究人员，我希望 README 和项目计划准确反映双组实验设计，以便其他研究者理解实验方法。

#### 验收标准

1. THE README SHALL 描述 DPO 阶段包含两个实验组（DPO-no-canary 和 DPO-with-canary）
2. THE README SHALL 更新训练阶段表格，包含 Stage 2a 和 Stage 2b 的描述
3. THE README SHALL 更新仓库结构说明，包含新增的模型目录和数据文件
4. THE PROJECT_PLAN SHALL 更新系统架构图，反映四阶段设计
5. THE PROJECT_PLAN SHALL 更新实验矩阵，包含四个阶段的审计维度

### 需求 7：数据完整性保障

**用户故事：** 作为研究人员，我希望两个 DPO 变体的训练数据除 canary 对外完全一致，以便确保消融实验的有效性。

#### 验收标准

1. THE Preference_Data_Generator SHALL 确保两个变体的普通偏好对内容和数量完全相同
2. WHEN 生成偏好数据时，THE Preference_Data_Generator SHALL 使用相同的随机种子生成普通偏好对
3. THE Preference_Data_Generator SHALL 支持通过移除 canary 对后逐行哈希比较来验证两个变体的普通偏好对一致性
4. IF 输入数据文件缺失，THEN THE Preference_Data_Generator SHALL 返回描述性错误信息并终止执行
5. IF 输入数据文件为空，THEN THE Preference_Data_Generator SHALL 返回描述性错误信息并终止执行
