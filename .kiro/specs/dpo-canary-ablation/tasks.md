# 实现计划：DPO Canary 消融实验

## 概述

将现有单一 DPO 流程拆分为双变体消融实验。按照数据生成 → 训练脚本 → Notebook → 审计 → 归因 → 文档的顺序递增实现，每步构建在前一步之上。

## 任务

- [x] 1. 重构偏好数据生成器支持双变体
  - [x] 1.1 为 `src/prepare_preference_data.py` 添加 argparse CLI 接口
    - 添加 `--no-canary`、`--with-canary`、`--seed` 参数
    - 实现标志冲突检测（同时指定两个标志时报错退出）
    - 默认行为（无标志）生成两个变体文件
    - _Requirements: 1.1, 1.2, 1.6, 1.7_

  - [x] 1.2 重构核心生成逻辑为 `generate_preference_data()` 函数
    - 提取现有逻辑为可测试的纯函数
    - 参数化 `include_canary`、`seed`、`num_normal_pairs`、`num_canary_pairs`
    - 使用固定种子生成普通偏好对，with-canary 变体在末尾追加 canary 对（不重新 shuffle 普通对）
    - _Requirements: 1.5, 7.1, 7.2_

  - [x] 1.3 添加输入验证和错误处理
    - 检查 wiki 数据文件和 canary 文件是否存在且非空
    - 缺失或空文件时输出描述性错误并 `sys.exit(1)`
    - _Requirements: 7.4, 7.5_

  - [x] 1.4 实现哈希验证函数 `verify_data_equivalence()`
    - 从两个文件中过滤 canary 对，逐行哈希比较普通偏好对
    - 返回 bool 结果和差异详情
    - _Requirements: 7.3_

  - [x] 1.5 更新输出路径逻辑
    - `--no-canary` → `data/preference_data_no_canary.jsonl`
    - `--with-canary` → `data/preference_data_with_canary.jsonl`
    - 默认模式生成两个文件
    - _Requirements: 1.3, 1.4_

  - [ ]* 1.6 编写 Property Test: Canary 包含正确性
    - **Property 1: Canary 包含正确性**
    - **Validates: Requirements 1.1, 1.2**

  - [ ]* 1.7 编写 Property Test: 普通偏好对跨变体等价性
    - **Property 2: 普通偏好对跨变体等价性**
    - **Validates: Requirements 7.1, 7.2, 1.5**

  - [ ]* 1.8 编写 Property Test: 哈希验证正确性
    - **Property 3: 哈希验证正确性**
    - **Validates: Requirements 7.3**

  - [ ]* 1.9 编写单元测试
    - 测试 CLI 参数解析（各种标志组合）
    - 测试错误处理（缺失文件、空文件、标志冲突）
    - 测试输出文件路径正确性
    - _Requirements: 1.3, 1.4, 1.6, 1.7, 7.4, 7.5_

- [x] 2. Checkpoint - 确保偏好数据生成器测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 3. 参数化 DPO 训练脚本
  - [x] 3.1 为 `src/train_dpo.py` 添加 argparse CLI 接口
    - 添加 `--preference-data`（必填）、`--output-dir`（必填）、`--sft-model`、`--base-model` 参数
    - 替换硬编码的路径常量为 CLI 参数值
    - 添加输入文件/目录存在性验证
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 编写单元测试
    - 测试 argparse 正确解析各种参数组合
    - 测试缺失必填参数时的错误行为
    - _Requirements: 2.1_

- [x] 4. 更新 DPO 训练 Notebook
  - [x] 4.1 修改 `notebooks/02_dpo_training.ipynb`
    - 添加数据准备 section（调用 `prepare_preference_data.py` 生成两个变体）
    - 添加 Section A：训练 DPO-no-canary（`--preference-data data/preference_data_no_canary.jsonl --output-dir models/stage2_dpo_no_canary`）
    - 添加 Section B：训练 DPO-with-canary（`--preference-data data/preference_data_with_canary.jsonl --output-dir models/stage2_dpo_with_canary`）
    - 添加可选 Google Drive 上传 section
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 5. 扩展隐私审计 Notebook 为四阶段
  - [x] 5.1 修改 `notebooks/05_privacy_audit.ipynb`
    - 更新 STAGES 字典为四阶段（Stage0_Base, Stage1_SFT, Stage2a_DPO_NoCanary, Stage2b_DPO_WithCanary）
    - 扩展审计循环从 3 次到 4 次
    - 更新 CSV 输出包含四行数据
    - 添加 DPO 变体并排对比展示
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 5.2 编写 Property Test: CSV 输出格式正确性
    - **Property 6: CSV 输出格式正确性**
    - **Validates: Requirements 4.3**

- [x] 6. Checkpoint - 确保审计流水线测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 7. 扩展阶段归因分析为四阶段
  - [x] 7.1 重构 `src/stage_attribution.py` 支持四阶段
    - 添加 argparse CLI 接口（`--input-csv`、`--output-dir`、`--tolerant`）
    - 更新 `compute_stage_deltas()` 支持四个阶段转换
    - 新增 `compute_dpo_comparison()` 函数计算两个 DPO 变体差异
    - 更新 `compute_attribution_scores()` 包含三个训练阶段的归因
    - 实现严格模式（默认，缺失阶段报错）和宽松模式（`--tolerant`，跳过缺失阶段）
    - _Requirements: 5.1, 5.2, 5.3, 5.5_

  - [x] 7.2 更新可视化函数
    - `plot_stage_comparison()` 从 3 组柱状图扩展为 4 组
    - `plot_attribution_breakdown()` 包含三个训练阶段的归因
    - 新增 DPO 变体对比子图
    - _Requirements: 5.4_

  - [ ]* 7.3 编写 Property Test: 阶段差值算术正确性
    - **Property 4: 阶段差值算术正确性**
    - **Validates: Requirements 5.1, 5.2**

  - [ ]* 7.4 编写 Property Test: 归因报告完整性
    - **Property 5: 归因报告完整性**
    - **Validates: Requirements 5.3**

  - [ ]* 7.5 编写单元测试: 严格/宽松模式行为
    - 严格模式（默认）：缺失 Stage 行时应报错退出
    - 宽松模式（`--tolerant`）：缺失 Stage 行时应输出警告并继续
    - _Requirements: 5.1, 5.3_

- [x] 8. 更新项目文档
  - [x] 8.1 更新 `README.md`
    - 更新训练阶段表格（Stage 2a + Stage 2b）
    - 更新 Canary Design 说明（两组实验设计）
    - 更新仓库结构（新增模型目录和数据文件）
    - 更新结果表格预留四阶段数据
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 8.2 更新 `doc/PROJECT_PLAN.md`
    - 更新系统架构图为四阶段
    - 更新实验矩阵为 4 stages × 3 metrics × 3 variants
    - 更新文件结构说明
    - _Requirements: 6.4, 6.5_

- [x] 9. 确保测试依赖可用
  - 确认 `pytest` 已在 `requirements.txt` 中
  - 当执行可选 property test 任务时，将 `hypothesis` 添加到 `requirements.txt`
  - 确保现有测试仍然通过

- [x] 10. 最终 Checkpoint - 全部测试通过
  - 确保所有测试通过，如有问题请询问用户。

## 备注

- 标记 `*` 的任务为可选，可跳过以加速 MVP。MVP 合并要求所有非 `*` 任务完成；`*` 任务可在后续加固阶段补充。
- 每个任务引用具体需求以确保可追溯性
- Checkpoint 确保增量验证
- Property test 验证通用正确性属性
- 单元测试验证具体示例和边界情况
- Notebook 修改（任务 4、5）需要在 Colab 环境中验证
- DPO 训练（任务 4）需要 GPU 环境执行
