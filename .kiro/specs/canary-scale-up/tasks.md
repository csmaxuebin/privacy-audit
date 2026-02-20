# 实施计划：Canary Scale-Up（10→50）

## 概述

将 canary 样本从 10 扩充到 50，涉及数据准备链参数化、插入策略重构、审计指标扩展、统计置信度改进、运行元数据落盘，以及全链路重跑训练和审计。实施按依赖顺序推进：先基础设施（元数据、依赖），再数据准备链，再审计扩展，再统计分析，最后文档更新。

## 任务

- [x] 1. 基础设施准备
  - [x] 1.1 创建 `src/run_metadata.py` 模块
    - 实现 `append_metadata()` 函数（JSONL 追加写入）
    - 实现 `load_metadata()` 函数（逐行解析，跳过损坏行）
    - 实现 `get_git_commit()` 辅助函数
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [ ]* 1.2 编写 property test：元数据追加一致性
    - **Property 11: 元数据追加一致性**
    - **Validates: Requirements 8.1, 8.2, 8.3**

  - [x] 1.3 添加测试依赖到 `requirements.txt`
    - 添加 `hypothesis`、`scikit-learn` 到依赖文件
    - _Requirements: 设计文档测试策略_

- [x] 2. Canary 生成参数化（`src/canary.py`）
  - [x] 2.1 添加 argparse CLI 接口
    - 添加 `--num-canaries`（默认 50）、`--seed`、`--output` 参数
    - 移除 `__main__` 中硬编码的 `10`
    - 添加 `--num-canaries < 1` 错误处理
    - 调用 `append_metadata()` 记录生成元数据
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 编写 property test：Canary 生成数量与文件一致性
    - **Property 1: Canary 生成数量与文件一致性**
    - **Validates: Requirements 1.1, 1.3**

  - [ ]* 2.3 编写 property test：Canary 生成种子确定性
    - **Property 2: Canary 生成种子确定性**
    - **Validates: Requirements 1.4, 10.2**

- [x] 3. Canary 插入策略重构（`src/prepare_data.py`）
  - [x] 3.1 实现动态间隔插入逻辑
    - 实现 `compute_insertion_positions()` 函数
    - 实现 `validate_distribution()` 函数
    - 移除硬编码 `INTERVAL=900`
    - 添加占比检查（>0.8% 警告，>1% 硬失败）
    - 添加日志输出（canary 数量、wiki 数量、总样本数、Canary_Ratio）
    - 调用 `append_metadata()` 记录数据准备元数据
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 3.2 编写 property test：Canary 插入分布均匀性
    - **Property 3: Canary 插入分布均匀性**
    - **Validates: Requirements 2.1, 2.2, 2.8, 10.1**

  - [ ]* 3.3 编写 property test：数据准备种子确定性
    - **Property 6: 数据准备种子确定性**
    - **Validates: Requirements 10.3**

- [x] 4. 偏好数据生成调整（`src/prepare_preference_data.py`）
  - [x] 4.1 移除硬编码 `NUM_CANARY_PAIRS`，改为自动计算
    - 添加 `PAIRS_PER_CANARY = 2` 常量
    - 添加 `compute_num_canary_pairs()` 函数
    - 修改 `generate_preference_data()` 使用自动计算的 canary 对数
    - 添加日志输出（canary 对数量、普通对数量、总对数）
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

  - [ ]* 4.2 编写 property test：Canary 偏好对数量与覆盖
    - **Property 4: Canary 偏好对数量与覆盖**
    - **Validates: Requirements 3.1, 3.3**

  - [ ]* 4.3 编写 property test：偏好数据变体等价性
    - **Property 5: 偏好数据变体等价性**
    - **Validates: Requirements 3.4, 10.5**

- [x] 5. Checkpoint - 数据准备链验证
  - 运行 `pytest tests/ -v -k "canary or prepare_data or preference"` 确保所有数据准备相关测试通过
  - 验证 `data/canary_output.txt` 包含 50 行非空内容
  - 验证 `data/wiki_trimmed_with_canary.jsonl` 行数为 10050
  - 如有测试失败，修复后重跑；如有歧义，询问用户

- [x] 6. 审计指标扩展
  - [x] 6.1 扩展 `src/audit/extraction.py`：添加 `compute_extraction_rate()` 和 `compute_topk_hit_rates()` 函数
    - 复用现有 `canary_extraction_test()` 实现 `compute_extraction_rate()`
    - 新增 `topk_hit_rate()` 和 `compute_topk_hit_rates()` 函数（k=5,10,50）
    - _Requirements: 6.1, 6.2_

  - [x] 6.2 扩展 `src/audit/mia.py`：添加 `compute_auc_metrics()` 函数
    - 使用 `sklearn.metrics.roc_auc_score` 和 `average_precision_score`
    - 保留现有 `loss_based_mia()` 返回的 MIA_Gap（标注 deprecated）
    - _Requirements: 6.3, 6.4_

  - [ ]* 6.3 编写 property test：AUC 指标有效性
    - **Property 7: AUC 指标有效性**
    - **Validates: Requirements 6.3**

  - [x] 6.4 更新审计 CSV 输出格式
    - 在 `reports/privacy_audit_summary.csv` 中新增 6 列扩展指标
    - 保留现有核心列确保向后兼容
    - _Requirements: 6.5_

  - [ ]* 6.5 编写 property test：扩展 CSV 与归因兼容性
    - **Property 8: 扩展 CSV 与归因兼容性**
    - **Validates: Requirements 6.5, 6.7**

- [x] 7. 统计置信度改进（`src/stage_attribution.py`）
  - [x] 7.1 扩展 `METRICS` 列表为 `EXTENDED_METRICS`（11 个指标）
    - 更新 `compute_stage_deltas()` 和 `compute_attribution_scores()` 支持扩展指标
    - 添加 Stage2a/Stage2b 全相等 sanity check 警告
    - _Requirements: 6.7, 5.6_

  - [x] 7.2 实现 `bootstrap_ci()` 和 `cohens_d()` 函数
    - 实现 Bootstrap 95% CI 计算
    - 实现 Cohen's d 效应量计算
    - 实现 `evaluate_criteria()` 判定标准评估函数
    - 将结果纳入归因报告 JSON
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 7.3 编写 property test：Bootstrap CI 正确性
    - **Property 9: Bootstrap CI 正确性**
    - **Validates: Requirements 7.1, 7.3**

  - [ ]* 7.4 编写 property test：Cohen's d 正确性
    - **Property 10: Cohen's d 正确性**
    - **Validates: Requirements 7.2**

  - [ ]* 7.5 编写 property test：判定标准评估正确性
    - **Property 12: 判定标准评估正确性**
    - **Validates: Requirements 8.5**

  - [x] 7.6 实现预定义判定标准和报告标注
    - 添加 `DECISION_CRITERIA` 常量
    - 在归因报告中标注每个指标是否满足判定标准
    - _Requirements: 8.5, 8.6_

- [x] 8. Checkpoint - 审计扩展与统计改进验证
  - 运行 `pytest tests/ -v -k "audit or attribution or bootstrap or auc"` 确保所有审计和统计相关测试通过
  - 验证 `src/stage_attribution.py` 的 `EXTENDED_METRICS` 包含 11 个指标
  - 如有测试失败，修复后重跑；如有歧义，询问用户

- [x] 9. DPO Trainer 扩展（`src/train_dpo.py`）
  - [x] 9.1 添加 `--seed` CLI 参数
    - 在 argparse 中添加 `--seed` 参数
    - 在训练逻辑中使用 seed 设置随机种子
    - 训练完成后调用 `append_metadata()` 记录训练元数据
    - _Requirements: 4.6, 8.1_

- [x] 10. Notebook 更新（训练链）
  - [x] 10.1 更新 `notebooks/01_sft_training.ipynb`
    - 确保使用 50 canary 的 `data/wiki_trimmed_with_canary.jsonl`
    - 添加 SFT 训练元数据记录
    - _Requirements: 4.1, 8.2_

  - [x] 10.2 更新 `notebooks/02_dpo_training.ipynb`
    - 更新数据准备步骤使用 50 canary
    - 添加训练有效性验证（权重差异检查）
    - _Requirements: 4.2, 4.3, 4.4, 4.5_

- [x] 11. Notebook 更新（审计链）
  - [x] 11.1 更新 `notebooks/03_audit_stage0_stage1.ipynb`
    - 使用 50 canary 列表作为审计目标
    - 添加扩展指标计算（Extraction_Rate, Top-k, ROC_AUC, PR_AUC）
    - _Requirements: 5.1, 5.2, 6.1, 6.2, 6.3_

  - [x] 11.2 更新 `notebooks/04_stress_test.ipynb`
    - 使用 50 canary 重新执行压力测试
    - 按 compare_tag 和 variant 分开报告方向一致率
    - _Requirements: 5.4, 5.5_

  - [x] 11.3 更新 `notebooks/05_privacy_audit.ipynb`
    - 使用 50 canary 执行四阶段审计
    - 输出扩展 CSV 格式
    - 添加 Stage2a/Stage2b 全相等 sanity check
    - 添加审计元数据记录
    - _Requirements: 5.1, 5.2, 5.3, 5.6, 6.5, 6.6, 8.3_

- [x] 12. Checkpoint - 全链路验证
  - 运行 `pytest tests/ -v` 确保全部测试通过
  - 验证 `reports/privacy_audit_summary.csv` 包含 4 行数据和 12 列（含扩展指标）
  - 验证 `reports/run_metadata.jsonl` 包含训练和审计的元数据记录
  - 如有测试失败，修复后重跑；如有歧义，询问用户

- [x] 13. 文档更新
  - [x] 13.1 更新 `README.md`
    - 更新 canary 数量（10→50）和 Canary_Ratio（0.5%）
    - 说明插入策略从固定间隔改为动态计算
    - _Requirements: 9.1, 9.2, 9.5_

  - [x] 13.2 更新 `doc/Research_Report_2026-02-10.md`
    - 更新实验配置部分
    - 更新审计结果表格和图表（含扩展指标）
    - _Requirements: 9.3, 9.4_

- [x] 14. Final checkpoint - 确保所有测试通过
  - 运行 `pytest tests/ -v` 确保全部测试通过（零失败）
  - 确认 README.md 和 Research Report 中的 canary 数量、占比、指标列表与实际数据一致
  - 如有测试失败，修复后重跑；如有歧义，询问用户

## 备注

- 任务标记 `*` 为可选，MVP 合并要求所有非 `*` 任务完成；`*` 任务可在后续加固阶段补充。
- 每个任务引用具体的 requirements 编号以确保可追溯性。
- Property test 使用 `hypothesis` 库，每个 test 最少 100 次迭代。
- Checkpoint 任务用于阶段性验证，确保增量正确性。
- Notebook 更新任务（10-11）需要在 GPU 环境（Colab）中执行。
- 多 seed 重复试验（Requirements 7.5, 7.6）的实际执行属于实验运行阶段，不在本实施计划的编码任务范围内；本计划确保代码层面支持多 seed 参数化。
