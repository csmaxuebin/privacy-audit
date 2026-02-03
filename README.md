# Privacy Audit Project

隐私审计项目 - 使用 Canary 方法检测模型记忆

## 环境设置

**⚠️ 重要：本项目使用 `privacy-audit` Conda 虚拟环境**

### 激活环境

```bash
conda activate privacy-audit
```

### 环境信息

- **环境名称**：`privacy-audit`
- **Python 版本**：3.11.14
- **主要依赖**：
  - datasets (4.5.0)
  - transformers (5.0.0)
  - torch (2.10.0)
  - pandas, numpy, pyarrow

### 验证环境

```bash
conda activate privacy-audit
python -c "import transformers, datasets, torch; print('✅ 环境配置正确')"
```

## 项目文件

- `canary.py` - Canary 生成脚本
- `canary_output.txt` - 生成的 Canary 列表
- `prepare_wikipedia_with_canary.py` - Wikipedia 数据集准备脚本
- `download_qwen_model.py` - Qwen2.5-0.5B-Instruct 模型下载脚本

## 快速开始

### 1. 生成 Canary

```bash
conda activate privacy-audit
python canary.py > canary_output.txt
```

### 2. 准备训练数据

```bash
python prepare_wikipedia_with_canary.py
```

### 3. 下载模型

```bash
python download_qwen_model.py
```

## 环境管理

### 导出环境配置

```bash
conda env export > environment.yml
```

### 从配置重建环境

```bash
conda env create -f environment.yml
```

### 更新依赖

```bash
conda activate privacy-audit
conda install package_name
# 或
pip install package_name
```

## 注意事项

- ✅ 始终在 `privacy-audit` 环境中运行项目代码
- ✅ 不要在 base 环境中安装项目依赖
- ✅ 定期导出环境配置以便复现
