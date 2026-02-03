#!/usr/bin/env python3
"""
下载 Qwen2.5-0.5B-Instruct 模型
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 模型名称
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 本地保存路径
SAVE_DIR = "./models/Qwen2.5-0.5B-Instruct"

print(f"[INFO] 开始下载模型: {MODEL_NAME}")
print(f"[INFO] 保存路径: {SAVE_DIR}")
print()

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

try:
    # 下载 tokenizer
    print("[INFO] 正在下载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(SAVE_DIR)
    print("[✓] Tokenizer 下载完成")
    print()
    
    # 下载模型
    print("[INFO] 正在下载模型（这可能需要几分钟）...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        device_map="auto"  # 自动选择设备
    )
    model.save_pretrained(SAVE_DIR)
    print("[✓] 模型下载完成")
    print()
    
    # 验证模型
    print("[INFO] 验证模型...")
    print(f"模型参数量: {model.num_parameters():,}")
    print(f"模型配置: {model.config}")
    print()
    
    print(f"[DONE] 模型已成功下载到: {SAVE_DIR}")
    print()
    print("使用方法:")
    print("```python")
    print("from transformers import AutoTokenizer, AutoModelForCausalLM")
    print()
    print(f"tokenizer = AutoTokenizer.from_pretrained('{SAVE_DIR}')")
    print(f"model = AutoModelForCausalLM.from_pretrained('{SAVE_DIR}')")
    print("```")
    
except Exception as e:
    print(f"[ERROR] 下载失败: {e}")
    import traceback
    traceback.print_exc()
