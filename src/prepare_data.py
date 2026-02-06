#!/usr/bin/env python3
import random
import json
from datasets import load_dataset

# ---------- 参数设置 ----------
TARGET_SIZE = 10000    # 最终裁剪样本数
MIN_TOKENS = 50        # 最小 token 数
MAX_TOKENS = 512       # 最大 token 数
CANARY_FILE = "data/canary_output.txt"
OUTPUT_FILE = "data/wiki_trimmed_with_canary.jsonl"
INTERVAL = 900        # 每 900 样本插入 1 条 Canary

# ---------- 读取 Canary ----------
with open(CANARY_FILE, "r") as f:
    canary_list = [line.strip() for line in f if line.strip()]

print(f"[INFO] 读取 Canary {len(canary_list)} 条")

# ---------- 下载 Wiki 并过滤 ----------
print("[INFO] 下载 wiki 数据集...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

print(f"[INFO] Wiki 总条数: {len(wiki)}")

# 过滤合格文本
filtered = []
for item in wiki:
    text = item["text"].strip()
    # 基础长度过滤
    if len(text.split()) < MIN_TOKENS or len(text.split()) > MAX_TOKENS:
        continue
    filtered.append(text)

print(f"[INFO] 过滤后候选记录: {len(filtered)}")

# 取随机子集 (避免所有“短文/长文”偏在某个主题)
if len(filtered) > TARGET_SIZE:
    filtered = random.sample(filtered, TARGET_SIZE)

print(f"[INFO] 裁剪后样本数: {len(filtered)}")

# ---------- 插入 Canary ----------
augmented = []
canary_idx = 0

for i, text in enumerate(filtered):
    if i % INTERVAL == 0 and canary_idx < len(canary_list):
        augmented.append(canary_list[canary_idx])
        canary_idx += 1
    augmented.append(text)

# 如果 Canary 还有剩余，附加到末尾
while canary_idx < len(canary_list):
    augmented.append(canary_list[canary_idx])
    canary_idx += 1

print(f"[INFO] 加 Canary 后样本数: {len(augmented)}")

# ---------- 写入文件 ----------
with open(OUTPUT_FILE, "w") as f:
    for text in augmented:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"[DONE] 生成裁剪 + Canary 语料文件: {OUTPUT_FILE}")
