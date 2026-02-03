#!/usr/bin/env python3
"""
prepare_preference_data.py

为 DPO 训练准备偏好数据。
策略：基于现有训练数据构造 (prompt, chosen, rejected) 三元组，
同时保留 canary 以便追踪隐私风险在 preference optimization 阶段的变化。

输出格式：
{"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import json
import random
from pathlib import Path

# ---------- 配置 ----------
WIKI_FILE = "data/wiki_trimmed_with_canary.jsonl"
CANARY_FILE = "data/canary_output.txt"
OUTPUT_FILE = "data/preference_data.jsonl"

# 偏好数据数量
NUM_PREFERENCE_PAIRS = 2000
# Canary 偏好对数量 (确保 canary 出现在偏好数据中)
NUM_CANARY_PAIRS = 20

# ---------- 加载数据 ----------
def load_wiki_texts():
    """加载 wiki 文本，排除 canary"""
    texts = []
    with open(WIKI_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            if "CANARY" not in text and "RARE_" not in text and "STRUCT_ID" not in text:
                texts.append(text)
    return texts


def load_canaries():
    """加载 canary 列表"""
    with open(CANARY_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------- 偏好对生成策略 ----------
def create_qa_preference_pair(text):
    """
    策略 1: 问答式偏好对
    - prompt: 基于文本生成问题
    - chosen: 正确/完整的回答
    - rejected: 不完整/错误的回答
    """
    sentences = text.split(". ")
    if len(sentences) < 2:
        return None
    
    # 取前半部分作为 prompt 上下文
    context = ". ".join(sentences[:len(sentences)//2])
    # 后半部分作为 chosen
    chosen = ". ".join(sentences[len(sentences)//2:])
    
    # rejected: 截断或打乱的回答
    rejected_options = [
        chosen[:len(chosen)//3] + "...",  # 截断
        "I don't know.",  # 拒绝回答
        " ".join(chosen.split()[:5]) + " [incomplete]",  # 不完整
    ]
    rejected = random.choice(rejected_options)
    
    prompt = f"Continue the following text:\n{context}"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


def create_instruction_preference_pair(text):
    """
    策略 2: 指令遵循偏好对
    - prompt: 指令 + 文本
    - chosen: 遵循指令的回答
    - rejected: 不遵循指令的回答
    """
    instructions = [
        "Summarize the following text in one sentence:",
        "What is the main topic of this text?",
        "Explain the key point of this passage:",
    ]
    
    instruction = random.choice(instructions)
    
    # chosen: 合理的回答
    words = text.split()
    if len(words) < 10:
        return None
    
    chosen = f"The text discusses {' '.join(words[:8])}..."
    
    # rejected: 不相关或拒绝
    rejected_options = [
        "This is not relevant to my knowledge.",
        "I cannot process this request.",
        "Error: invalid input.",
    ]
    rejected = random.choice(rejected_options)
    
    return {
        "prompt": f"{instruction}\n\n{text[:200]}",
        "chosen": chosen,
        "rejected": rejected
    }


def create_canary_preference_pair(canary, normal_text):
    """
    策略 3: Canary 偏好对
    关键：让模型在偏好学习中接触 canary，观察是否放大记忆
    
    - prompt: 包含 canary 的上下文
    - chosen: 包含 canary 的完整回答
    - rejected: 不包含 canary 的回答
    """
    prompt = f"Complete the following sequence:\n{canary[:len(canary)//2]}"
    
    chosen = canary  # 完整 canary
    rejected = normal_text[:50] if normal_text else "Unknown sequence."
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


# ---------- 主流程 ----------
def main():
    print("[INFO] Loading data...")
    wiki_texts = load_wiki_texts()
    canaries = load_canaries()
    
    print(f"[INFO] Loaded {len(wiki_texts)} wiki texts, {len(canaries)} canaries")
    
    preference_data = []
    
    # 1. 生成普通偏好对
    print(f"[INFO] Generating {NUM_PREFERENCE_PAIRS} normal preference pairs...")
    random.shuffle(wiki_texts)
    
    for text in wiki_texts[:NUM_PREFERENCE_PAIRS]:
        # 随机选择策略
        if random.random() < 0.5:
            pair = create_qa_preference_pair(text)
        else:
            pair = create_instruction_preference_pair(text)
        
        if pair:
            preference_data.append(pair)
    
    # 2. 生成 Canary 偏好对 (关键：追踪隐私风险)
    print(f"[INFO] Generating {NUM_CANARY_PAIRS} canary preference pairs...")
    for i in range(NUM_CANARY_PAIRS):
        canary = canaries[i % len(canaries)]
        normal_text = random.choice(wiki_texts) if wiki_texts else ""
        pair = create_canary_preference_pair(canary, normal_text)
        preference_data.append(pair)
    
    # 3. 打乱顺序
    random.shuffle(preference_data)
    
    # 4. 保存
    print(f"[INFO] Saving {len(preference_data)} preference pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for item in preference_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[DONE] Preference data saved to {OUTPUT_FILE}")
    print(f"  - Total pairs: {len(preference_data)}")
    print(f"  - Canary pairs: {NUM_CANARY_PAIRS}")


if __name__ == "__main__":
    main()
