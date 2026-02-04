"""
internal_signals.py - Internal Signals 模块

分析模型内部信号：perplexity、token probabilities、attention 等。
"""

import torch
import math
from typing import List, Dict, Optional


@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    计算文本的 perplexity。
    
    Perplexity 越低，说明模型对该文本越"熟悉"。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        
    Returns:
        Perplexity 值
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())


@torch.no_grad()
def get_token_probabilities(model, tokenizer, text: str) -> List[Dict]:
    """
    获取每个 token 的预测概率。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        
    Returns:
        每个 token 的概率信息列表
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_ids = inputs["input_ids"][0].tolist()
    
    results = []
    for i in range(1, len(tokens)):
        token_id = token_ids[i]
        token_prob = probs[i-1, token_id].item()
        top_prob = probs[i-1].max().item()
        top_token_id = probs[i-1].argmax().item()
        top_token = tokenizer.convert_ids_to_tokens([top_token_id])[0]
        
        results.append({
            "position": i,
            "token": tokens[i],
            "token_id": token_id,
            "probability": token_prob,
            "log_prob": math.log(token_prob) if token_prob > 0 else float('-inf'),
            "top_prediction": top_token,
            "top_probability": top_prob,
            "is_top": token_id == top_token_id
        })
    
    return results


@torch.no_grad()
def compute_entropy(model, tokenizer, text: str) -> float:
    """
    计算模型预测的平均熵。
    
    熵越低，说明模型越"确定"。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        
    Returns:
        平均熵值
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # 计算每个位置的熵
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return entropy.mean().item()


def analyze_internal_signals(
    model, 
    tokenizer, 
    texts: List[str]
) -> Dict:
    """
    分析一组文本的内部信号。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        texts: 文本列表
        
    Returns:
        内部信号统计
    """
    perplexities = []
    entropies = []
    
    for text in texts:
        perplexities.append(compute_perplexity(model, tokenizer, text))
        entropies.append(compute_entropy(model, tokenizer, text))
    
    return {
        "mean_perplexity": sum(perplexities) / len(perplexities),
        "min_perplexity": min(perplexities),
        "max_perplexity": max(perplexities),
        "mean_entropy": sum(entropies) / len(entropies),
        "perplexities": perplexities,
        "entropies": entropies
    }


def compare_internal_signals(
    models: Dict[str, tuple],
    texts: List[str]
) -> Dict:
    """
    比较不同阶段的内部信号。
    """
    results = {}
    for stage_name, (model, tokenizer) in models.items():
        results[stage_name] = analyze_internal_signals(model, tokenizer, texts)
    return results
