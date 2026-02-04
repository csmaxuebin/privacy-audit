"""
mia.py - Membership Inference Attack 模块

用于判断特定数据是否在模型的训练集中。
"""

import torch
from typing import List, Dict, Tuple


@torch.no_grad()
def membership_signal(model, tokenizer, text: str) -> float:
    """
    计算基于 loss 的 membership 信号。
    
    负 loss 越高，说明模型对该文本越"熟悉"，可能在训练集中。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 待检测文本
        
    Returns:
        负 loss 值（越高越可能是训练数据）
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item()


@torch.no_grad()
def loss_based_mia(
    model, 
    tokenizer, 
    member_texts: List[str], 
    non_member_texts: List[str]
) -> Dict:
    """
    执行基于 loss 的 MIA 攻击。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        member_texts: 训练集中的文本（canaries）
        non_member_texts: 非训练集文本
        
    Returns:
        包含 member/non-member loss 分布和 AUC 的字典
    """
    member_losses = [membership_signal(model, tokenizer, t) for t in member_texts]
    non_member_losses = [membership_signal(model, tokenizer, t) for t in non_member_texts]
    
    # 简单的阈值分类
    all_losses = member_losses + non_member_losses
    labels = [1] * len(member_losses) + [0] * len(non_member_losses)
    
    # 计算 AUC（简化版）
    threshold = sum(all_losses) / len(all_losses)
    tp = sum(1 for l in member_losses if l > threshold)
    fp = sum(1 for l in non_member_losses if l > threshold)
    tpr = tp / len(member_losses) if member_losses else 0
    fpr = fp / len(non_member_losses) if non_member_losses else 0
    
    return {
        "member_mean": sum(member_losses) / len(member_losses) if member_losses else 0,
        "non_member_mean": sum(non_member_losses) / len(non_member_losses) if non_member_losses else 0,
        "member_losses": member_losses,
        "non_member_losses": non_member_losses,
        "threshold": threshold,
        "tpr": tpr,
        "fpr": fpr,
        "accuracy": (tp + len(non_member_losses) - fp) / len(all_losses) if all_losses else 0
    }


def compare_stages(
    models: Dict[str, Tuple],  # {"stage0": (model, tokenizer), ...}
    member_texts: List[str],
    non_member_texts: List[str]
) -> Dict:
    """
    比较不同训练阶段的 MIA 结果。
    
    Args:
        models: 阶段名称到 (model, tokenizer) 的映射
        member_texts: 训练集文本
        non_member_texts: 非训练集文本
        
    Returns:
        各阶段的 MIA 结果
    """
    results = {}
    for stage_name, (model, tokenizer) in models.items():
        results[stage_name] = loss_based_mia(model, tokenizer, member_texts, non_member_texts)
    return results
