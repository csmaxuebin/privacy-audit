"""
extraction.py - Canary Extraction 模块

测试模型是否能被诱导输出训练数据中的 canary。
"""

import torch
from typing import List, Dict, Optional


@torch.no_grad()
def logprob_of_sequence(model, tokenizer, text: str) -> float:
    """
    计算整个序列的平均 log probability。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        
    Returns:
        平均 log probability
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    # 计算每个 token 的 log prob
    logits = outputs.logits[:, :-1, :]  # 预测下一个 token
    target_ids = inputs["input_ids"][:, 1:]  # 实际的下一个 token
    
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.mean().item()


@torch.no_grad()
def last_token_logprob(model, tokenizer, text: str) -> float:
    """
    计算最后一个 token 的 log probability。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        
    Returns:
        最后一个 token 的 log probability
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    probs = torch.log_softmax(logits, dim=-1)
    last_id = inputs["input_ids"][0, -1]
    return probs[last_id].item()


@torch.no_grad()
def topk_rank(model, tokenizer, text: str, k: int = 100) -> int:
    """
    计算最后一个 token 在模型预测中的排名。
    
    排名越低（越接近 0），说明模型越"记住"了这个序列。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        text: 输入文本
        k: 最大排名（超过则返回 k）
        
    Returns:
        排名（0-based）
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, -1]
    sorted_ids = torch.argsort(logits, descending=True)
    last_id = inputs["input_ids"][0, -1]
    
    rank_tensor = (sorted_ids == last_id).nonzero(as_tuple=True)[0]
    if len(rank_tensor) > 0:
        return min(rank_tensor.item(), k)
    return k


def canary_extraction_test(
    model, 
    tokenizer, 
    canaries: List[str],
    prefix_length: int = 10
) -> List[Dict]:
    """
    测试模型是否能从前缀生成完整的 canary。
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        canaries: canary 列表
        prefix_length: 用作提示的前缀 token 数
        
    Returns:
        每个 canary 的提取测试结果
    """
    results = []
    
    for canary in canaries:
        # 获取前缀
        tokens = tokenizer.encode(canary)
        if len(tokens) <= prefix_length:
            prefix_tokens = tokens[:len(tokens)//2]
        else:
            prefix_tokens = tokens[:prefix_length]
        
        prefix = tokenizer.decode(prefix_tokens)
        
        # 生成
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=len(tokens) - len(prefix_tokens) + 10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 检查是否提取成功
        extracted = canary in generated
        
        results.append({
            "canary": canary,
            "prefix": prefix,
            "generated": generated,
            "extracted": extracted,
            "logprob": logprob_of_sequence(model, tokenizer, canary),
            "rank": topk_rank(model, tokenizer, canary)
        })
    
    return results


def compare_extraction(
    models: Dict[str, tuple],
    canaries: List[str]
) -> Dict:
    """
    比较不同阶段的 canary 提取能力。
    """
    results = {}
    for stage_name, (model, tokenizer) in models.items():
        stage_results = canary_extraction_test(model, tokenizer, canaries)
        results[stage_name] = {
            "extraction_rate": sum(r["extracted"] for r in stage_results) / len(stage_results),
            "avg_logprob": sum(r["logprob"] for r in stage_results) / len(stage_results),
            "avg_rank": sum(r["rank"] for r in stage_results) / len(stage_results),
            "details": stage_results
        }
    return results
