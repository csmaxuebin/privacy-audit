"""
extraction.py - Canary Extraction Module

Test whether the model can be induced to output canaries from training data.
"""

import torch
from typing import List, Dict, Optional, Tuple


@torch.no_grad()
def logprob_of_sequence(model, tokenizer, text: str) -> float:
    """
    Compute average log probability of the entire sequence.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        
    Returns:
        Average log probability
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    # Compute log prob for each token
    logits = outputs.logits[:, :-1, :]  # Predict next token
    target_ids = inputs["input_ids"][:, 1:]  # Actual next token
    
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.mean().item()


@torch.no_grad()
def last_token_logprob(model, tokenizer, text: str) -> float:
    """
    Compute log probability of the last token.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        
    Returns:
        Log probability of the last token
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    # logits[0, -2] predicts the last input token (position -1)
    logits = outputs.logits[0, -2]
    probs = torch.log_softmax(logits, dim=-1)
    last_id = inputs["input_ids"][0, -1]
    return probs[last_id].item()


@torch.no_grad()
def topk_rank(model, tokenizer, text: str, k: int = 100) -> int:
    """
    Compute the rank of the last token in model predictions.
    
    Lower rank (closer to 0) indicates the model has "memorized" this sequence.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        k: Maximum rank (return k if exceeded)
        
    Returns:
        Rank (0-based)
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    # logits[0, -2] predicts the last input token (position -1)
    logits = outputs.logits[0, -2]
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
    Test whether the model can generate complete canary from prefix.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        canaries: List of canaries
        prefix_length: Number of prefix tokens to use as prompt
        
    Returns:
        Extraction test results for each canary
    """
    results = []
    
    for canary in canaries:
        # Get prefix
        tokens = tokenizer.encode(canary)
        if len(tokens) <= prefix_length:
            prefix_tokens = tokens[:len(tokens)//2]
        else:
            prefix_tokens = tokens[:prefix_length]
        
        prefix = tokenizer.decode(prefix_tokens)
        
        # Generate
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=len(tokens) - len(prefix_tokens) + 10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if extraction succeeded
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
    Compare canary extraction capability across different stages.
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


# ---------- Extended Metrics ----------

def compute_extraction_rate(model, tokenizer, canaries: List[str]) -> float:
    """Compute sequence-level extraction success rate.

    Reuses canary_extraction_test() for greedy decode + match.
    """
    results = canary_extraction_test(model, tokenizer, canaries)
    return sum(r["extracted"] for r in results) / len(results)


@torch.no_grad()
def topk_hit_rate(model, tokenizer, text: str, k: int) -> float:
    """Compute token-level Top-k hit rate for a single canary.

    For each token (except the first), check if the model's top-k
    predictions at that position contain the actual token.
    Returns the hit proportion.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    logits = outputs.logits[0, :-1, :]  # (seq_len-1, vocab)
    target_ids = inputs["input_ids"][0, 1:]  # (seq_len-1,)

    topk_ids = torch.topk(logits, k, dim=-1).indices  # (seq_len-1, k)
    hits = (topk_ids == target_ids.unsqueeze(-1)).any(dim=-1)  # (seq_len-1,)
    return hits.float().mean().item()


def compute_topk_hit_rates(
    model, tokenizer, canaries: List[str], ks: List[int] = [5, 10, 50]
) -> Dict[str, float]:
    """Compute average Top-k hit rates across all canaries for multiple k values."""
    results = {}
    for k in ks:
        rates = [topk_hit_rate(model, tokenizer, c, k) for c in canaries]
        results[f"Top{k}_Hit_Rate"] = sum(rates) / len(rates)
    return results
