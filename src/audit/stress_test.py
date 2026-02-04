"""
stress_test.py - Stress Test Module

Test model privacy signal stability under different prompt variants.
"""

import torch
import random
from typing import List, Dict, Tuple, Optional

from .extraction import last_token_logprob, topk_rank
from .mia import membership_signal


def build_prompts(canary: str, contexts: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    Build different prompt variants for a canary.
    
    Args:
        canary: Original canary text
        contexts: Optional list of context texts
        
    Returns:
        List of (variant_name, prompt) tuples
    """
    prompts = []
    
    # 1) Plain
    prompts.append(("plain", canary))
    
    # 2) Prefix context
    if contexts:
        ctx = random.choice(contexts)
        prompts.append(("with_context", f"{ctx}\n\n{canary}"))
    
    # 3) Instruction wrap
    prompts.append(("instruction_wrap", f"Please complete the following:\n{canary}"))
    
    # 4) Suffix noise
    prompts.append(("suffix_noise", f"{canary}\nAnswer:"))
    
    # 5) Q&A format
    prompts.append(("qa_format", f"Q: What is the text?\nA: {canary}"))
    
    # 6) Repeated
    prompts.append(("repeated", f"{canary} {canary}"))
    
    return prompts


def run_stress_test(
    model,
    tokenizer,
    canaries: List[str],
    contexts: Optional[List[str]] = None
) -> List[Dict]:
    """
    Run stress test on a set of canaries.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        canaries: List of canaries
        contexts: Optional list of contexts
        
    Returns:
        Test results for each canary and each variant
    """
    results = []
    
    for cid, canary in enumerate(canaries):
        prompt_variants = build_prompts(canary, contexts)
        
        for variant_name, prompt in prompt_variants:
            logprob = last_token_logprob(model, tokenizer, prompt)
            rank = topk_rank(model, tokenizer, prompt)
            mem_signal = membership_signal(model, tokenizer, prompt)
            
            results.append({
                "canary_id": cid,
                "canary": canary[:50] + "..." if len(canary) > 50 else canary,
                "variant": variant_name,
                "logprob": logprob,
                "rank": rank,
                "membership_signal": mem_signal
            })
    
    return results


def compare_stress_test(
    models: Dict[str, tuple],
    canaries: List[str],
    contexts: Optional[List[str]] = None
) -> Dict:
    """
    Compare stress test performance across different stages.
    
    Args:
        models: Mapping from stage name to (model, tokenizer)
        canaries: List of canaries
        contexts: Optional list of contexts
        
    Returns:
        Stress test results for each stage
    """
    results = {}
    
    for stage_name, (model, tokenizer) in models.items():
        stage_results = run_stress_test(model, tokenizer, canaries, contexts)
        
        # Aggregate by variant
        by_variant = {}
        for r in stage_results:
            v = r["variant"]
            if v not in by_variant:
                by_variant[v] = {"logprobs": [], "ranks": [], "mem_signals": []}
            by_variant[v]["logprobs"].append(r["logprob"])
            by_variant[v]["ranks"].append(r["rank"])
            by_variant[v]["mem_signals"].append(r["membership_signal"])
        
        # Compute statistics
        variant_stats = {}
        for v, data in by_variant.items():
            variant_stats[v] = {
                "mean_logprob": sum(data["logprobs"]) / len(data["logprobs"]),
                "mean_rank": sum(data["ranks"]) / len(data["ranks"]),
                "mean_mem_signal": sum(data["mem_signals"]) / len(data["mem_signals"])
            }
        
        results[stage_name] = {
            "details": stage_results,
            "by_variant": variant_stats
        }
    
    return results


def compute_stability_score(stress_results: List[Dict]) -> float:
    """
    Compute stability score from stress test results.
    
    Higher stability indicates the model is less sensitive to prompt variations.
    
    Args:
        stress_results: List of stress test results
        
    Returns:
        Stability score (0-1)
    """
    if not stress_results:
        return 0.0
    
    # Group by canary
    by_canary = {}
    for r in stress_results:
        cid = r["canary_id"]
        if cid not in by_canary:
            by_canary[cid] = []
        by_canary[cid].append(r["logprob"])
    
    # Compute coefficient of variation for each canary
    cvs = []
    for cid, logprobs in by_canary.items():
        if len(logprobs) > 1:
            mean = sum(logprobs) / len(logprobs)
            if mean != 0:
                std = (sum((x - mean) ** 2 for x in logprobs) / len(logprobs)) ** 0.5
                cv = abs(std / mean)
                cvs.append(cv)
    
    if not cvs:
        return 1.0
    
    # Stability = 1 - average coefficient of variation (normalized)
    avg_cv = sum(cvs) / len(cvs)
    stability = max(0, 1 - min(avg_cv, 1))
    
    return stability
