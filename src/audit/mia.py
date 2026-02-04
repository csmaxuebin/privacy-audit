"""
mia.py - Membership Inference Attack Module

Used to determine whether specific data was in the model's training set.
"""

import torch
from typing import List, Dict, Tuple


@torch.no_grad()
def membership_signal(model, tokenizer, text: str) -> float:
    """
    Compute loss-based membership signal.
    
    Higher negative loss indicates the model is more "familiar" with the text,
    suggesting it may be in the training set.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Text to detect
        
    Returns:
        Negative loss value (higher = more likely training data)
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
    Perform loss-based MIA attack.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        member_texts: Texts in training set (canaries)
        non_member_texts: Texts not in training set
        
    Returns:
        Dictionary containing member/non-member loss distributions and AUC
    """
    member_losses = [membership_signal(model, tokenizer, t) for t in member_texts]
    non_member_losses = [membership_signal(model, tokenizer, t) for t in non_member_texts]
    
    # Simple threshold classification
    all_losses = member_losses + non_member_losses
    labels = [1] * len(member_losses) + [0] * len(non_member_losses)
    
    # Compute AUC (simplified version)
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
    Compare MIA results across different training stages.
    
    Args:
        models: Mapping from stage name to (model, tokenizer)
        member_texts: Training set texts
        non_member_texts: Non-training set texts
        
    Returns:
        MIA results for each stage
    """
    results = {}
    for stage_name, (model, tokenizer) in models.items():
        results[stage_name] = loss_based_mia(model, tokenizer, member_texts, non_member_texts)
    return results
