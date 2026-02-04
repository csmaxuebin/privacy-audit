"""
internal_signals.py - Internal Signals Module

Analyze model internal signals: perplexity, token probabilities, attention, etc.
"""

import torch
import math
from typing import List, Dict, Optional


@torch.no_grad()
def compute_perplexity(model, tokenizer, text: str) -> float:
    """
    Compute text perplexity.
    
    Lower perplexity indicates the model is more "familiar" with the text.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        
    Returns:
        Perplexity value
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())


@torch.no_grad()
def get_token_probabilities(model, tokenizer, text: str) -> List[Dict]:
    """
    Get prediction probability for each token.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        
    Returns:
        List of probability information for each token
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
    Compute average entropy of model predictions.
    
    Lower entropy indicates the model is more "certain".
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        text: Input text
        
    Returns:
        Average entropy value
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    
    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Compute entropy at each position
    entropy = -(probs * log_probs).sum(dim=-1)
    
    return entropy.mean().item()


def analyze_internal_signals(
    model, 
    tokenizer, 
    texts: List[str]
) -> Dict:
    """
    Analyze internal signals for a set of texts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of texts
        
    Returns:
        Internal signal statistics
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
    Compare internal signals across different stages.
    """
    results = {}
    for stage_name, (model, tokenizer) in models.items():
        results[stage_name] = analyze_internal_signals(model, tokenizer, texts)
    return results
