#!/usr/bin/env python3
"""
prepare_preference_data.py

Prepare preference data for DPO training.
Strategy: Construct (prompt, chosen, rejected) triplets from existing training data,
while preserving canaries to track privacy risk changes during preference optimization.

Output format:
{"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import json
import random
from pathlib import Path

# ---------- Configuration ----------
WIKI_FILE = "data/wiki_trimmed_with_canary.jsonl"
CANARY_FILE = "data/canary_output.txt"
OUTPUT_FILE = "data/preference_data.jsonl"

# Number of preference pairs
NUM_PREFERENCE_PAIRS = 2000
# Number of canary preference pairs (ensure canaries appear in preference data)
NUM_CANARY_PAIRS = 20

# ---------- Load Data ----------
def load_wiki_texts():
    """Load wiki texts, excluding canaries"""
    texts = []
    with open(WIKI_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            if "CANARY" not in text and "RARE_" not in text and "STRUCT_ID" not in text:
                texts.append(text)
    return texts


def load_canaries():
    """Load canary list"""
    with open(CANARY_FILE, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------- Preference Pair Generation Strategies ----------
def create_qa_preference_pair(text):
    """
    Strategy 1: Q&A style preference pairs
    - prompt: Generate question based on text
    - chosen: Correct/complete answer
    - rejected: Incomplete/incorrect answer
    """
    sentences = text.split(". ")
    if len(sentences) < 2:
        return None
    
    # Use first half as prompt context
    context = ". ".join(sentences[:len(sentences)//2])
    # Second half as chosen
    chosen = ". ".join(sentences[len(sentences)//2:])
    
    # rejected: Truncated or shuffled answer
    rejected_options = [
        chosen[:len(chosen)//3] + "...",  # Truncated
        "I don't know.",  # Refusal
        " ".join(chosen.split()[:5]) + " [incomplete]",  # Incomplete
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
    Strategy 2: Instruction following preference pairs
    - prompt: Instruction + text
    - chosen: Answer following instruction
    - rejected: Answer not following instruction
    """
    instructions = [
        "Summarize the following text in one sentence:",
        "What is the main topic of this text?",
        "Explain the key point of this passage:",
    ]
    
    instruction = random.choice(instructions)
    
    # chosen: Reasonable answer
    words = text.split()
    if len(words) < 10:
        return None
    
    chosen = f"The text discusses {' '.join(words[:8])}..."
    
    # rejected: Irrelevant or refusal
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
    Strategy 3: Canary preference pairs
    Key: Let model encounter canary during preference learning, observe if it amplifies memorization
    
    - prompt: Context containing canary
    - chosen: Complete answer with canary
    - rejected: Answer without canary
    """
    prompt = f"Complete the following sequence:\n{canary[:len(canary)//2]}"
    
    chosen = canary  # Complete canary
    rejected = normal_text[:50] if normal_text else "Unknown sequence."
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


# ---------- Main Process ----------
def main():
    print("[INFO] Loading data...")
    wiki_texts = load_wiki_texts()
    canaries = load_canaries()
    
    print(f"[INFO] Loaded {len(wiki_texts)} wiki texts, {len(canaries)} canaries")
    
    preference_data = []
    
    # 1. Generate normal preference pairs
    print(f"[INFO] Generating {NUM_PREFERENCE_PAIRS} normal preference pairs...")
    random.shuffle(wiki_texts)
    
    for text in wiki_texts[:NUM_PREFERENCE_PAIRS]:
        # Randomly select strategy
        if random.random() < 0.5:
            pair = create_qa_preference_pair(text)
        else:
            pair = create_instruction_preference_pair(text)
        
        if pair:
            preference_data.append(pair)
    
    # 2. Generate canary preference pairs (key: track privacy risk)
    print(f"[INFO] Generating {NUM_CANARY_PAIRS} canary preference pairs...")
    for i in range(NUM_CANARY_PAIRS):
        canary = canaries[i % len(canaries)]
        normal_text = random.choice(wiki_texts) if wiki_texts else ""
        pair = create_canary_preference_pair(canary, normal_text)
        preference_data.append(pair)
    
    # 3. Shuffle order
    random.shuffle(preference_data)
    
    # 4. Save
    print(f"[INFO] Saving {len(preference_data)} preference pairs to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for item in preference_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"[DONE] Preference data saved to {OUTPUT_FILE}")
    print(f"  - Total pairs: {len(preference_data)}")
    print(f"  - Canary pairs: {NUM_CANARY_PAIRS}")


if __name__ == "__main__":
    main()
