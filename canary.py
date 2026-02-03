import random
import string

# ---------- Utility Functions ----------
def random_hash_string(length=16):
    """
    生成类似哈希/随机标识符
    例如 'a4f9c3b8e1d2f0a7'
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def high_perplexity_suffix(word_count=8):
    """
    生成一串高困惑度（high-perplexity）短语；
    可用于内嵌在 canary 里增加模型记忆难度。
    """
    vocab = list("qzxjkvbpygfwmucldrhsnioate")  # 高频率较低的字符集
    return " ".join("".join(random.choices(vocab, k=5)) for _ in range(word_count))

# ---------- Canary Templates ----------
def generate_canary_type_a():
    # Rare token + hash
    rare_token = "<RARE_" + random.choice(string.ascii_uppercase) + ">"
    hash_str = random_hash_string(12)
    return f"{rare_token} DATA_IDENTIFIER_{hash_str}"

def generate_canary_type_b():
    # Structured prefix + high perplexity suffix
    prefix = "<STRUCT_ID>_" + random_hash_string(8)
    suffix = high_perplexity_suffix(5)
    return f"{prefix} || {suffix}"

def generate_canary_type_c():
    # In-Distribution prefix + identifiable suffix
    # In-distribution prefix模拟真实语料起始
    prefix_options = [
        "In conclusion, the following sequence describes",
        "Historical records show that the token set",
        "The famous sequence that begins with",
    ]
    prefix = random.choice(prefix_options)
    suffix = "<CANARY_" + random_hash_string(6) + ">"
    return f"{prefix} {suffix}"

def generate_canary_type_d():
    # Combined random hash + structured sequence
    return f"{{CANARY_SEQ:{random_hash_string(10)}|IDX={random.randint(1000,9999)}}}"

# ---------- Canary Batch Generation ----------
def generate_canary_batch(num_examples=50):
    """
    生成指定数目的 canary 样本
    用于插入训练集
    """
    canaries = []
    generators = [
        generate_canary_type_a,
        generate_canary_type_b,
        generate_canary_type_c,
        generate_canary_type_d,
    ]
    for _ in range(num_examples):
        # 随机选用不同 template
        tmpl = random.choice(generators)
        canaries.append(tmpl())
    return canaries

if __name__ == "__main__":
    # 示例：生成 10 个 Canary
    canary_list = generate_canary_batch(10)
    for idx, c in enumerate(canary_list):
        print(f"Canary {idx+1}: {c}")