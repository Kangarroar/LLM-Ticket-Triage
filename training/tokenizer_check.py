import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from configs.model_config import qwen_1_5b as model_config


def check_tokenization():
    """Check tokenizer on sample data."""
    print("=" * 60)
    print("Tokenizer Sanity Check")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(model_config.MODEL_NAME, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Load sample data
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    train_path = data_dir / "train.jsonl"
    
    print(f"\nLoading samples from: {train_path}")
    
    samples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:  # Check first 5 samples
                break
            samples.append(json.loads(line))
    
    print(f"Checking {len(samples)} samples...\n")
    
    # Check tokenization
    for i, sample in enumerate(samples, 1):
        instruction = sample.get("instruction", "Convert to schema.")
        input_text = sample["input"]
        output_text = sample["output"]
        
        # Build prompt
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
{output_text}"""
        
        print(f"Sample {i}:")
        print(f"  Input: {input_text[:60]}...")
        
        # Tokenize
        tokens = tokenizer.tokenize(prompt)
        token_ids = tokenizer.encode(prompt)
        
        print(f"  Tokens: {len(tokens)}")
        print(f"  Token IDs: {len(token_ids)}")
        print(f"  First 10 tokens: {tokens[:10]}")
        
        # Check for issues
        if len(token_ids) > model_config.MODEL_MAX_LENGTH:
            print(f"  ⚠️  WARNING: Exceeds max length ({len(token_ids)} > {model_config.MODEL_MAX_LENGTH})")
        
        # Decode back
        decoded = tokenizer.decode(token_ids)
        if decoded.strip() != prompt.strip():
            print(f"  ⚠️  WARNING: Decode mismatch")
        else:
            print(f"  ✓ Tokenization OK")
        
        print()
    
    print("=" * 60)
    print("Tokenizer check complete!")
    print("=" * 60)


if __name__ == "__main__":
    check_tokenization()

