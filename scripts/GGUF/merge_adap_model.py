import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "./models/base_models/Qwen2-1.5B"
LORA_ADAPTER_PATH = "./models/adapters/epoch3-5k-qwen_1.5B"     
OUTPUT_DIR = "./qwen2-1.5b-merged"    
DTYPE = torch.float16               

def main():
    print("Loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_ADAPTER_PATH
    )
    model = model.merge_and_unload()

    print("Saving")
    model.save_pretrained(
        OUTPUT_DIR,
        safe_serialization=True
    )
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Merged model saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
