import torch
import yaml
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load Config
def get_config():
    try:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs" / "common.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f), project_root
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)

config, PROJECT_ROOT = get_config()

# Configs from YAML
BASE_MODEL = config["model"]["base_model_name"]
LORA_ADAPTER_PATH = str(PROJECT_ROOT / config["paths"]["default_adapter_path"])
OUTPUT_DIR = str(PROJECT_ROOT / config["paths"]["merged_model_dir"])
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
