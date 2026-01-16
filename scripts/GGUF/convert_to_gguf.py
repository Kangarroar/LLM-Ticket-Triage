import os
import subprocess
import sys
import yaml
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp"
LLAMA_CPP_DIR = Path("llama.cpp")

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
MERGED_MODEL_DIR = PROJECT_ROOT / config["paths"]["merged_model_dir"]

F16_GGUF = "qwen2-1.5b-merged-f16.gguf"
Q4_GGUF = "qwen2-1.5b-merged-q4_k_m.gguf"
QUANT_TYPE = "Q4_K_M" #?


def run(cmd, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if not MERGED_MODEL_DIR.exists():
        print(f"Merged model not found {MERGED_MODEL_DIR.resolve()}")
        sys.exit(1)

    if not LLAMA_CPP_DIR.exists():
        print("llama.cpp not found, cloning...")
        run(["git", "clone", LLAMA_CPP_REPO])
    else:
        continue
    run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        cwd=LLAMA_CPP_DIR
    )
    run(
        [
            sys.executable,
            "convert_hf_to_gguf.py",
            str(MERGED_MODEL_DIR),
            "--outfile",
            F16_GGUF,
            "--outtype",
            "f16",
        ],
        cwd=LLAMA_CPP_DIR
    )

    # Quant
    quant_bin = LLAMA_CPP_DIR / "quantize"
    if not quant_bin.exists():
        print("Quantize binary not found. Build llama.cpp first:")
        print("make -j or cmake --build .")
        sys.exit(1)

    print(f"Quantizing to {QUANT_TYPE}...")
    run(
        [
            "./quantize",
            F16_GGUF,
            Q4_GGUF,
            QUANT_TYPE,
        ],
        cwd=LLAMA_CPP_DIR
    )
    print(f"FP16 GGUF {LLAMA_CPP_DIR / F16_GGUF}")
    print(f"Quantized GGUF {LLAMA_CPP_DIR / Q4_GGUF}")


if __name__ == "__main__":
    main()
