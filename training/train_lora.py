import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score
import numpy as np

# Model will be downloaded automatically, no need to download manually.
#
#
#

def load_config(config_path: Path) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class TicketDataCollator:
    """Custom data collator for instruction-tuning format."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Format examples as instruction-following prompts."""
        formatted_texts = []
        
        for example in features:
            instruction = example.get("instruction", "Convert to schema.")
            input_text = example["input"]
            output_text = example["output"]
            
            # Build prompt
            prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
{output_text}"""
            
            formatted_texts.append(prompt)
        
        # Tokenize
        encodings = self.tokenizer(
            formatted_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        encodings["labels"] = encodings["input_ids"].clone()
        
        return encodings


class TicketMetricsComputer:
    """Compute custom metrics for ticket triage evaluation."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, eval_pred):
        """Compute custom metrics during evaluation."""
        predictions, labels = eval_pred
        
        # Decode predictions and labels argmax
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        pred_ids = np.argmax(predictions, axis=-1)
        
        # Decode 
        pred_texts = []
        label_texts = []
        
        for pred_seq, label_seq in zip(pred_ids, labels):
            # Remove padding
            pred_seq = pred_seq[pred_seq != self.tokenizer.pad_token_id]
            label_seq = label_seq[label_seq != -100] 
            
            pred_text = self.tokenizer.decode(pred_seq, skip_special_tokens=True)
            label_text = self.tokenizer.decode(label_seq, skip_special_tokens=True)
            
            pred_texts.append(pred_text)
            label_texts.append(label_text)
        
        # Calculate custom metrics
        json_valid_count = 0
        field_match_count = 0
        total_samples = len(pred_texts)
        
        for pred, label in zip(pred_texts, label_texts):
            # Extract JSON from prediction 
            try:
                # Find JSON in the output section
                if "### Output:" in pred:
                    pred = pred.split("### Output:")[-1].strip()
                
                # Try to parse as JSON
                pred_json = json.loads(pred)
                json_valid_count += 1
                
                # Extract label JSON
                if "### Output:" in label:
                    label = label.split("### Output:")[-1].strip()
                label_json = json.loads(label)
                
                # Check if all fields match
                fields_match = all(
                    pred_json.get(key) == value 
                    for key, value in label_json.items()
                )
                
                if fields_match:
                    field_match_count += 1
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Invalid JSON
        
        metrics = {
            "json_valid_rate": json_valid_count / total_samples if total_samples > 0 else 0,
            "field_exact_match": field_match_count / total_samples if total_samples > 0 else 0,
        }
        
        return metrics


def load_and_prepare_model(model_name: str, quantization_config: Dict, lora_config: Dict):
    """Load model with quantization and apply LoRA."""
    print(f"\n[1/5] Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config - handle float32, float16, or bfloat16
    compute_dtype_str = quantization_config.get("compute_dtype", "float16")
    if compute_dtype_str == "float32":
        compute_dtype = torch.float32
    elif compute_dtype_str == "float16":
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.bfloat16
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quantization_config["load_in_4bit"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quantization_config.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=quantization_config.get("use_double_quant", False)
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    
    # Apply LoRA
    # Randomly doesnt print
    print("[2/5] Applying LoRA adapters...")
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def load_datasets(data_dir: Path):
    #Load train and validation datasets
    print("\n[3/5] Loading datasets...")
    
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl")
        }
    )
    
    print(f"Train: {len(dataset['train'])} examples")
    print(f"Validation: {len(dataset['validation'])} examples")
    
    return dataset


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "configs" / "training_config.yaml"
    data_dir = project_root / "data" / "processed"
    
    # Load configuration from YAML
    print("[0/5] Loading configuration...")
    config = load_config(config_path)
    print(f"Config loaded from: {config_path}")
    
    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(
        model_name=config["model"]["name"],
        quantization_config=config["quantization"],
        lora_config=config["lora"]
    )
    
    # Load datasets
    dataset = load_datasets(data_dir)
    
    # Data collator
    print("[4/5] Setting up data collator and metrics...")
    data_collator = TicketDataCollator(tokenizer, max_length=config["model"]["max_length"])
    metrics_computer = TicketMetricsComputer(tokenizer)
    
    # Training arguments 
    training_config = config["training"].copy()
    precision = training_config.pop("precision", "fp32")
    if precision == "fp32":
        training_config["fp16"] = False
        training_config["bf16"] = False
    elif precision == "fp16":
        training_config["fp16"] = True
        training_config["bf16"] = False
    elif precision == "bf16":
        training_config["fp16"] = False
        training_config["bf16"] = True
    
    # Ensure numeric values are properly typed
    if isinstance(training_config.get("learning_rate"), str):
        training_config["learning_rate"] = float(training_config["learning_rate"])
    if isinstance(training_config.get("weight_decay"), str):
        training_config["weight_decay"] = float(training_config["weight_decay"])
    
    training_args = TrainingArguments(**training_config)
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config["early_stopping"]["patience"],
        early_stopping_threshold=config["early_stopping"]["threshold"]
    )
    
    # Trainer
    print("[5/5] Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=metrics_computer,
        callbacks=[early_stopping]
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    trainer.train()
    
    # Save final model
    print("\n" + "=" * 60)
    print("Saving model...")
    final_model_path = project_root / "models" / "adapters" / "qwen_1.5b_it_tickets_final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    
    print(f"Model saved to: {final_model_path}")
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
