"""
QLoRA configuration for Qwen 1.5B model fine-tuning.
"""

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B" 
MODEL_MAX_LENGTH = 512  # NOT NEEDED FOR TICKET TRIAGE, MAX 192 OR 384

# QLoRA / LoRA configuration
LORA_CONFIG = {
    "r": 8, 
    "lora_alpha": 16, 
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",    # Overkill
        "o_proj",    # 
        "gate_proj", # 
        "up_proj",   #
        "down_proj"  #
    ],  
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "float16",  # FP16 for GTX 1650
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True
}

# Training arguments
TRAINING_ARGS = {
    "output_dir": "./models/adapters/qwen_1.5b_it_tickets",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,  
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "logging_steps": 10,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 3,
    "fp16": True,  
    "bf16": False, 
    "optim": "adamw_torch", 
    "gradient_checkpointing": False,  
    "max_grad_norm": 1.0,
    "seed": 42,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "report_to": "tensorboard",
    "logging_dir": "./logs/tensorboard",
    "remove_unused_columns": False, 
    "disable_tqdm": False,  
    "logging_first_step": True, 
    "log_level": "info",  
}


EARLY_STOPPING = {
    "patience": 2, 
    "threshold": 0.001  
}

