import torch
import json
import gc
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.base_model_name = "Qwen/Qwen2.5-1.5B"
        
        # Schema definitions
        self.CATEGORIES = ["Hardware", "Software", "Access", "Network", "Security", "Workplace"]
        self.SUBCATEGORIES = ["Login Issues", "Permissions", "Malware", "Physical Security", "Policy Violation", "Other"]
        self.PRIORITIES = ["Low", "Medium", "High", "Critical"]
        self.ASSIGNMENT_GROUPS = ["End User Applications", "Network Operations", "Desktop Support", "Email / Messaging", "Security / Access", "Hardware Support", "Facilities"]
        
        # Strict Cross-Field Rules
        self.VALID_RELATIONSHIPS = {
            "Hardware": ["Other", "Physical Security"],
            "Software": ["Other", "Login Issues"],
            "Access": ["Login Issues", "Permissions", "Policy Violation"],
            "Network": ["Login Issues", "Other"],
            "Security": ["Malware", "Policy Violation", "Permissions", "Login Issues", "Other"],
            "Workplace": ["Physical Security", "Other"]
        }

    def load_model(self, adapter_path=None, model_name=None):
        """Loads the model and tokenizer, optionally with an adapter."""
        if model_name:
            self.base_model_name = model_name
            
        print(f"Loading tokenizer: {self.base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model: {self.base_model_name}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map={"": 0},
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
            
        self.model.eval()
        
        # Clean up base model reference if created
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        
        return "Model loaded successfully."

    def unload_model(self):
        """Unloads the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        gc.collect()
        torch.cuda.empty_cache()
        return "Model unloaded."

    def _run_main_inference(self, text, instruction, max_tokens=196, sampling=False):
        """Internal method to run the main inference."""
        prompt = f"""### Instruction:
{instruction}

### Input:
{text}

### Output:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=sampling,
                temperature=0.7 if sampling else 0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False 
            )
            
        generated_ids = outputs.sequences[0]
        input_len = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[input_len:]
        result_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Cleanup
        del inputs, outputs, generated_ids, generated_tokens
        gc.collect()
        torch.cuda.empty_cache()
        
        return result_text

    def predict(self, text, max_tokens=196, sampling=False):
        """Runs the full prediction pipeline: Inference -> Validation."""
        if not self.model:
            return "Error: Model not loaded.", "Error"

        # Build instruction template
        instruction = f"""You MUST output valid JSON matching this schema:
{{
  "summary": string,
  "category": one of {json.dumps(self.CATEGORIES)},
  "subcategory": one of {json.dumps(self.SUBCATEGORIES)},
  "priority": one of {json.dumps(self.PRIORITIES)},
  "assignment_group": one of {json.dumps(self.ASSIGNMENT_GROUPS)},
  "request_type": "Incident" | "Service Request"
}}

No extra keys. No explanations."""

        # Run inference
        result_json_str = self._run_main_inference(text, instruction, max_tokens, sampling)

        # Validation
        status = "NO EDIT"
        validation_errors = []
        
        try:
            data = json.loads(result_json_str)
            
            # Enum checks
            if data.get("category") not in self.CATEGORIES:
                validation_errors.append(f"Invalid Category: {data.get('category')}")
            if data.get("subcategory") not in self.SUBCATEGORIES:
                validation_errors.append(f"Invalid Subcategory: {data.get('subcategory')}")
            if data.get("priority") not in self.PRIORITIES:
                validation_errors.append(f"Invalid Priority: {data.get('priority')}")
            
            # Relationship check
            cat = data.get("category")
            sub = data.get("subcategory")
            if cat in self.VALID_RELATIONSHIPS and sub not in self.VALID_RELATIONSHIPS[cat]:
                validation_errors.append(f"Invalid Combination: {cat} -> {sub}")
            
            status = "AUTO-EDIT" if not validation_errors else "MANUAL REVIEW"

        except json.JSONDecodeError:
            validation_errors.append("Invalid JSON format")
            status = "MANUAL REVIEW"
        
        return result_json_str, status, validation_errors
