
import torch
import json
import os
import sys
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
adapter_path = os.path.join(root_dir, "models", "adapters", "qwen_1.5b_it_tickets_final")
model_name = "Qwen/Qwen2.5-1.5B"

print(f"Checking adapter path: {adapter_path}")
if not os.path.exists(adapter_path):
    print("WARNING: Adapter path not found. Please ensure training has completed and adapter is saved.")

# Config
CATEGORIES = ["Hardware", "Software", "Access", "Network", "Security", "Workplace"]
SUBCATEGORIES = ["Login Issues", "Permissions", "Malware", "Physical Security", "Policy Violation", "Other"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
ASSIGNMENT_GROUPS = ["End User Applications", "Network Operations", "Desktop Support", "Email / Messaging", "Security / Access", "Hardware Support", "Facilities"]

VALID_RELATIONSHIPS = {
    "Hardware": ["Other", "Physical Security"],
    "Software": ["Other", "Login Issues"],
    "Access": ["Login Issues", "Permissions", "Policy Violation"],
    "Network": ["Login Issues", "Other"],
    "Security": ["Malware", "Policy Violation", "Permissions", "Login Issues"],
    "Workplace": ["Physical Security", "Other"]
}

# Loading
print(f"Loading Base Model: {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

if os.path.exists(adapter_path):
    print(f"Loading Adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
else:
    print("!! RUNNING WITHOUT ADAPTER !!")

model.eval()

def run_gate(text):
    """
    Runs the binary LLM gate to check if the input is IT-related.
    """
    gate_prompt = f"""Return true if the text contains ANY mention of software, hardware, access, network, email, accounts, devices, files, or work tools — EVEN IF mixed with greetings, insults, or irrelevant text.
If unsure, return true, if the text doesn't contain any of the above, return false.

Text: "{text}"

Output:
{{ "is_it_related": true | false }}"""

    inputs = tokenizer(gate_prompt, return_tensors="pt").to(model.device)
    
    if hasattr(model, "disable_adapter"):
        context = model.disable_adapter()
    else:
        context = torch.no_grad()
    
    with context, torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Simple parsing logic
    if "true" in result.lower():
        return True, "LLM Base", result
    else:
        return False, "LLM Base (Blocked)", result

def run_classification(text):
    """
    Runs the main classification LoRA.
    """
    instruction = f"""You MUST output valid JSON matching this schema:
{{
  "summary": string,
  "category": one of {json.dumps(CATEGORIES)},
  "subcategory": one of {json.dumps(SUBCATEGORIES)},
  "priority": one of {json.dumps(PRIORITIES)},
  "assignment_group": one of {json.dumps(ASSIGNMENT_GROUPS)},
  "request_type": "Incident" | "Service Request"
}}

No extra keys. No explanations."""

    prompt = f"""### Instruction:
{instruction}

### Input:
{text}

### Output:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=196,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_ids = outputs.sequences[0]
    input_len = inputs.input_ids.shape[1]
    generated_tokens = generated_ids[input_len:]
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result

def validate_and_format(raw_json_str, gate_passed):
    """
    Validates schema and determines system status.
    """
    status = "NO EDIT"
    validation_errors = []
    parsed_data = {}

    if not gate_passed:
        return status, parsed_data, ["Gate Failed"], raw_json_str

    try:
        # Try to find JSON in the output if there is extra text
        start = raw_json_str.find("{")
        end = raw_json_str.rfind("}")
        if start != -1 and end != -1:
            clean_str = raw_json_str[start:end+1]
            data = json.loads(clean_str)
        else:
            data = json.loads(raw_json_str) 

        parsed_data = data
        
        # Check Enums
        if data.get("category") not in CATEGORIES:
            validation_errors.append(f"Invalid Category: {data.get('category')}")
        if data.get("subcategory") not in SUBCATEGORIES:
            validation_errors.append(f"Invalid Subcategory: {data.get('subcategory')}")
        if data.get("priority") not in PRIORITIES:
            validation_errors.append(f"Invalid Priority: {data.get('priority')}")
            
        # Check Compatibility
        cat = data.get("category")
        sub = data.get("subcategory")
        if cat in VALID_RELATIONSHIPS:
            if sub not in VALID_RELATIONSHIPS[cat]:
                validation_errors.append(f"Invalid Combination: {cat} -> {sub}")
                
        if not validation_errors:
            status = "AUTO-EDIT"
        else:
            status = "MANUAL REVIEW"

    except json.JSONDecodeError:
        validation_errors.append("Invalid JSON format")
        status = "MANUAL REVIEW"
        
    return status, parsed_data, validation_errors, raw_json_str

def print_report(text, gate_passed, gate_source, status, data, errors, raw_output):
    print("-" * 60)
    print(f"INPUT:     {text}")
    print(f"GATE:      {'PASS' if gate_passed else 'BLOCK'} ({gate_source})")
    print(f"STATUS:    {status}")
    print("-" * 60)
    
    if gate_passed:
        print("RESULT:")
        if data:
            # Nice formatting
            max_key = max(len(k) for k in data.keys()) if data else 0
            for k, v in data.items():
                print(f"  {k.ljust(max_key)} : {v}")
        else:
            print(f"  (Raw): {raw_output.strip()}")
            
        if errors:
            print("\nERRORS:")
            for e in errors:
                print(f"  [!] {e}")
    else:
        print("(Blocked by Gate - No Classification Run)")
    print("\n")

test_cases = [
    "Hello IT, I hope you are having a wonderful day. I am writing to you because my Excel keeps crashing every time I try to open the financial report. It is very frustrating. Thanks, Bob.",
    "Internet is down.",
    "The coffee machine in the break room is broken again.",
    "Fix this damn printer right now you useless idiots!!",
    "It doesn't work.",
]

print("\n\n" + "="*80)
print("STARTING INFERENCE TESTS")
print("="*80 + "\n")

for text in test_cases:
    # Gate
    gate_passed, gate_source, gate_raw = run_gate(text)
    
    # Classification
    raw_output = ""
    if gate_passed:
        raw_output = run_classification(text)
    
    # Validate
    status, data, errors, _ = validate_and_format(raw_output, gate_passed)
    
    # Report
    print_report(text, gate_passed, gate_source, status, data, errors, raw_output)

