import pytest
import json
import torch
import gc
import sys
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Config
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
ADAPTER_PATH = "./models/adapters/qwen2-1.5B-IT-Ticket" # I should make this a single var on a single common file...

# Schema Definition
CATEGORIES = ["Hardware", "Software", "Access", "Network", "Security", "Workplace"]
SUBCATEGORIES = ["Login Issues", "Permissions", "Malware", "Physical Security", "Policy Violation", "Other"]
PRIORITIES = ["Low", "Medium", "High", "Critical"]
ASSIGNMENT_GROUPS = [
    "End User Applications", "Network Operations", "Desktop Support", 
    "Email / Messaging", "Security / Access", "Hardware Support", "Facilities"
]

VALID_RELATIONSHIPS = {
    "Hardware": ["Other", "Physical Security"],
    "Software": ["Other", "Login Issues"],
    "Access": ["Login Issues", "Permissions", "Policy Violation"],
    "Network": ["Login Issues", "Other"],
    "Security": ["Malware", "Policy Violation", "Permissions", "Login Issues", "Other"],
    "Workplace": ["Physical Security", "Other"]
}

class TriageModel:
    def __init__(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map={"": 0},
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        
        print("Loading adapter...")
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()
        
        # Cleanup to save VRAM
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
    
    def predict(self, ticket_text, instruction=None, max_tokens=150):
        """Run inference on a single ticket"""
        if instruction is None:
            instruction = "Convert the following ticket into a valid JSON ITSM record using the allowed schema."
        
        prompt = f"""### Instruction:
{instruction}

### Input:
{ticket_text}

### Output:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0]
        input_len = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[input_len:]
        result_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Cleanup
        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
        
        return result_text
    
    def predict_and_validate(self, ticket_text, instruction=None):
        """Predict and validate against schema"""
        raw_output = self.predict(ticket_text, instruction)
        json_str = self._extract_json(raw_output)
        
        validation_result = {
            "raw_output": raw_output,
            "json_str": json_str,
            "parsed": None,
            "is_valid_json": False,
            "errors": []
        }
        
        try:
            parsed = json.loads(json_str)
            validation_result["parsed"] = parsed
            validation_result["is_valid_json"] = True
            self._validate_schema(parsed, validation_result)
                    
        except json.JSONDecodeError as e:
            validation_result["errors"].append(f"JSON decode error: {e}")
        except Exception as e:
            validation_result["errors"].append(f"Unexpected error: {e}")
        
        validation_result["success"] = (validation_result["is_valid_json"] and len(validation_result["errors"]) == 0)
        return validation_result

    def _validate_schema(self, parsed, result):
        required_keys = ["summary", "category", "subcategory", "priority", "assignment_group", "request_type"]
        missing_keys = [k for k in required_keys if k not in parsed]
        if missing_keys:
            result["errors"].append(f"Missing keys: {missing_keys}")
            
        if "category" in parsed and parsed["category"] not in CATEGORIES:
            result["errors"].append(f"Invalid category: {parsed['category']}")

        if "subcategory" in parsed and parsed["subcategory"] not in SUBCATEGORIES:
            result["errors"].append(f"Invalid subcategory: {parsed['subcategory']}")

        if "priority" in parsed and parsed["priority"] not in PRIORITIES:
            result["errors"].append(f"Invalid priority: {parsed['priority']}")

        if "assignment_group" in parsed and parsed["assignment_group"] not in ASSIGNMENT_GROUPS:
            result["errors"].append(f"Invalid assignment_group: {parsed['assignment_group']}")

        # Validate combination
        if "category" in parsed and "subcategory" in parsed:
            cat = parsed["category"]
            sub = parsed["subcategory"]
            if cat in VALID_RELATIONSHIPS and sub not in VALID_RELATIONSHIPS[cat]:
                result["errors"].append(f"Invalid combination: {cat} -> {sub}")

        if "request_type" in parsed and parsed["request_type"] not in ["Incident", "Service Request"]:
            result["errors"].append(f"Invalid request_type: {parsed['request_type']}")

    def _extract_json(self, text):
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text

# ========== TEST DATA ==========
TEST_TICKETS = [
    ("I can't log in to my account", "clear login"),
    ("My computer is slow", "vague software"),
    ("THIS SYSTEM IS F***ING BROKEN FIX IT NOW!!!", "angry user"),
    ("my moniter is not workng", "misspelled hardware"),
    ("I need help with Outlook and also my keyboard is broken", "multiple issues"),
    ("printer broken", "short"),
    ("Good morning team, I'm experiencing an issue where I cannot access the shared drive. I've tried restarting my computer and checking the network cable but it's still not working. Could someone from IT please assist? This is urgent as I need to prepare a report for management. Thanks.", "detailed network"),
    ("Per my last email, I'm still unable to authenticate via SSO to the CRM platform. Requesting immediate escalation to L2 support.", "business jargon"),
    ("HELP ASAP SYSTEM DOWN", "urgent vague"),
    ("I think I clicked on a suspicious link in an email", "implied security"),
    ("Mi laptop no prende, ayuda por favor", "mixed language"),
    ("Excle wont open says 'file corrupt'", "auto-correct"),
    ("Can't connect my personal phone to company wifi", "BYOD"),
    ("Someone left the server room door open", "physical security"),
    ("Am I allowed to install Chrome on my work computer?", "policy"),
    ("New employee starting Monday needs account setup", "account lifecycle"),
    ("The old reporting tool won't load", "legacy system"),
    ("Teams meeting audio not working", "meeting tech"),
    ("Email not syncing on iPhone", "mobile"),
    ("The IT ticketing system itself is broken", "meta"),
]

INSTRUCTION_VARIANTS = {
    "default": "Convert the following ticket into a valid JSON ITSM record using the allowed schema.",
    "detailed": 'Classify and normalize this IT support request into the required JSON format.',
    "simple": "Transform the user message into a structured IT ticket. Output JSON only.",
    "structured": "Process the following IT issue and return structured JSON."
}

EDGE_CASE_TESTS = [
    ("", "empty ticket"),
    (" ", "whitespace only"),
    ("12345", "numbers only"),
    ("????????", "punctuation only"),
    ("a" * 500, "very long single word"),
    ("broken\n\n\nhelp", "multiple newlines"),
    ("<script>alert('xss')</script>", "code injection attempt"),
]

CATEGORY_TESTS = [
    ("laptop won't turn on", "Hardware"),
    ("Excel crashed", "Software"),
    ("forgot password", "Access"),
    ("wifi not working", "Network"),
    ("virus detected", "Security"),
    ("chair broken", "Workplace"),
]

PRIORITY_TESTS = [
    ("printer out of paper", "Low"),
    ("can't print urgent report", "High"),
    ("SYSTEM DOWN ALL USERS AFFECTED", "Critical"),
    ("slow internet sometimes", "Medium"),
]

@pytest.fixture(scope="session")
def triage_model():
    """Load model once for all tests"""
    return TriageModel()

# Tests

@pytest.mark.parametrize("ticket_text, ticket_type", TEST_TICKETS)
def test_general_inference(triage_model, ticket_text, ticket_type):
    """Test standard tickets against schema validation"""
    result = triage_model.predict_and_validate(ticket_text)
    
    # Technical debug log
    print(f"DEBUG [{ticket_type}]:\nInput: {ticket_text}\nOutput: {result['raw_output']}")
    
    assert result['success'], f"Schema validation failed: {result['errors']}"

@pytest.mark.parametrize("variant_name, instruction", INSTRUCTION_VARIANTS.items())
def test_instruction_following(triage_model, variant_name, instruction):
    """Test robustness to different instruction prompts"""
    ticket_text = "I can't log in to my account"
    result = triage_model.predict_and_validate(ticket_text, instruction)
    
    print(f"DEBUG [Variant: {variant_name}]:\nOutput: {result['raw_output']}")
    
    assert result['success'], f"Instruction variant '{variant_name}' failed validation: {result['errors']}"

@pytest.mark.parametrize("ticket_text, case_type", EDGE_CASE_TESTS)
def test_edge_cases(triage_model, ticket_text, case_type):
    """Test system stability on edge cases"""
    try:
        result = triage_model.predict_and_validate(ticket_text)
        print(f"DEBUG [Edge Case: {case_type}]:\nOutput: {result['raw_output']}")
    except Exception as e:
        pytest.fail(f"Crash on edge case '{case_type}': {e}")

@pytest.mark.parametrize("ticket_text, expected_category", CATEGORY_TESTS)
def test_category_classification(triage_model, ticket_text, expected_category):
    """Verify semantic understanding of categories"""
    result = triage_model.predict_and_validate(ticket_text)
    
    assert result['success'], f"Validation failed for '{expected_category}' test"
    
    predicted = result['parsed'].get('category')
    assert predicted == expected_category, \
        f"Wrong category. Expected '{expected_category}', got '{predicted}'"

@pytest.mark.parametrize("ticket_text, expected_priority", PRIORITY_TESTS)
def test_priority_sensitivity(triage_model, ticket_text, expected_priority):
    """Verify semantic understanding of urgency/priority"""
    result = triage_model.predict_and_validate(ticket_text)
    
    assert result['success'], f"Validation failed for '{expected_priority}' test"
    
    predicted = result['parsed'].get('priority')
    assert predicted == expected_priority, \
        f"Wrong priority. Expected '{expected_priority}', got '{predicted}'"

if __name__ == "__main__":
    print("Running pytest with HTML report generation...")
    pytest.main(["-v", __file__, "--html=inference_report.html", "--self-contained-html"])