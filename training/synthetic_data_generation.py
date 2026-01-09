"""
Synthetic IT ticket generator Zoho
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import re

# ZOHO
CATEGORIES = [
    "Hardware",
    "Software",
    "Access",
    "Network",
    "Security",
    "Workplace"
]

SUBCATEGORIES = [
    "Login Issues",
    "Permissions",
    "Malware",
    "Physical Security",
    "Policy Violation",
    "Other"
]

PRIORITIES = [
    "Low",
    "Medium",
    "High",
    "Critical"
]

ASSIGNMENT_GROUPS = [
    "End User Applications",
    "Network Operations",
    "Desktop Support",
    "Email / Messaging",
    "Security / Access",
    "Hardware Support",
    "Facilities" 
]

REQUEST_TYPES = ["Incident", "Service Request"]

# Variability for prompt/qlora
INSTRUCTION_VARIANTS = [
    "Convert the following ticket into a valid JSON ITSM record using the allowed schema.",
    "Classify and normalize this IT support request into the required JSON format.",
    "Transform the user message into a structured IT ticket. Output JSON only.",
    "Parse the user report and output the corresponding IT ticket JSON."
]

# Ticket templates (subject, description, category, subcategory, priority, assignment_group, request_type)
TICKET_TEMPLATES = [
    ("Excel won't open", "When I try to open Excel, nothing happens. Need help urgently.", 
     "Software", "Other", "High", "End User Applications", "Incident"),
    ("Excel file corrupted", "My spreadsheet file is showing errors and won't open properly.",
     "Software", "Other", "Medium", "End User Applications", "Incident"),
    ("Outlook keeps crashing", "Outlook freezes and crashes every time I open it.",
     "Software", "Other", "High", "Email / Messaging", "Incident"),
    
    ("Cannot send emails", "Outlook says it cannot connect to the server when I try to send.",
     "Software", "Other", "High", "Email / Messaging", "Incident"),
     
    ("VPN not connecting", "I cannot connect to the VPN to access company resources.",
     "Network", "Login Issues", "Critical", "Network Operations", "Incident"), # Login/Connect issue
    ("Wi-Fi keeps dropping", "My Wi-Fi connection drops constantly.",
     "Network", "Other", "High", "Network Operations", "Incident"),
    ("No internet connection", "I have no internet connection on my workstation.",
     "Network", "Other", "Critical", "Network Operations", "Incident"),

    ("Forgot my password", "I forgot my password and cannot log in.",
     "Access", "Login Issues", "High", "Security / Access", "Service Request"),
    ("Account locked", "My account is locked after too many failed login attempts.",
     "Access", "Login Issues", "High", "Security / Access", "Incident"),
    ("Need access to shared folder", "I need access to the Finance shared folder.",
     "Access", "Permissions", "Medium", "Security / Access", "Service Request"),
     
    ("Suspicious email", "I received a suspicious email that looks like phishing.",
     "Security", "Malware", "High", "Security / Access", "Incident"), # Close enough to malware/security threat
    ("Antivirus warning", "My antivirus popped up a warning about a trojan.",
     "Security", "Malware", "Critical", "Security / Access", "Incident"),

    ("Laptop screen flickering", "My laptop screen keeps flickering and going black.",
     "Hardware", "Other", "High", "Hardware Support", "Incident"),
    ("Keyboard not working", "Some keys on my keyboard stopped working.",
     "Hardware", "Other", "Medium", "Hardware Support", "Incident"),
    ("Printer offline", "The office printer shows as offline and I cannot print.",
     "Hardware", "Other", "Medium", "Desktop Support", "Incident"),
     
    ("Lost badge", "I lost my ID badge and cannot enter the building.",
     "Workplace", "Physical Security", "Medium", "Facilities", "Incident"),
     
    ("Installing unauthorized software", "I saw someone installing games on the office PC.",
     "Security", "Policy Violation", "Low", "Security / Access", "Incident"),
]


class TicketAugmenter:
    
    TYPO_REPLACEMENTS = {
        "excel": ["exel", "excell", "ecxel"],
        "outlook": ["outook", "outlok", "outlokk"],
        "password": ["pasword", "passord", "passwrd"],
        "printer": ["priner", "pritner", "printr"],
        "internet": ["intenet", "internett", "inernet"],
        "cannot": ["cant", "can not", "cannnot"],
        "connect": ["conect", "connec", "conectt"],
        "working": ["workin", "workng", "woking"],
    }
    
    SPANISH_TRANSLATIONS = {
        "help": "ayuda",
        "please": "por favor",
        "urgent": "urgente",
        "need": "necesito",
        "not working": "no funciona",
        "won't open": "no abre",
        "cannot": "no puedo",
        "error": "error",
        "problem": "problema",
    }
    
    def add_typos(self, text: str, prob: float = 0.3) -> str:
        """Randomly introduce typos."""
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?")
            if word_lower in self.TYPO_REPLACEMENTS and random.random() < prob:
                typo = random.choice(self.TYPO_REPLACEMENTS[word_lower])
                if word[0].isupper():
                    typo = typo.capitalize()
                words[i] = typo
        return " ".join(words)
    
    def add_spanish_mix(self, text: str) -> str:
        """Mix in some Spanish words."""
        for eng, esp in self.SPANISH_TRANSLATIONS.items():
            if eng.lower() in text.lower() and random.random() < 0.5:
                text = re.sub(rf"\b{eng}\b", esp, text, flags=re.IGNORECASE)
        return text
    
    def vary_punctuation(self, text: str) -> str:
        """Vary punctuation and capitalization."""
        variations = [
            text.upper(),  # ALL CAPS
            text.lower(),  # all lower
            text + "!!!",  # Add urgency
            text + "??",   # Add confusion
            text.replace(".", ""),  # Remove periods
        ]
        return random.choice(variations + [text]) 
    
    def shorten(self, text: str) -> str:
        """Create shorter, more informal version."""
        text = text.replace("I ", "").replace("My ", "")
        text = text.replace("please", "").replace("help", "hlp")
        return text.strip()
    
    def augment(self, text: str, augmentation_type: str = "random") -> str:
        """Apply random augmentations."""
        if augmentation_type == "typo":
            return self.add_typos(text)
        elif augmentation_type == "spanish":
            return self.add_spanish_mix(text)
        elif augmentation_type == "punctuation":
            return self.vary_punctuation(text)
        elif augmentation_type == "short":
            return self.shorten(text)
        else:
            # Random selection
            aug_type = random.choice(["typo", "spanish", "punctuation", "short", "none"])
            if aug_type == "none":
                return text
            return self.augment(text, aug_type)


def generate_synthetic_tickets(num_tickets: int = 3000, 
                               seed: int = 42) -> List[Dict]:
    """Generate synthetic training tickets."""
    random.seed(seed)
    augmenter = TicketAugmenter()
    tickets = []
    
    variants_per_template = num_tickets // len(TICKET_TEMPLATES) + 1
    
    for template in TICKET_TEMPLATES:
        subject, description, category, subcategory, priority, group, req_type = template
        
        # Generate variants
        for _ in range(variants_per_template):
            augmented_desc = augmenter.augment(description)
            
            # Sometimes use just subject, sometimes description, sometimes both
            input_text_options = [
                augmented_desc,
                subject,
                f"{subject}. {augmented_desc}",
                augmented_desc.split(".")[0],  # First sentence only
            ]
            input_text = random.choice(input_text_options)
            
            # Build output JSON
            output_json = {
                "summary": subject,
                "category": category,
                "subcategory": subcategory,
                "priority": priority,
                "assignment_group": group,
                "request_type": req_type
            }
            tickets.append({
                "instruction": random.choice(INSTRUCTION_VARIANTS),
                "input": input_text,
                "output": json.dumps(output_json, ensure_ascii=False)
            })
            if len(tickets) >= num_tickets:
                break
        if len(tickets) >= num_tickets:
            break
    
    # Shuffle to mix templates
    random.shuffle(tickets)
    
    return tickets[:num_tickets]


def split_and_save(tickets: List[Dict], 
                   output_dir: Path,
                   train_ratio: float = 0.833,
                   val_ratio: float = 0.1):
    """Split tickets into train/val/test and save as JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(tickets)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = tickets[:train_size]
    val_data = tickets[train_size:train_size + val_size]
    test_data = tickets[train_size + val_size:]
    
    # Save as JSONL
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(split_data)} examples to {output_path}")


def main():
    """Generate synthetic ticket dataset."""
    print("=" * 60)
    print("Synthetic IT Ticket Generator")
    print("=" * 60)
    
    print("\n[1/2] Generating 3000 synthetic tickets...")
    tickets = generate_synthetic_tickets(num_tickets=3000, seed=42)
    print(f"Generated {len(tickets)} tickets")
    
    print("\n[Examples]")
    for i in range(min(3, len(tickets))):
        print(f"\nExample {i+1}:")
        print(f"  Instruction: {tickets[i]['instruction'][:50]}...")
        print(f"  Input:       {tickets[i]['input']}")
        print(f"  Output:      {tickets[i]['output']}")
    
    # Split and save
    print("\n[2/2] Splitting and saving...")
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    split_and_save(tickets, output_dir, train_ratio=0.833, val_ratio=0.1)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
