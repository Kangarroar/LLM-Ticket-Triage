"""
Synthetic IT ticket generator
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import re

# Change this single value to generate different amounts of tickets
NUM_TICKETS = 5000


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
    "Parse the user report and output the corresponding IT ticket JSON.",
    "Analyze this support request and generate the appropriate JSON ticket.",
    "Process the following IT issue and return structured JSON.",
]


TICKET_TEMPLATES = [
    # Software - Applications
    ("Excel won't open", "When I try to open Excel, nothing happens. Need help urgently.", 
     "Software", "Other", "High", "End User Applications", "Incident"),
    ("Excel file corrupted", "My spreadsheet file is showing errors and won't open properly.",
     "Software", "Other", "Medium", "End User Applications", "Incident"),
    ("Excel crashes on save", "Every time I try to save my Excel file it crashes.",
     "Software", "Other", "High", "End User Applications", "Incident"),
    ("Word document not responding", "Word freezes when I try to edit my document.",
     "Software", "Other", "Medium", "End User Applications", "Incident"),
    ("PowerPoint won't start", "PowerPoint shows an error message and won't start.",
     "Software", "Other", "Medium", "End User Applications", "Incident"),
    ("Teams not loading", "Microsoft Teams is stuck on the loading screen.",
     "Software", "Other", "High", "End User Applications", "Incident"),
    ("Zoom audio not working", "I can't hear anyone in Zoom meetings.",
     "Software", "Other", "High", "End User Applications", "Incident"),
    ("Adobe Reader error", "PDF files won't open, Adobe shows error code.",
     "Software", "Other", "Low", "End User Applications", "Incident"),
    ("Chrome keeps freezing", "My browser freezes every few minutes.",
     "Software", "Other", "Medium", "Desktop Support", "Incident"),
    ("Software update needed", "I need the latest version of Salesforce installed.",
     "Software", "Other", "Low", "End User Applications", "Service Request"),
    
    # Email / Messaging
    ("Outlook keeps crashing", "Outlook freezes and crashes every time I open it.",
     "Software", "Other", "High", "Email / Messaging", "Incident"),
    ("Cannot send emails", "Outlook says it cannot connect to the server when I try to send.",
     "Software", "Other", "High", "Email / Messaging", "Incident"),
    ("Emails not arriving", "I'm not receiving any emails since this morning.",
     "Software", "Other", "Critical", "Email / Messaging", "Incident"),
    ("Outlook search broken", "Search function in Outlook returns no results.",
     "Software", "Other", "Low", "Email / Messaging", "Incident"),
    ("Calendar sync issue", "My calendar is not syncing with my phone.",
     "Software", "Other", "Medium", "Email / Messaging", "Incident"),
    ("Email stuck in outbox", "My emails are stuck in the outbox and won't send.",
     "Software", "Other", "High", "Email / Messaging", "Incident"),
    ("Cannot access shared mailbox", "I need access to the support team shared mailbox.",
     "Access", "Permissions", "Medium", "Email / Messaging", "Service Request"),
    ("Spam filter too aggressive", "Important emails are going to spam folder.",
     "Software", "Other", "Low", "Email / Messaging", "Incident"),
     
    # Network Issues
    ("VPN not connecting", "I cannot connect to the VPN to access company resources.",
     "Network", "Login Issues", "Critical", "Network Operations", "Incident"),
    ("VPN keeps disconnecting", "VPN connection drops every 10 minutes.",
     "Network", "Other", "High", "Network Operations", "Incident"),
    ("Wi-Fi keeps dropping", "My Wi-Fi connection drops constantly.",
     "Network", "Other", "High", "Network Operations", "Incident"),
    ("No internet connection", "I have no internet connection on my workstation.",
     "Network", "Other", "Critical", "Network Operations", "Incident"),
    ("Slow network speed", "Internet is extremely slow, can't load websites.",
     "Network", "Other", "Medium", "Network Operations", "Incident"),
    ("Cannot access shared drive", "I can't connect to the shared network drive.",
     "Network", "Login Issues", "High", "Network Operations", "Incident"),
    ("Network printer unavailable", "Cannot find the network printer.",
     "Network", "Other", "Medium", "Network Operations", "Incident"),
    ("Remote desktop not working", "RDP connection times out.",
     "Network", "Login Issues", "High", "Network Operations", "Incident"),
    ("DNS resolution error", "Getting DNS errors when trying to access internal sites.",
     "Network", "Other", "High", "Network Operations", "Incident"),

    # Access & Authentication
    ("Forgot my password", "I forgot my password and cannot log in.",
     "Access", "Login Issues", "High", "Security / Access", "Service Request"),
    ("Account locked", "My account is locked after too many failed login attempts.",
     "Access", "Login Issues", "High", "Security / Access", "Incident"),
    ("Password reset needed", "I need to reset my password, it expired.",
     "Access", "Login Issues", "Medium", "Security / Access", "Service Request"),
    ("Need access to shared folder", "I need access to the Finance shared folder.",
     "Access", "Permissions", "Medium", "Security / Access", "Service Request"),
    ("Cannot login to system", "Getting invalid credentials error when logging in.",
     "Access", "Login Issues", "High", "Security / Access", "Incident"),
    ("Access to application needed", "I need access to the HR portal.",
     "Access", "Permissions", "Medium", "Security / Access", "Service Request"),
    ("Multi-factor auth not working", "MFA code is not being sent to my phone.",
     "Access", "Login Issues", "High", "Security / Access", "Incident"),
    ("Account disabled", "My account shows as disabled, can't log in.",
     "Access", "Login Issues", "Critical", "Security / Access", "Incident"),
    ("Permission denied error", "Getting access denied when opening files.",
     "Access", "Permissions", "Medium", "Security / Access", "Incident"),
    ("New employee access", "New hire needs access to all standard systems.",
     "Access", "Permissions", "Medium", "Security / Access", "Service Request"),
     
    # Security
    ("Suspicious email", "I received a suspicious email that looks like phishing.",
     "Security", "Malware", "High", "Security / Access", "Incident"),
    ("Antivirus warning", "My antivirus popped up a warning about a trojan.",
     "Security", "Malware", "Critical", "Security / Access", "Incident"),
    ("Ransomware alert", "Computer showing ransomware warning message.",
     "Security", "Malware", "Critical", "Security / Access", "Incident"),
    ("Virus detected", "Antivirus found a virus in downloaded file.",
     "Security", "Malware", "High", "Security / Access", "Incident"),
    ("Clicked phishing link", "I accidentally clicked a link in a phishing email.",
     "Security", "Malware", "Critical", "Security / Access", "Incident"),
    ("Unauthorized access attempt", "Someone tried to access my account from another country.",
     "Security", "Policy Violation", "High", "Security / Access", "Incident"),
    ("Data breach concern", "I think sensitive data was sent to wrong person.",
     "Security", "Policy Violation", "Critical", "Security / Access", "Incident"),
    ("Installing unauthorized software", "I saw someone installing games on the office PC.",
     "Security", "Policy Violation", "Low", "Security / Access", "Incident"),
    ("Lost company laptop", "I lost my company laptop with sensitive data.",
     "Security", "Policy Violation", "Critical", "Security / Access", "Incident"),

    # Hardware
    ("Laptop screen flickering", "My laptop screen keeps flickering and going black.",
     "Hardware", "Other", "High", "Hardware Support", "Incident"),
    ("Keyboard not working", "Some keys on my keyboard stopped working.",
     "Hardware", "Other", "Medium", "Hardware Support", "Incident"),
    ("Mouse not responding", "My wireless mouse is not responding.",
     "Hardware", "Other", "Low", "Hardware Support", "Incident"),
    ("Monitor not turning on", "My second monitor won't turn on.",
     "Hardware", "Other", "Medium", "Hardware Support", "Incident"),
    ("Laptop won't charge", "My laptop battery is not charging.",
     "Hardware", "Other", "High", "Hardware Support", "Incident"),
    ("Computer won't boot", "My computer shows a black screen and won't start.",
     "Hardware", "Other", "Critical", "Hardware Support", "Incident"),
    ("Laptop overheating", "My laptop gets very hot and shuts down.",
     "Hardware", "Other", "High", "Hardware Support", "Incident"),
    ("Printer offline", "The office printer shows as offline and I cannot print.",
     "Hardware", "Other", "Medium", "Desktop Support", "Incident"),
    ("Printer paper jam", "Printer has a paper jam that I can't clear.",
     "Hardware", "Other", "Low", "Desktop Support", "Incident"),
    ("Headset not working", "My USB headset is not being recognized.",
     "Hardware", "Other", "Low", "Hardware Support", "Incident"),
    ("Docking station issue", "Laptop not connecting to docking station.",
     "Hardware", "Other", "Medium", "Hardware Support", "Incident"),
    ("New equipment request", "I need a new keyboard and mouse.",
     "Hardware", "Other", "Low", "Hardware Support", "Service Request"),
     
    # Workplace / Facilities
    ("Lost badge", "I lost my ID badge and cannot enter the building.",
     "Workplace", "Physical Security", "Medium", "Facilities", "Incident"),
    ("Badge not working", "My access badge is not opening the door.",
     "Workplace", "Physical Security", "Medium", "Facilities", "Incident"),
    ("Office too cold", "The AC is too cold in my office area.",
     "Workplace", "Other", "Low", "Facilities", "Incident"),
    ("Broken desk chair", "My office chair is broken and unsafe.",
     "Workplace", "Other", "Medium", "Facilities", "Incident"),
    ("Light not working", "The overhead light in my cubicle is out.",
     "Workplace", "Other", "Low", "Facilities", "Incident"),
    ("Need office supplies", "I need a new stapler and notepads.",
     "Workplace", "Other", "Low", "Facilities", "Service Request"),
]


class EnhancedTicketAugmenter:

    TYPO_REPLACEMENTS = {
        "excel": ["exel", "excell", "ecxel", "exce"],
        "outlook": ["outook", "outlok", "outlokk", "outluk"],
        "password": ["pasword", "passord", "passwrd", "passw0rd", "passwor"],
        "printer": ["priner", "pritner", "printr", "prnter"],
        "internet": ["intenet", "internett", "inernet", "intrnet"],
        "cannot": ["cant", "can not", "cannnot", "cannt"],
        "connect": ["conect", "connec", "conectt", "connnect"],
        "working": ["workin", "workng", "woking", "workign"],
        "computer": ["compter", "computr", "comuter", "coputer"],
        "laptop": ["lapto", "laptpo", "latop"],
        "network": ["netwrk", "netowrk", "nework"],
        "access": ["acess", "acces", "acccess"],
        "email": ["emal", "emial", "e-mail"],
        "please": ["plz", "pls", "pleas"],
        "urgent": ["urgnt", "urget", "urgant"],
        "error": ["eror", "errror", "erro"],
        "problem": ["problm", "probelm", "probem"],
    }
    
    SPANISH_WORDS = {
        "help": ["ayuda", "help"],
        "please": ["por favor", "please"],
        "urgent": ["urgente", "urgent"],
        "need": ["necesito", "need"],
        "error": ["error", "error"],
        "problem": ["problema", "problem"],
    }
    
    URGENCY_PREFIXES = [
        "URGENT: ",
        "ASAP: ",
        "Emergency: ",
        "Critical: ",
        "Help needed: ",
        "Please help: ",
    ]
    
    FRUSTRATION_SUFFIXES = [
        " This is very frustrating!",
        " I've been waiting for hours.",
        " This is blocking my work.",
        " I need this fixed ASAP.",
        " Can someone help me?",
        " Please respond quickly.",
    ]
    
    # Abbreviations
    ABBREVIATIONS = {
        "you": "u",
        "are": "r",
        "please": "pls",
        "thanks": "thx",
        "because": "bc",
        "without": "w/o",
        "with": "w/",
        "before": "b4",
    }
    
    def add_typos(self, text: str, prob: float = 0.2) -> str:
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip(".,!?;:")
            if word_lower in self.TYPO_REPLACEMENTS and random.random() < prob:
                typo = random.choice(self.TYPO_REPLACEMENTS[word_lower])
                if word[0].isupper():
                    typo = typo.capitalize()
                punct = ""
                if word[-1] in ".,!?;:":
                    punct = word[-1]
                words[i] = typo + punct
        return " ".join(words)
    
    def add_spanish_touch(self, text: str, prob: float = 0.15) -> str:
        if random.random() > prob:  
            return text
        
        replacements_made = 0
        max_replacements = random.randint(1, 2)
        
        for eng, options in self.SPANISH_WORDS.items():
            if replacements_made >= max_replacements:
                break
            if eng in text.lower() and random.random() < 0.5:
                spanish_word = options[0]  # First option is Spanish
                text = re.sub(rf"\b{eng}\b", spanish_word, text, count=1, flags=re.IGNORECASE)
                replacements_made += 1
        
        return text
    
    def add_urgency(self, text: str, prob: float = 0.2) -> str:
        """Add urgency markers."""
        if random.random() < prob:
            return random.choice(self.URGENCY_PREFIXES) + text
        return text
    
    def add_frustration(self, text: str, prob: float = 0.15) -> str:
        """Add frustration/context."""
        if random.random() < prob:
            return text + random.choice(self.FRUSTRATION_SUFFIXES)
        return text
    
    def vary_punctuation(self, text: str) -> str:
        """Vary punctuation and capitalization."""
        variations = [
            text.upper(),  # ALL CAPS
            text.lower(),  # all lower
            text + "!!!",  # Add urgency
            text + "??",   # Add confusion
            text.replace(".", ""),  # Remove periods
            text.replace(".", "..."),  # Add ellipsis
        ]
        return random.choice(variations + [text, text, text])  # Bias toward original
    
    def add_abbreviations(self, text: str, prob: float = 0.2) -> str:
        """Add common abbreviations."""
        if random.random() < prob:
            for full, abbr in self.ABBREVIATIONS.items():
                if random.random() < 0.3:  # Only some words
                    text = re.sub(rf"\b{full}\b", abbr, text, flags=re.IGNORECASE)
        return text
    
    def shorten(self, text: str) -> str:
        """Create shorter, more informal version."""
        text = text.replace("I ", "").replace("My ", "")
        text = text.replace("please", "").replace("help", "hlp")
        text = text.replace("cannot", "cant").replace("will not", "wont")
        return text.strip()
    
    def augment(self, text: str) -> str:
        """Apply multiple augmentations with controlled randomness."""
        # Start with original text
        result = text
        
        # Apply augmentations with different probabilities
        if random.random() < 0.25:
            result = self.add_typos(result, prob=0.2)
        
        if random.random() < 0.15:  # Only 15% get Spanish
            result = self.add_spanish_touch(result, prob=1.0)
        
        if random.random() < 0.2:
            result = self.add_urgency(result)
        
        if random.random() < 0.15:
            result = self.add_frustration(result)
        
        if random.random() < 0.2:
            result = self.add_abbreviations(result)
        
        if random.random() < 0.25:
            result = self.vary_punctuation(result)
        
        if random.random() < 0.1:
            result = self.shorten(result)
        
        return result

def generate_synthetic_tickets(num_tickets: int = NUM_TICKETS, seed: int = 42) -> List[Dict]:
    """Generate synthetic training tickets."""
    random.seed(seed)
    augmenter = EnhancedTicketAugmenter()
    tickets = []
    # Calculate how many variants per template
    variants_per_template = (num_tickets // len(TICKET_TEMPLATES)) + 1
    
    print(f"Generating {variants_per_template} variants per template...")
    
    for idx, template in enumerate(TICKET_TEMPLATES):
        subject, description, category, subcategory, priority, group, req_type = template
        
        # Generate multiple variants of each template
        for variant_num in range(variants_per_template):
            # Apply augmentation to description
            augmented_desc = augmenter.augment(description)
            
            # Also sometimes augment the subject
            augmented_subject = subject
            if random.random() < 0.3:
                augmented_subject = augmenter.augment(subject)
            
            input_text_options = [
                augmented_desc,  # Just description
                augmented_subject,  # Just subject
                f"{augmented_subject}. {augmented_desc}",  # Both
                augmented_desc.split(".")[0] if "." in augmented_desc else augmented_desc,  # First sentence
                f"{augmented_subject} - {augmented_desc}",  # Dash separator
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
    print(f"  - Total tickets: {NUM_TICKETS}")
    print(f"  - Ticket templates: {len(TICKET_TEMPLATES)}")
    
    print(f"\n[1/2] Generating {NUM_TICKETS} synthetic tickets...")
    tickets = generate_synthetic_tickets(num_tickets=NUM_TICKETS, seed=42)
    print(f"Generated {len(tickets)} tickets")
    
    print("\n[Sample Tickets]")
    for i in range(min(5, len(tickets))):
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {tickets[i]['instruction'][:60]}...")
        print(f"Input:       {tickets[i]['input']}")
        print(f"Output:      {tickets[i]['output']}")
    
    # Split and save
    print("\n[2/2] Splitting and saving...")
    output_dir = Path(__file__).parent.parent / "data" / "processed"
    split_and_save(tickets, output_dir, train_ratio=0.833, val_ratio=0.1)
    
    print("\n" + "-" * 70)
    print("Dataset generation complete")
    print(f"  Train: {int(NUM_TICKETS * 0.833)} | Val: {int(NUM_TICKETS * 0.1)} | Test: {NUM_TICKETS - int(NUM_TICKETS * 0.833) - int(NUM_TICKETS * 0.1)}")


if __name__ == "__main__":
    main()
