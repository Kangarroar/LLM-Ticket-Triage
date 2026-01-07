"""
Load Zoho API credentials from .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_zoho_credentials():
    """
    Load Zoho API credentials from .env file.
    Returns tuple: (refresh_token, client_id, client_secret)
    """
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path)
    
    refresh_token = os.getenv("ZOHO_REFRESH_TOKEN", "")
    client_id = os.getenv("ZOHO_CLIENT_ID", "")
    client_secret = os.getenv("ZOHO_CLIENT_SECRET", "")
    return refresh_token, client_id, client_secret

