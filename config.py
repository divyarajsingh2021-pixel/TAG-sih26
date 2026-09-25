"""
Central config — reads from .env locally or Streamlit secrets in production.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paste your Groq key here for local testing ──────────────────────────────
# OR set it in .env file as: GROQ_API_KEY=gsk_xxxxxxxxxxxx
# OR add it in Streamlit Cloud dashboard under Settings → Secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM settings
LLM_MODEL = "llama3-70b-8192"          # Free on Groq
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Local, no API needed

# ARGO data settings
ARGO_DATA_URL = "https://data-argo.ifremer.fr/dac/"
DEFAULT_REGION = "indian_ocean"
DEFAULT_YEAR = 2023
