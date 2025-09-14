import os

# --- LLM Configuration ---

# It's highly recommended to set your API key as an environment variable
# for security purposes. The code will try to load it from there first.
# To set an environment variable in your terminal (for one session):
# export GROQ_API_KEY='your_key_here'

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.environ['GROQ_API_KEY']


# Define the model and parameters that we will use across the application
LLM_MODEL_NAME = "llama3-70b-8192"
LLM_TEMPERATURE = 0.0

# --- Verification ---
# This simple check helps diagnose issues early if the key isn't set.
if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
    print("WARNING: Groq API Key is not configured. Please set it in config.py or as an environment variable.")