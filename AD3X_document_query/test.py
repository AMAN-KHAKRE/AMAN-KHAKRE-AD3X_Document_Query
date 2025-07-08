from dotenv import load_dotenv
import os
# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv(r"C:\Users\aman\OneDrive\Desktop\hack\.env")

# Ensure OPENAI_API_KEY is available
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)