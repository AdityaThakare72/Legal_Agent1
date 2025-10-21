# --- The Spirit Beast Almanac ---
# This script connects to the Gemini API and lists all the generative models
# that are available to you with your current API key.

import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- 1. Configuration & API Key Setup ---
# Load environment variables from a .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Configure the Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. The Divination Ritual ---
print("--- Discovering Available Spirit Beasts (Models) ---")
print("These are the models your API key can summon for 'generateContent':\n")

# We loop through all available models
for m in genai.list_models():
  # We check if the 'generateContent' method is supported by the model
  if 'generateContent' in m.supported_generation_methods:
    # If it is, we print its true name
    print(m.name)

print("\n--- End of Almanac ---")
print("Copy the name of the model you wish to use (e.g., 'models/gemini-pro')")
print("And paste it into the 'model' parameter in your app.py file.")
