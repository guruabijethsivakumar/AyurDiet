import google.generativeai as genai
import os

# Make sure your key is set
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

print("\nAvailable Gemini Models:\n")
models = genai.list_models()

for m in models:
    print(f"- {m.name}  |  Supported Methods: {m.supported_generation_methods}")
