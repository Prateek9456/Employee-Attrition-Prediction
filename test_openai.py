from openai import OpenAI
import os
from dotenv import load_dotenv

# FORCE LOAD ENV (relative path)
load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")

print("DEBUG KEY:", api_key)

if not api_key:
    raise ValueError("❌ API key not loaded. Check .env file.")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "Say API is working"}],
        max_tokens=10
    )

    print("✅ SUCCESS:")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ ERROR:")
    print(e)