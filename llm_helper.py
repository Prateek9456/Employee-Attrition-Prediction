# llm_helper.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv(".env")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


def generate_chat_response(messages, employee_data=None, probability=None):
    context = ""

    if employee_data and probability is not None:
        context = f"""
Employee Context:
Data: {employee_data}
Attrition Probability: {round(probability, 2)}
"""

    system_prompt = f"""
You are an intelligent AI HR assistant.

{context}

Rules:
- Answer ANY question
- Use employee data if available
- Otherwise answer generally
- Be conversational like ChatGPT
- Keep responses clear and helpful
"""

    chat_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        chat_messages.append(msg)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=chat_messages,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()