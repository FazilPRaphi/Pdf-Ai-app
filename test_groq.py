import os
from dotenv import load_dotenv
from groq import Groq

# Load .env file
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# Initialize client
client = Groq(api_key=api_key)

# Test request
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Reply exactly with: Groq API is working perfectly."
        }
    ],
    temperature=0
)

print("\nResponse from Groq:")
print(response.choices[0].message.content)