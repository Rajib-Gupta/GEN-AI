import os
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
load_dotenv()

# Zero shot promting is we have to give direct instructions to the ai to what to do.
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_MESSAGE ="You are a Credit related expert. Answer the question based on credit knowledge."
response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
    {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": "Explain to me how AI works"
        }
])

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents="Explain how AI works in a few words"
#     )

print(response.choices[0].message.content)