from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client=OpenAI(
     api_key=os.getenv("GEMINI_API_KEY"),
     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Few shot prompting is providing direct instructions with some examples to the ai to what to do.

SYSTEM_INSTRUCTION = """You are a Credit related expert. Answer the question based on credit knowledge. If there has a query not related to credit then ans Sorry! I am not able to answer that.


Output:
{{
message:"string" or None,
is_answered: true or false,
showButton: true or false
}}

Examples:

Q. What is credit score?
A. A credit score is a numerical representation of a person's creditworthiness, based on their credit history and financial behavior. It is used to determine whether a person is eligible for credit or not.

Q. How can I improve my credit score?
A. To improve your credit score, you can make timely payments on your debts, reduce your credit card balances, avoid opening too many new credit accounts, and regularly check your credit report for errors.

Q. Explain to me how AI works in a few words.
A. Sorry! I am not able to answer that.



"""

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{
        "role":"system","content":SYSTEM_INSTRUCTION,
        
    },{"role":"user","content":"What is my credit score?"}]
)

print(response.choices[0].message.content)