from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="deepseek-r1-distill-llama-70b",
    messages=[
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=0.7,
    max_completion_tokens=131072,
    top_p=0.95,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
