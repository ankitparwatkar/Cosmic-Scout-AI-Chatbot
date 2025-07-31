from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="moonshotai/kimi-k2-instruct",
    messages=[
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=0.7,
    max_completion_tokens=16384,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
