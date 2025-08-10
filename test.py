from openai import OpenAI
OPENAI_API_KEY="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"

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model="gpt-5",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)