#!/usr/bin/env python3
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model = 'gpt-5-mini',
    input = "Write a one-sentence story about AI."
)

print(response.output_text)
