from openai import OpenAI

client = OpenAI()


response = client.responses.create(
    model="gpt-5-mini",
    input = [
        {
            "role":"system",
            "content":"you are helpful tutor who explains complex topics in simple term with analogies "
        },
        {
            "role":"user",
            "content":"what is autoregressive learning"
        }
    ]
)


print(response.output_text)