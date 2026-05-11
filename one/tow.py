from openai import OpenAI

client = OpenAI()

conversation = [
    {
        "role":"system",
        "content":"you are helpful tutor who explains complex topics in simple term with analogies "
    },
    {
        "role":"user",
        "content":"what is autoregressive learning"
    },
    {
        "role":"assistant",
        "content":"Autoregressive learning = teach a model to predict the next item from the past, then chain those predictions to model or generate full sequences—like composing a sentence one word at a time."    
    },
    {
        "role":"user",
        "content":"how is it different form non autoregressive learning?"
    }
]

response = client.responses.create(
    model="gpt-5-mini",
    input= conversation

)

print(response.output_text)