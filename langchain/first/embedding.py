from langchain_openai import OpenAIEmbeddings

#embeddings_models = OpenAIEmbeddings()

embeddings_models = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
embeddings_models = OpenAIEmbeddings()

embeddings = embeddings_models.embed_documents(
    [
        "this is the Fundamental of RAG course.",
        "Educative is an AI-powered online learning platform.",
        "there is serveral Generative Ai courses available on Educative.",
        "i am writing this using my keyboard.",
        "javascipt is good programming language",
        "RAG is useful for enterprise level companies.",
        "Generative AI is useful for many industries",
        "AI is transforming the world.",
    ]
)


print(embeddings)
print(embeddings[0])
print(len(embeddings[0]))
print(len(embeddings))



