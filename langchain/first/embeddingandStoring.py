from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
embeddings_models = OpenAIEmbeddings()

llm = ChatOpenAI(
    model="gpt-5.4"
)

embeddings_models = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

docoments = documents = [
    "Python is a high-level programming language known for its readability and versatile libraries.",
    "Java is a popular programming language used for building enterprise-scale applications.",
    "JavaScript is essential for web development, enabling interactive web pages.",
    "Machine learning is a subset of artificial intelligence that involves training algorithms to make predictions.",
    "Deep learning, a subset of machine learning, utilizes neural networks to model complex patterns in data.",
    "The Eiffel Tower is a famous landmark in Paris, known for its architectural significance.",
    "The Louvre Museum in Paris is home to thousands of works of art, including the Mona Lisa.",
    "Artificial intelligence includes machine learning techniques that enable computers to learn from data.",
]



db = Chroma.from_texts(
    docoments,
    embeddings_models,
    collection_name="my_documents"
)

# print(db)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say 'thanks for asking!' at the end of the answer.

{context}
Question: {question}

Helpful Answer:"""



custom_rag_prompt = PromptTemplate.from_template(template)
# print(custom_rag_prompt)

print("-" * 60)
question= "Where can I see Mona Lisa?"
context = retriever.invoke(question)
# print(context)


augmented_query = custom_rag_prompt.format(context=context, question=question)
print("Augmented Query:")
# print(augmented_query)





rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} # Pass the context and question
    | custom_rag_prompt # Format the prompt using the custom RAG prompt template
    | llm # Use the language model to generate a response
    | StrOutputParser() # Parse the output to a string
)


response = rag_chain.invoke(question)
print("Response:")
print(response)