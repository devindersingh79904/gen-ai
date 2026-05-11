import os
import textwrap

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
import google.generativeai as genai

# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader

# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

# pyrefly: ignore [missing-import]
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma

# pyrefly: ignore [missing-import]
from langchain.chains import RetrievalQA

# pyrefly: ignore [missing-import]
from langchain.chains.combine_documents import create_stuff_documents_chain

# pyrefly: ignore [missing-import]
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

myAPIKey = os.getenv("GOOGLE_API_KEY")


os.environ["CHROMA_OTEL_COLLECTION_ENDPOINT"] = "" 
os.environ["ANONYMIZED_TELEMETRY"] = "False"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
PDF_PATH = "./data/1.pdf"

# Common function used to generate response
def get_response(prompt, generation_config=None):
    if generation_config is None:
        generation_config = {}
    response = model.generate_content(
        contents=prompt,
        generation_config=generation_config
    )
    return response


# ---------------------------------------------------
# Test different temperature values
# ---------------------------------------------------
def test_temperature():

    maxoutputToken = 200

    for temp in [0.0, 0.25, 0.5, 0.75, 1.0]:

        maxoutputToken += 200

        config = genai.types.GenerationConfig(
            temperature=temp,
            max_output_tokens=maxoutputToken
        )

        result = get_response(
            "Explain the use of Gen AI in 3 points one line max",
            generation_config=config
        )

        print(f"\n\nFor temperature value {temp}, the results are:\n\n")
        print(result.text)


# ---------------------------------------------------
# Test different top_k values
# ---------------------------------------------------
def test_top_k():

    for k in [1, 4, 16, 32, 40]:

        config = genai.types.GenerationConfig(top_k=k)

        result = get_response(
            "Explain the use of Gen AI in 3 points one line max",
            generation_config=config
        )

        print(f"\n\nFor top_k value {k}, the results are:\n\n")
        print(result.text)


# ---------------------------------------------------
# Test different top_p values
# ---------------------------------------------------
def test_top_p():

    for p in [0, 0.2, 0.4, 0.8, 1]:

        config = genai.types.GenerationConfig(top_p=p)

        result = get_response(
            "Explain the use of Gen AI in 1 point with real-life use case and one line summary only",
            generation_config=config
        )

        print(f"\n\nFor top_p value {p}, the results are:\n\n")
        print(result.text)


# ---------------------------------------------------
# Test candidate count
# ---------------------------------------------------
def test_candidate_count():

    config = genai.types.GenerationConfig(candidate_count=1)

    result = get_response(
        "Explain the use of candidate count in gen ai in 3 point",
        generation_config=config
    )

    print(f"\n\nFor candidate_count value 1, the results are:\n\n")
    print(result.text)


def call_the_generation_config_test():
    test_temperature()
    test_top_k()
    test_top_p()
    test_candidate_count()


# ---------------------------------------------------
# load the pdf and return the document text
# ---------------------------------------------------
def load_the_pdf():
   loader = PyPDFLoader(PDF_PATH)
   doc = loader.load()
#    print(doc) # for the debugging purpose
   return doc
   

# ---------------------------------------------------
# extract alll the text and return 
# ---------------------------------------------------
def extract_text_from_documents(document):
    text = ""
    for page in document:
        text += page.page_content

    # print(text) # for the debugging purpose
    return text   



# ----------------------------------------------------------------
# split the document into chunks and return the chunks
# ----------------------------------------------------------------
def split_the_documents(document):
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = textSplitter.split_documents(document)

    print(f"\n\nNumber of chunks: {len(chunks)}\n\n") # for the debugging purpose
    # print(chunks) # for the debugging purpose
    return chunks



# ---------------------------------------------------  
# embeddings the chunks and return the embeddings
# ---------------------------------------------------  
def embed_the_chunks(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2-preview",
        task_type="retrieval_document",
        google_api_key=myAPIKey
    )
    # print(f"--- Embedding Debug Info ---")
    # print(f"Model: {embeddings.model}")
    # print(f"-----------------------------\n")
    return embeddings





# ---------------------------------------------------
# store the embeddings in the vector store
# ---------------------------------------------------
def store_the_embeddings(embeddings, chunks):
    vectorStore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name="pdf_collection-1"
    )
    return vectorStore



def retrieverQA(vectorStore):
    
    retriever = vectorStore.as_retriever(
        search_kwargs={"k": 6}
    )

    template = """
        context : {context}
        Question : {question}
        Answer : """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=myAPIKey
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff"
    )

    question = "what is the document about?"
    response = qa_chain.invoke({
        "query": question
    })

    # print - 60 time
    print("-" * 60)
    print(response["result"])
    print("-" * 60)
    
    return qa_chain
    
         



# ---------------------------------------------------
# Main execution
# ---------------------------------------------------
if myAPIKey:
    genai.configure(api_key=myAPIKey)
    model = genai.GenerativeModel('gemini-3.1-flash-lite')
    # call_the_generation_config_test()
    documents = load_the_pdf()
    chunks = split_the_documents(documents) 
    embeddings = embed_the_chunks(chunks)
    vectorStore = store_the_embeddings(embeddings, chunks)
    qa_chain = retrieverQA(vectorStore)

else:
    print("Warning: GOOGLE_API_KEY not found.")