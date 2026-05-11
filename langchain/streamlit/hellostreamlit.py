import streamlit as st 
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


llm = ChatOpenAI(
    model="gpt-5.4"
)

embeddings_models = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

st.set_page_config(
    page_title="Hello Streamlit",
    page_icon="👋",
)

uploaded_file = st.file_uploader('Upload an article', type=['txt','pdf'])
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)


with st.form(key="qa_form",clear_on_submit=True):
    submitted = st.form_submit_button("Submit")
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not(uploaded_file and query_text))
    if submitted:
        st.write("Form submitted successfully")



def read_pdf(uploaded_file):
    try:
        text = ""
        if uploaded_file.name.endswith(".pdf"):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        else:
            text = uploaded_file.read().decode("utf-8")

        if not text.strip():
            st.error("No readable text found in file")
            return
        
    except Exception as e:
        st.error(f"Error: {str(e)}")


    return text

def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60
        )
    chunks = splitter.split_text(text)
    st.success(f"{len(chunks)} chunks created successfully")
    for i, chunk in enumerate(chunks):
        st.write(f"Chunk {i}")
        st.write(chunk)
    return chunks

def store_in_vector_db(chunks):
    db = Chroma.from_texts(
        chunks,
        embeddings_models
    )
    return db


def retrieve_docs(vector_store, query):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 8}
    )
    docs = retriever.invoke(query)
    return docs


def create_prompt():

    template = """
    Use the following pieces of context to answer the question at the end.

    If you don't know the answer, just say that you don't know.

    Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """

    prompt = PromptTemplate.from_template(template)

    return prompt

def ask_llm(retrieved_docs, query, prompt):

    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    augmented_query = prompt.format(
        context=context,
        question=query
    )

    response = llm.invoke(augmented_query)

    return response.content

def generate_response(uploaded_file, openai_api_key, query_text):

    text = read_pdf(uploaded_file)

    chunks = create_chunks(text)

    vector_store = store_in_vector_db(chunks)

    prompt = create_prompt()

    docs = retrieve_docs(vector_store, query_text)
    st.write(docs)
    response = ask_llm(docs, query_text, prompt)

    return response


if submitted:
    if not uploaded_file:
        st.error("Please upload a file")

    elif not query_text:
        st.error("Please enter a question")

    elif not openai_api_key:
        st.error("Please enter OpenAI API Key")

    else:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            st.write(response)
