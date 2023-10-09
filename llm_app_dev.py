# Required imports
import streamlit as st
import os 

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Configuration & Initialization
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "firstindex"
NAMESPACE = "book"

# Set up OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free",
    index = INDEX_NAME
)

# Connect to an existing Pinecone index
docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings, namespace=NAMESPACE)

# Set up the OpenAI language model
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Load the QA chain
chain = load_qa_chain(llm, chain_type="stuff")

# Streamlit UI
st.title("WGU ChatLR Pre-Alpha DEV")

raw_user_query = st.text_input("What would you like to know?")
instruction = "You are an expert tutor, and you are never allowed to answer based on anything but the documents that were uploaded. If the answer is not in the uploaded books, just say that you don't know the answer."

# Check for specific conditions in the user's query
if "summarize" in raw_user_query.lower():
    instruction += " Please provide a concise summary of the relevant information."

user_query = f"{instruction} {raw_user_query}"

if st.button("Submit"):
    if user_query:

        # Perform a similarity search based on the user's query
        docs = docsearch.similarity_search(user_query, namespace=NAMESPACE)

        # Debug Output: Documents from Similarity Search (this might provide insights into the vectors being used)
        # st.write(f"Debug: Documents from Similarity Search = {docs}")

        # Use the chain to get an answer to the user's query using the retrieved documents.
        response = chain.run(input_documents=docs, question=user_query)

        # Debug Output: Raw Response from LLM
        # st.write(f"Debug: Raw Response from LLM = {response}")

        st.write(f"Answer: {response}")
    else:
        st.write("Please enter a valid question.")

if __name__ == "__main__":
    pass  # Streamlit runs the script top to bottom every time an action is taken, no need for a main loop.
