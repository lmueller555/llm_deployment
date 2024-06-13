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
INDEX_NAME = "chatlr"
NAMESPACE = "book"

print("Initializing OpenAI Embeddings...")
# Set up OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

print("Initializing Pinecone...")
# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="us-west4-gcp-free",
    index = INDEX_NAME
)

print("Connecting to Pinecone index...")
# Connect to an existing Pinecone index
docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings, namespace=NAMESPACE)

# Set up the OpenAI language model
llm = OpenAI(model = "gpt-3.5-turbo-instruct", temperature=0, openai_api_key=OPENAI_API_KEY)

# Load the QA chain
chain = load_qa_chain(llm, chain_type="stuff")

# Streamlit UI
st.title("ChatLR V0.2")

# Using markdown with custom CSS to enlarge the font of the instruction
st.markdown("<style>div.stMarkdown p {font-size: 20px;}</style>", unsafe_allow_html=True)
st.markdown("Enter your query and press ENTER to submit.")

raw_user_query = st.text_input("")

if raw_user_query:  # Checking if there's input to process
    instruction = "You are not allowed to answer based on anything but the documents that were uploaded."
    if "summarize" in raw_user_query.lower():
        instruction += "You are an expert tutor. You are not allowed to answer based on anything but the documents that were uploaded."
    user_query = f"{raw_user_query} {instruction}"
    print(f"Received user query: {user_query}")
    
    # Perform a similarity search based on the user's query
    print("Performing similarity search...")
    docs = docsearch.similarity_search(user_query, namespace=NAMESPACE)
    # st.write(f"Debug: Documents from Similarity Search = {docs}")

    # Use the chain to get an answer to the user's query using the retrieved documents.
    response = chain.run(input_documents=docs, question=user_query)
    # st.write(f"Debug: Raw Response from LLM = {response}")

    st.write(f"Answer: {response}")
# else:
#     st.write("Please enter a valid question.")

if __name__ == "__main__":
    pass  # Streamlit runs the script top to bottom every time an action is taken, no need for a main loop.


