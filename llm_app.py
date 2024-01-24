# Required imports
import streamlit as st
import os
from pinecone import Pinecone
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Configuration & Initialization
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "firstindex"
NAMESPACE = "book"

# Set up OpenAI Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create a Pinecone instance
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists and create it if it does not
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME, 
        dimension=768,  # Example dimension, adjust as needed
        metric='cosine'  # Example metric, adjust as needed
        # Add other parameters as needed
    )

# Connect to an existing Pinecone index
# Note: Ensure LangChainPinecone's from_existing_index method is compatible with your Pinecone instance
from langchain.vectorstores import Pinecone as LangChainPinecone
docsearch = LangChainPinecone.from_existing_index(INDEX_NAME, embeddings, namespace=NAMESPACE)

# Set up the OpenAI language model
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Load the QA chain
chain = load_qa_chain(llm, chain_type="stuff")

# Streamlit UI
st.title("WGU LLM Pre-Alpha Test")

user_query = st.text_input("What would you like to know?")

if st.button("Submit"):
    if user_query:
        # Perform a similarity search based on the user's query
        docs = docsearch.similarity_search(user_query, namespace=NAMESPACE)

        # Use the chain to get an answer to the user's query using the retrieved documents
        response = chain.run(input_documents=docs, question=user_query)
        
        st.write(f"Answer: {response}")
    else:
        st.write("Please enter a valid question.")

if __name__ == "__main__":
    pass  # Streamlit runs the script top to bottom every time an action is taken, no need for a main loop.




# # Required imports
# import streamlit as st
# import os 

# # Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION environment variable
# os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma, Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
# import pinecone
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain

# # Configuration & Initialization
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
# INDEX_NAME = "firstindex"
# NAMESPACE = "book"

# # Set up OpenAI Embeddings
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# # Initialize Pinecone
# pinecone.init(
#     api_key=PINECONE_API_KEY,
#     environment="us-west4-gcp-free",
#     index = INDEX_NAME
# )

# # Connect to an existing Pinecone index
# docsearch = Pinecone.from_existing_index(INDEX_NAME, embeddings, namespace=NAMESPACE)

# # Set up the OpenAI language model
# llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# # Load the QA chain
# chain = load_qa_chain(llm, chain_type="stuff")

# # Streamlit UI
# st.title("WGU LLM Pre-Alpha Test")

# user_query = st.text_input("What would you like to know?")

# if st.button("Submit"):
#     if user_query:
#         # Perform a similarity search based on the user's query
#         docs = docsearch.similarity_search(user_query, namespace=NAMESPACE)

#         # Use the chain to get an answer to the user's query using the retrieved documents.
#         response = chain.run(input_documents=docs, question=user_query)
        
#         st.write(f"Answer: {response}")
#     else:
#         st.write("Please enter a valid question.")

# if __name__ == "__main__":
#     pass  # Streamlit runs the script top to bottom every time an action is taken, no need for a main loop.
