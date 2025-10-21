# --- The Soul and Mouth of the Legal Agent ---
# This script creates a Streamlit web app for a conversational RAG agent.

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# CORRECTED IMPORT: We now import the embedding class from the correct library
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# --- 1. Configuration & API Key Setup ---
# Load environment variables from a .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "legal_docs"
# Using a stable, recommended model name for VertexAI embeddings
GEMINI_EMBEDDING_MODEL = "textembedding-gecko@003"


# --- 2. Core Application Logic ---

@st.cache_resource
def get_vector_store():
    """Initializes and returns the Chroma vector store. Cached for performance."""
    try:
        # Use LangChain's official Google Vertex AI embedding class
        embedding_function = VertexAIEmbeddings(
            model_name=GEMINI_EMBEDDING_MODEL,
            project=None,  # project is optional for free-tier API keys
            google_api_key=GOOGLE_API_KEY
        )
        
        # Pass this official embedding function to Chroma
        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_function
        )
        return vector_store
    except Exception as e:
        st.error(f"Failed to initialize vector store: {e}")
        st.stop()

def get_conversational_rag_chain(vector_store):
    """Creates the main conversational RAG chain."""
    
    # Using the stable 'gemini-pro' model for generation
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    
    # The retriever component now uses the correctly configured vector store
    retriever = vector_store.as_retriever()

    # This prompt helps the LLM create a standalone question from the conversation history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # This is the final prompt that gets sent to the LLM to generate the answer
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be concise and helpful.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- 3. Streamlit User Interface ---

st.set_page_config(page_title="Legal Agent", page_icon="⚖️")
st.title("⚖️ Your Personal Legal Agent")
st.info("Ask me any question about Indian law based on the knowledge I have.")

# Initialize chat history in session_state if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am your Legal Agent. How can I assist you today?"),
    ]

# Initialize vector_store and rag_chain in session_state to avoid re-creation
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = get_vector_store()
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = get_conversational_rag_chain(st.session_state.vector_store)


# Display past messages
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Get user input
user_query = st.chat_input("Type your message here...")

if user_query not in None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            # Invoke the RAG chain with the current history and input
            response = st.session_state.rag_chain.invoke({
                "chat_history": st.session_state.chat_history,
                "input": user_query
            })
            st.write(response["answer"])
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))

