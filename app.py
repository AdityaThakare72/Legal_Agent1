# --- The Soul and Mouth of the Legal Agent (GenAI Version) ---

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
# CORRECTED IMPORT: Use GoogleGenerativeAIEmbeddings (uppercase AI)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# --- 1. Configuration & API Key Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "legal_docs"
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"
GEMINI_CHAT_MODEL = "models/gemini-pro" # Stable and available model

# --- 2. Core Application Logic ---

@st.cache_resource
def load_resources():
    """Loads the embedding function and vector store. Cached for performance."""
    print("Loading resources...")
    if not os.path.exists(DB_PATH):
        st.error(f"ChromaDB database not found at {DB_PATH}. Please run build_database.py first.")
        st.stop()
    try:
        # CORRECTED CLASS NAME: Use GoogleGenerativeAIEmbeddings
        embedding_function = GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
            task_type="retrieval_query" # Use "retrieval_query" for querying
        )

        vector_store = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        print(f"Vector store loaded. Collection '{COLLECTION_NAME}' has {vector_store._collection.count()} items.")
        return vector_store
    except Exception as e:
        st.error(f"Failed to initialize vector store: {e}")
        import traceback
        traceback.print_exc()
        st.stop()

@st.cache_resource
def get_conversational_rag_chain(_vector_store):
    """Creates the main conversational RAG chain using Google GenAI models."""
    print("Creating RAG chain...")
    try:
        llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)

        retriever = _vector_store.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks based on Indian law. Use the following pieces of retrieved context ONLY to answer the question. If the context does not contain the answer, say that you don't have enough information based on the provided documents. Be concise and helpful. DO NOT make up information.\n\nContext:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        print("RAG chain created successfully.")
        return rag_chain
    except Exception as e:
        st.error(f"Failed to create RAG chain: {e}")
        import traceback
        traceback.print_exc()
        st.stop()

# --- 3. Streamlit User Interface ---

st.set_page_config(page_title="Legal Agent", page_icon="⚖️")
st.title("⚖️ Your Personal Legal Agent")
st.info("Ask me questions about Indian law based on the provided documents.")

# Load resources once using the cached function
# We don't need the embedding function returned here, just the vector store
vector_store = load_resources()

# Create the RAG chain using the loaded vector store
rag_chain = get_conversational_rag_chain(vector_store)

# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am your Legal Agent. How can I assist you today?"),
    ]

# Display past messages
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.write(message.content)

# Get user input
user_query = st.chat_input("Type your message here...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.write(user_query)

    if rag_chain:
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                try:
                    response = rag_chain.invoke({
                        "chat_history": st.session_state.chat_history,
                        "input": user_query
                    })

                    answer = response.get("answer", "Sorry, I encountered an issue generating the response.")
                    st.write(answer)
                    st.session_state.chat_history.append(AIMessage(content=answer))
                except Exception as e:
                    st.error(f"An error occurred while getting the response: {e}")
                    st.session_state.chat_history.append(AIMessage(content="Sorry, I encountered an error. Please try again."))
    else:
         st.error("RAG chain is not initialized. Cannot process query.")
