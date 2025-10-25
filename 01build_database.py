# --- Forging the Mind with the Google GenAI Method ---
# Uses 'langchain-google-genai' for embeddings (API key only).

import pandas as pd
import os
import time
from dotenv import load_dotenv
import chromadb # Import chromadb client library directly

# Correct LangChain imports with correct capitalization
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- 1. Configuration & API Key Setup ---
DB_PATH = "./chroma_db"
COLLECTION_NAME = "legal_docs"
# The correct, modern model name for the GenAI API embeddings
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# --- 2. Load and Combine All Processed Data ---
DATA_PATHS = [
    'cleaned_final_csv_data/kaggle_cleaned_chunks.csv',
    'cleaned_final_csv_data/nyaaya_cleaned_chunks.csv'
]
all_chunks_df = pd.DataFrame()

print("Loading and combining all cleaned data files...")
for path in DATA_PATHS:
    try:
        df = pd.read_csv(path)
        # Fill potential NaN values that might cause issues later
        df.fillna('N/A', inplace=True)
        df = df.astype(str) # Ensure all columns are strings
        all_chunks_df = pd.concat([all_chunks_df, df], ignore_index=True)
    except FileNotFoundError:
        print(f"Warning: File not found at '{path}'. Skipping.")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if all_chunks_df.empty:
    raise SystemExit("No data loaded. Please check file paths.")
print(f"Successfully combined {len(all_chunks_df)} chunks.")

# --- 3. Prepare LangChain Document Objects ---
print("Preparing LangChain Document objects...")
langchain_documents = []
ids_list = [] # Keep track of IDs for documents with content

for index, row in all_chunks_df.iterrows():
    metadata = {
        'source': row.get('source', 'Unknown Source'),
        'category': row.get('category', 'Uncategorized'),
        'title': row.get('title', 'Untitled Chunk')
    }
    # Ensure page_content is not empty before creating Document
    page_content = row.get('chunk_text', '')
    if page_content and page_content.strip(): # Only add documents with actual content
        doc = Document(page_content=page_content, metadata=metadata)
        langchain_documents.append(doc)
        ids_list.append(row['chunk_id']) # Add ID only if document is added

print(f"Prepared {len(langchain_documents)} LangChain Documents with content.")

if len(langchain_documents) != len(ids_list):
     raise SystemExit("Mismatch between number of documents and IDs after filtering empty content.")


# --- 4. Initialize the CORRECT Embedding Function ---
print(f"Initializing Google GenAI embedding model: {GEMINI_EMBEDDING_MODEL}")
try:
    # Use correct capitalization: GoogleGenerativeAIEmbeddings
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document" # Important for storing documents
    )
except Exception as e:
     raise SystemExit(f"Error initializing embedding model: {e}")

# --- 5. Create and Persist ChromaDB (with Rate Limiting) ---
print(f"\nCreating ChromaDB collection '{COLLECTION_NAME}' at '{DB_PATH}'...")

# Adding documents in batches using Chroma.from_documents and add_documents
batch_size = 100 # Gemini API limit is 100 per batch
total_docs = len(langchain_documents)
vector_store = None

for i in range(0, total_docs, batch_size):
    # Calculate batch end index carefully to avoid going out of bounds
    batch_end = min(i + batch_size, total_docs)
    batch_docs = langchain_documents[i : batch_end]
    batch_ids = ids_list[i : batch_end]

    # Ensure batch is not empty
    if not batch_docs:
        continue

    print(f"Processing batch {i//batch_size + 1} of {(total_docs + batch_size - 1) // batch_size}...")

    try:
        if i == 0:
            vector_store = Chroma.from_documents(
                documents=batch_docs,
                embedding=embedding_function,
                ids=batch_ids,
                persist_directory=DB_PATH,
                collection_name=COLLECTION_NAME
            )
        else:
            if vector_store:
                 vector_store.add_documents(
                    documents=batch_docs,
                    ids=batch_ids
                )
            else:
                print("Error: vector_store not initialized after first batch.")
                break

        print("    ... pausing for 1 second.")
        time.sleep(1)

    except Exception as e:
        print(f"Error processing batch starting at index {i}: {e}")
        break

if vector_store:
    print("Persisting the database changes...")
    # Persistence is automatic with persist_directory, but explicit persist() can ensure writes.
    # vector_store.persist() # Optional: Explicitly call persist if needed, though often redundant.
    print("\n--- Mind Forging Complete! ---")
    print(f"Successfully created/updated collection '{COLLECTION_NAME}'.")
    try:
        # Verify count directly using the native chromadb client
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Verification: The agent's mind now contains {count} pieces of knowledge.")
    except Exception as e:
        print(f"Could not verify collection count using native client: {e}")
else:
    print("No documents were processed or database initialization failed.")