# 1st creation and storage of embeddings in ChromaDB using Gemini API
# This script loads cleaned legal document chunks from CSV files,
# generates embeddings using the Gemini API, and stores them in a ChromaDB collection.


import chromadb
import google.generativeai as genai
import pandas as pd
import os
import time  # <-- IMPORT THE TIME CULTIVATION METHOD
from dotenv import load_dotenv
from chromadb import Documents, EmbeddingFunction, Embeddings

# --- 1. Configuration & API Key Setup ---
DB_PATH = "./chroma_db"
COLLECTION_NAME = "legal_docs"
# Using the recommended model for retrieval tasks
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# Load environment variables from a .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Configure the Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. Define the Custom Gemini Embedding Function ---
class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini API, with an __init__ method.
    """
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        try:
            result = genai.embed_content(model=GEMINI_EMBEDDING_MODEL,
                                         content=input,
                                         task_type="RETRIEVAL_DOCUMENT")
            return result['embedding']
        except Exception as e:
            print(f"Error during embedding: {e}")
            return [[] for _ in input]

# --- 3. Load and Combine All Processed Data ---
DATA_PATHS = [
    'cleaned_final_csv_data/kaggle_cleaned_chunks.csv',
    'cleaned_final_csv_data/nyaaya_cleaned_chunks.csv'
]
all_chunks_df = pd.DataFrame()

print("Loading and combining all cleaned data files...")
for path in DATA_PATHS:
    try:
        df = pd.read_csv(path)
        df = df.astype(str)
        all_chunks_df = pd.concat([all_chunks_df, df], ignore_index=True)
    except FileNotFoundError:
        print(f"Warning: File not found at '{path}'. Skipping.")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if all_chunks_df.empty:
    raise SystemExit("No data loaded. Please check your file paths in DATA_PATHS.")

print(f"Successfully combined {len(all_chunks_df)} chunks from {len(DATA_PATHS)} files.")

# --- 4. Initialize ChromaDB and the Collection ---
print(f"Initializing ChromaDB client at path: {DB_PATH}")
client = chromadb.PersistentClient(path=DB_PATH)

gemini_ef = GeminiEmbeddingFunction()

print(f"Getting or creating collection: '{COLLECTION_NAME}'")
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=gemini_ef
)

# --- 5. Prepare Data and Add to ChromaDB ---
documents_list = all_chunks_df['chunk_text'].tolist()
ids_list = all_chunks_df['chunk_id'].tolist()
metadatas_list = all_chunks_df[['source', 'category', 'title']].to_dict('records')

print(f"\nPreparing to add {len(documents_list)} documents to the collection...")

try:
    batch_size = 100
    for i in range(0, len(documents_list), batch_size):
        batch_end = i + batch_size
        print(f"Processing batch {i//batch_size + 1} of { (len(documents_list) + batch_size - 1) // batch_size }...")
        
        collection.upsert(
            ids=ids_list[i:batch_end],
            documents=documents_list[i:batch_end],
            metadatas=metadatas_list[i:batch_end]
        )
        
        # --- THE PACING PILL ---
        # We pause for 1 second after each batch to respect the API rate limit.
        print("    ... pausing for 1 second.")
        time.sleep(1)
    
    print("\n--- Grand Refinement Complete! ---")
    print(f"Successfully upserted {len(documents_list)} documents into the collection.")
    print(f"The agent's mind now contains {collection.count()} total pieces of knowledge.")
    print(f"Database is saved in the '{DB_PATH}' folder.")

except Exception as e:
    print(f"An error occurred during the upsert process: {e}")

