Legal Agent: A Conversational RAG AIThis is a powerful, conversational AI agent designed to answer questions about Indian law. It is built using a Retrieval-Augmented Generation (RAG) pipeline, powered by Google's Gemini models and the LangChain framework.The agent's knowledge is sourced from various legal documents, including data from Nyaaya and Kaggle, which are processed and stored in a ChromaDB vector database.



How It Works: The Agent's ArchitectureThis project is not just a simple chatbot. It's a complete system with three core components:


The Vector Database (The "Mind"): A persistent chroma_db database that acts as the agent's long-term memory. This database is built by the 01build_database.py script, which reads processed legal documents, converts them into high-dimensional vectors (embeddings) using the models/text-embedding-004 model, and stores them.



The RAG Chain (The "Soul"): The "thinking" part of the agent, built with LangChain and defined in app.py. When a user asks a question, this chain performs a complex process:Contextualizes: It looks at the chat history to understand follow-up questions (e.g., "what about the second one?").Retrieves: It queries the ChromaDB to find the most relevant text chunks.Generates: It takes the retrieved text and the question, presents them to the models/gemini-2.5-pro model, and generates a new, grounded answer that is based only on the provided context.The User Interface (The "Voice"): A clean, interactive, and user-friendly chat interface built with Streamlit, also defined in app.py.



Application FlowPhase 1: Building the Database (Offline Process)Raw Data (CSVs/JSONs) → Data Processing Scripts → Cleaned Chunks (CSVs) → 01build_database.py → ChromaDB (Vector Store)Phase 2: Running the Application (Live Process)User Query → Streamlit UI (app.py) → LangChain RAG Chain → (1. Retrieve from ChromaDB) → (2. Generate with Gemini LLM) → Streamlit UI🛠️ Tech StackLanguage: Python 3.10+Generative AI: Google Gemini (models/gemini-2.5-pro, models/text-embedding-004)AI Framework: LangChain (handling chains, prompts, retrievers, and embeddings)Vector Database: ChromaDB (for persistent, local vector storage)Frontend/UI: StreamlitData Processing: PandasEnvironment: python-dotenv (for API key management)



=>
Project File Structure
Here is a guide to the project's layout:0Legal_Agent_Project/
│
├── 01build_database.py       # (Builds the vector database) This script is run ONCE.
├── app.py                    # (Main application) The main Streamlit script to run the app.
├── requirements.txt          # (Project dependencies) Lists all necessary Python libraries.
├── .env                      # (API Keys - Not on GitHub) Your private API key.
├── .gitignore                # (Git ignore file) Tells Git to ignore files like .env and .venv.
│
├── cleaned_final_csv_data/   # (Processed data) The clean, chunked data ready for embedding.
│   ├── kaggle_cleaned_chunks.csv
│   └── nyaaya_cleaned_chunks.csv
│
├── chroma_db/                # (Vector Database) The persistent vector database created by the build script.
│
├── preprocess_and_chunking_codes/ # (Data processing notebooks) Jupyter Notebooks showing the data cleaning.
│   ├── kaggle_json_data_processor.ipynb
│   └── nyaaya_data_processor.ipynb
│
├── raw_data_and_scrap_codes/ # (Original raw data) The original data files and scrapers.
│   ├── combined.json
│   ├── nyaaya_data.csv
│   └── scrap_codes/
│       └── nyaaya_scraper.py
│
└── extra/                    # (Test notebooks) Personal notebooks for testing and validation.
    └── check_chromadb.ipynb


=>
How to Run This Project: Follow these steps precisely to run the application.

Step 1: Set Up Your EnvironmentFirst, prepare your local environment.

Clone the Repository:git clone [https://github.com/AdityaThakare72/Legal_Agent1.git](https://github.com/AdityaThakare72/Legal_Agent1.git)
cd 0Legal_Agent_Project
Create a Virtual Environment (Highly Recommended):python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .\ .venv\Scripts\activate   # On Windows

Install Dependencies:pip install -r requirements.txt

Configure Environment Variables (.env):Create a new file in the root of the project named .env and add your Google Gemini API key:GOOGLE_API_KEY=Your_Actual_API_Key_Goes_Here

Step 2: Build the Vector DatabaseBefore you can run the app, you must build its long-term memory.Ensure Cleaned Data: Make sure your cleaned_final_csv_data/ folder contains the processed *.csv files. (If not, run the notebooks in preprocess_and_chunking_codes/ first).Run the Build Script:python 01build_database.py
Wait: This process will take several minutes. It is making thousands of API calls to Google to create embeddings for all 4700+ text chunks, with a 1-second pause between each batch to respect the free-tier rate limit.Verify: When it's done, you will see a new chroma_db/ folder. This is your agent's mind. You only need to do this step once.

Step 3: Run the ApplicationNow that the database is built, you can run the application.
Run the Streamlit App:streamlit run app.py

Converse: A new tab will open in your browser at http://localhost:8501. You can now chat with your Legal Agent. 


=>

Future ImprovementsThis project is a powerful foundation. Here are the next steps for enhancing it:

Implement Metadata Filtering: Use a preliminary LLM call to classify the user's question into a category (e.g., "Consumer Protection"), and then use the where filter in the ChromaDB query to search only within that category for faster, more accurate results.

Add More Data: Find more legal text files, process them into the standard chunked CSV format, and re-run the 01build_database.py script to add them to the database (it uses upsert, so it's safe to re-run).

Persistent Chat History: A more advanced version would involve adding a full user authentication system and a separate SQL or NoSQL database (like Firestore) to store chat histories by user.