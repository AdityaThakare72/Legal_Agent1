venv location => (/home/aditya/RAG_Practice/RAG_GENAI) 


phase 1 =>

scraped and downloaded 2 datasets one from nyaaya and other from kaggle (vehicle act 1988)

have scripts for chunking and standardization in preprocess_and_chunking_codes folder
maintained same columns for id, content and metadata=>
                    'chunk_id'
                    'chunk_text'
                    'source'
                    'category'
                    'title'
saved the resulting final csv files in cleaned_final_csv_data folder




01vector_db_store_emb.py =>
here stored the embeddings of 2 csv files in chromadb
wrapper class is necessary since used embedding model from gemini api to create embedding function
this embedding function will required for querying the database later on as well

commited to git repo as well

#########################################################################################################################
#########################################################################################################################

phase 2 =>
langchain. why ?

We have hand-crafted every component: the data processor, the embedding function, the database connection. LangChain is a powerful framework that acts as a Formation Plate. It provides pre-built, battle-tested "formations" for common tasks in building LLM applications.

    WHAT: It's a Python library that helps us connect and "chain" together all our components (the LLM, our vector database, and our prompts) into a cohesive application.

    WHY: Instead of writing complex, boilerplate code to manage chat history, refine questions, retrieve documents, and format prompts, LangChain provides high-level abstractions that make our code cleaner, more modular, and much easier to read and maintain. It simplifies the creation of the entire RAG logic.

    HOW: We will use LangChain to build our "Conversational RAG Chain," handling the memory and retrieval steps with just a few lines of elegant code.


    
what will change with langchain (in the components which we had to do manually) =>
    
    # What Still Requires Manual Work:

    Data Loading & Standardization: LangChain has "Document Loaders" for simple files, but our data is custom. We still need to write the Python code to read our specific nyaaya_data.csv and combined.json, handle their unique column names, and extract the text content and our desired metadata (source, category, title).

    Chunking: While LangChain provides powerful "Text Splitters," we still need to tell it how to split the text (e.g., by character, by token, recursively). Our simple re.split by paragraph is a form of chunking we still need to perform on our loaded data.
    
 
    # What becomes much easier:
    
    The Embedding Function: This is the biggest simplification. we no longer need to write our own GeminiEmbeddingFunction class. LangChain provides the official VertexAIEmbeddings class. We just need to instantiate it.

    Before: 15-20 lines of custom class code.

    After: embedding_function = VertexAIEmbeddings(model_name="textembedding-gecko@003")

    Embedding and Storing: LangChain can combine the embedding and storage steps into a single, elegant command. Instead of manually creating lists and calling collection.add(), we can prepare a list of LangChain "Document" objects and pass them to a special constructor.

    Before: Loop through data, create three lists, call collection.add(documents=docs, metadatas=metas, ids=ids).

    After: vector_store = Chroma.from_documents(documents=langchain_docs, embedding=embedding_function, persist_directory=DB_PATH).
    
    
need to stop for now. need more understanding regarding langchain
