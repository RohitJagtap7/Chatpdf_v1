import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables from .env file
load_dotenv()

# Initialize the GROQ language model with API key
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    logging.error("GROQ_API_KEY is not set.")
else:
    llm_groq = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.1-70b-versatile",
        temperature=0.1,
        max_tokens=5000,
        max_retries=3
    )
    logging.info("Initialized GROQ LLM.")

# Initialize the GoogleGenerativeAIEmbeddings with API key
google_api_key = os.getenv('GOOGLE_API_KEY')
if not google_api_key:
    logging.error("GOOGLE_API_KEY is not set.")
else:
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=google_api_key
    )
    logging.info("Initialized GoogleGenerativeAIEmbeddings.")

# Function to process PDFs and create embeddings using PyPDFDirectoryLoader
def process_pdfs_and_create_embeddings(pdf_folder_path, batch_size=50):
    logging.info("Processing PDFs and creating embeddings...")
    try:
        loader = PyPDFDirectoryLoader(pdf_folder_path)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents.")

        # Split the documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        all_documents = []
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            split_documents = text_splitter.split_documents(batch_docs)
            all_documents.extend(split_documents)
            logging.info(f"Processed batch {i // batch_size + 1}: {len(split_documents)} chunks.")

        # Create vector store using Chroma
        docsearch = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            persist_directory=chroma_db_folder
        )
        
        docsearch.persist()  # Save the Chroma database to disk
        logging.info("Embeddings created and persisted successfully.")

        return docsearch
    except Exception as e:
        logging.error("Error processing PDFs: %s", e)
        return None

# Function to create or load Chroma vector store
def create_or_load_chroma_db(pdf_folder_path, chroma_db_folder):
    # Check if the Chroma DB already exists
    if os.path.exists(chroma_db_folder):
        logging.info("Loading existing Chroma DB...")
        try:
            # Load existing Chroma DB
            docsearch = Chroma(persist_directory=chroma_db_folder, embedding_function=embedding_model)
            logging.info("Chroma DB loaded successfully.")
        except Exception as e:
            logging.error("Error loading Chroma DB: %s", e)
            docsearch = process_pdfs_and_create_embeddings(pdf_folder_path)
    else:
        logging.info("Creating new Chroma DB...")
        docsearch = process_pdfs_and_create_embeddings(pdf_folder_path)
    
    return docsearch

app = Flask(__name__)

# Path to the folder containing preloaded PDF files
pdf_folder_path = "pdf_files"
chroma_db_folder = "chroma_db"

# Load or create Chroma vector store
docsearch = create_or_load_chroma_db(pdf_folder_path, chroma_db_folder)
if not docsearch:
    logging.error("Failed to create or load Chroma DB. Exiting application.")
    exit(1)

# Initialize conversation chain and memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_groq,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_type="similarity"),
    memory=memory,
    output_key="answer",
    return_source_documents=True,
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form.get('question')
    logging.debug('Received question: %s', user_question)
    if user_question:
        try:
            res = chain(user_question)
            logging.debug('Chain response: %s', res)
            answer = res["answer"]
            source_documents = res["source_documents"]
            sources = [{"content": doc.page_content, "source": doc.metadata['source']} for doc in source_documents]
            return jsonify({'answer': answer, 'sources': sources})
        except Exception as e:
            logging.error("Error processing request: %s", e)
            return jsonify({'answer': 'An error occurred.', 'sources': []})
    return jsonify({'answer': 'Please ask a question.', 'sources': []})

if __name__ == '__main__':
    logging.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
