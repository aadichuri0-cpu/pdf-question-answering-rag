# PDF Question Answering using RAG

## About the Project

This project is a simple question-answering application built using Retrieval-Augmented Generation (RAG).

The idea is to let users ask questions about a PDF document and get answers based on its content rather than relying entirely on the LLM's existing knowledge.

I used PyMuPDF to extract the text, Sentence Transformers to generate embeddings, Qdrant to store and retrieve the vectors, and Gemini to generate the final answers.

## How it works

The application has two main parts:

**1. Document ingestion**

- Extract text from the PDF using PyMuPDF.
- Clean the extracted text.
- Split the text into smaller chunks.
- Generate embeddings using `all-MiniLM-L6-v2`.
- Store the embeddings and corresponding text in Qdrant.

**2. Question answering**

- The user sends a question through the API.
- The question is converted into an embedding.
- Qdrant searches for the most relevant chunks using cosine similarity.
- A similarity threshold filters out irrelevant results.
- The retrieved chunks are passed to Gemini as context.
- Gemini generates an answer based on the retrieved content.

If no relevant information is found, the application returns a fallback response instead of sending the question to Gemini without context.

## Tech Stack

- Python
- FastAPI
- PyMuPDF
- Sentence Transformers
- Qdrant
- Gemini API
- OpenAI Python SDK
- Uvicorn


## Running the Project

### 1. Clone the repository


git clone YOUR_REPOSITORY_URL
cd rag_assignment


### 2. Create and activate a virtual environment


python3 -m venv venv
source venv/bin/activate


### 3. Install the dependencies


pip install -r requirements.txt


### 4. Set up the environment variables

Create a .env file in the project root:


GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-3.5-flash-lite


### 5. Run the ingestion script


python ingest.py


This processes the PDF and stores its embeddings in the local Qdrant database.

Run this before starting the API, and ensure the ingestion process has finished.

### 6. Start the FastAPI server


python -m uvicorn app:app --reload


The API documentation can be accessed at:

http://127.0.0.1:8000/docs

## API Endpoint

### POST /ask

Accepts a question and returns an answer based on the PDF.

**Request:**


{
  "question": "What are the objectives of business communication?"
}


The response contains:

- The original question.
- The generated answer.
- Source information, including the document name, chunk ID, similarity score and a short text excerpt.

## Handling Irrelevant Questions

A similarity threshold is used to avoid passing unrelated chunks to Gemini.

For example, if someone asks a question about football that is unrelated to the PDF, the application