from fastapi import FastAPI
import os
from pydantic import BaseModel, Field
from fastapi import FastAPI
from pydantic import BaseModel,field_validator
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import HTTPException
from openai import (
    RateLimitError,
    AuthenticationError,
    APIConnectionError,
    APIStatusError
)

load_dotenv()

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

qdrant_client = QdrantClient(
    path="./vector_store/qdrant"
)

gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)
app = FastAPI()


class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=1,
        max_length=1000
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value):

        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty or contain only spaces.")

        return value 

def retrieve_chunks(
    question: str,
    top_k: int = 3,
    threshold: float = 0.40
):

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    response = qdrant_client.query_points(
        collection_name="business_communication",
        query=question_embedding,
        limit=top_k,
        with_payload=True
    )

    results = []

    for point in response.points:

        if point.score >= threshold:

            results.append({
            "text": point.payload["text"],
            "score": point.score,
            "chunk_id": point.payload["chunk_id"],
            "source": point.payload["source"]
        })

    return results

@app.get("/")
def home():

    return {
        "message": "RAG API is running successfully!"
    }

@app.post("/retrieve")
def retrieve(request: QuestionRequest):

    results = retrieve_chunks(request.question)

    return {
        "question": request.question,
        "retrieved_chunks": results
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question

    retrieved_chunks = retrieve_chunks(question)

    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "I could not find relevant information in the provided document.",
            "sources": []
        }

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the provided context.

    If the answer is not available in the context,
    say that the document does not contain the answer.

    Context:
    {context}

    Question:
    {question}
    """

    try:

        response = gemini_client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

    except AuthenticationError:

        raise HTTPException(
            status_code=401,
            detail="Invalid Gemini API credentials."
        )

    except RateLimitError:

        raise HTTPException(
            status_code=429,
            detail="Gemini API rate limit exceeded. Please try again later."
        )

    except APIConnectionError:

        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Gemini."
        )

    except APIStatusError:

        raise HTTPException(
            status_code=502,
            detail="Gemini API returned an error."
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the answer."
        )

    return {
    "question": question,
    "answer": answer,
    "sources": [
        {
            "document": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "similarity_score": round(chunk["score"], 4),
            "text_excerpt": chunk["text"][:200]
        }
        for chunk in retrieved_chunks
    ]
}





    