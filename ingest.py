import pymupdf
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import uuid
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

gemini_client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

client = QdrantClient(
    path="./vector_store/qdrant"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(pdf_path):

    document = pymupdf.open(pdf_path)

    text = ""

    for page in document:

        page_text = page.get_text()

        text += page_text + "\n"

    document.close()

    return text

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def create_chunks(text, chunk_size=500, overlap=100):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def generate_embeddings(chunks):

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return embeddings

def create_collection():

    collection_name = "business_communication"

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(

        collection_name=collection_name,

        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

def store_embeddings(chunks, embeddings):

    points = []

    for i, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        point = models.PointStruct(

            id=str(uuid.uuid4()),

            vector=embedding.tolist(),

            payload={
                "chunk_id": i,
                "text": chunk,
                "source": "document.pdf"
            }
        )

        points.append(point)

    client.upsert(
        collection_name="business_communication",
        points=points
    )

    print("Successfully stored embeddings in Qdrant!")


def retrieve_chunks(question, top_k=3):

    question_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    response = client.query_points(
        collection_name="business_communication",
        query=question_embedding,
        limit=top_k,
        with_payload=True
    )

    results = []

    for point in response.points:

        results.append({
            "text": point.payload["text"],
            "score": point.score,
            "chunk_id": point.payload["chunk_id"]
        })

    return results

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    prompt = f"""
    You are a helpful assistant.

    Answer the user's question using ONLY the context below.

    If the answer is not available in the context,
    say that the provided document does not contain the answer.

    Context:
    {context}

    Question:
    {question}
    """

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

    return response.choices[0].message.content


if __name__ == "__main__":

    text = extract_text("data/document.pdf")

    cleaned_text = clean_text(text)

    chunks = create_chunks(cleaned_text)

    embeddings = generate_embeddings(chunks)

    create_collection()

    store_embeddings(chunks, embeddings)

    print("Total chunks:", len(chunks))

    question = "What are the objectives of business communication?"

    results = retrieve_chunks(question)

    print("\nRETRIEVED CHUNKS:")

    for result in results:
        print("\nChunk ID:", result["chunk_id"])
        print("Similarity:", result["score"])
        print("Text:", result["text"][:300])

    answer = generate_answer(question, results)

    print("\nFINAL ANSWER:")
    print(answer)

    client.close()