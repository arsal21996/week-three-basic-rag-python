import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
EMBEDDINGS_FILE = ROOT / "embeddings.json"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6-luna"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.35
CHUNK_WORDS = 80


SYSTEM_RULE = """You are a grounded RAG assistant.
Use ONLY the supplied context to answer the question.
If the context does not contain enough information, say:
I don't know based on the provided context.
Do not add outside facts.
"""


def api_request(endpoint, payload):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run: set OPENAI_API_KEY=your-key"
        )

    request = urllib.request.Request(
        f"https://api.openai.com/v1/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def embed_text(text):
    result = api_request(
        "embeddings",
        {"model": EMBEDDING_MODEL, "input": text},
    )
    return result["data"][0]["embedding"]


def generate_text(prompt):
    result = api_request(
        "responses",
        {"model": CHAT_MODEL, "input": prompt},
    )
    if result.get("output_text"):
        return result["output_text"]

    # Defensive fallback for response payloads where output_text is absent.
    pieces = []
    for item in result.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                pieces.append(content.get("text", ""))
    return "\n".join(pieces).strip()


def chunk_text(text, words_per_chunk=CHUNK_WORDS):
    """Simple manual word-count chunking."""
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def load_documents():
    documents = []
    for path in sorted(DOCS_DIR.glob("*.txt")):
        documents.append((path.name, path.read_text(encoding="utf-8")))
    if not documents:
        raise RuntimeError(f"No .txt documents found in {DOCS_DIR}")
    return documents


def build_chunks():
    chunks = []
    for filename, text in load_documents():
        for chunk_id, chunk in enumerate(chunk_text(text)):
            chunks.append(
                {
                    "source": filename,
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "embedding": embed_text(chunk),
                }
            )
    return chunks


def save_embeddings(chunks):
    EMBEDDINGS_FILE.write_text(
        json.dumps(chunks, indent=2),
        encoding="utf-8",
    )


def load_embeddings():
    return json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))


def cosine_similarity(a, b):
    """Manual cosine similarity using NumPy, as used in the exercise."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def retrieve(query, chunks, top_k=TOP_K):
    query_embedding = embed_text(query)
    scored = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append(
            {
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def print_results(results):
    print("\nTop matches:")
    print("-" * 72)
    for index, result in enumerate(results, start=1):
        print(
            f"{index}. score={result['score']:.4f} "
            f"source={result['source']} chunk={result['chunk_id']}"
        )
        print(f"   {result['text']}")


def build_context(results):
    return "\n\n".join(
        f"[Source: {item['source']}, chunk {item['chunk_id']}]\n{item['text']}"
        for item in results
    )


def answer_with_context(query, results):
    context = build_context(results)
    prompt = f"""{SYSTEM_RULE}

CONTEXT:
{context}

QUESTION:
{query}
"""
    return generate_text(prompt)


def answer_without_context(query):
    prompt = f"Answer this question normally:\n\n{query}"
    return generate_text(prompt)


def answer_with_threshold(query, chunks):
    results = retrieve(query, chunks)
    print_results(results)

    best_score = results[0]["score"] if results else 0.0
    print(f"\nBest similarity: {best_score:.4f}")
    print(f"Threshold: {SIMILARITY_THRESHOLD:.4f}")

    if best_score < SIMILARITY_THRESHOLD:
        return "I don't know based on the provided documents."

    return answer_with_context(query, results)


def ensure_embeddings(force=False):
    if EMBEDDINGS_FILE.exists() and not force:
        print(f"Loading cached embeddings from {EMBEDDINGS_FILE.name}")
        return load_embeddings()

    print("Generating embeddings for document chunks...")
    chunks = build_chunks()
    save_embeddings(chunks)
    print(f"Saved {len(chunks)} chunks to {EMBEDDINGS_FILE.name}")
    return chunks


def demo(chunks):
    query = "What is retrieval augmented generation?"

    print("\n" + "=" * 72)
    print("1) RETRIEVAL: TOP 3 MATCHES")
    print("=" * 72)
    results = retrieve(query, chunks)
    print_results(results)

    print("\n" + "=" * 72)
    print("2) ANSWER WITH RETRIEVED CONTEXT")
    print("=" * 72)
    print(answer_with_context(query, results))

    print("\n" + "=" * 72)
    print("3) ANSWER WITHOUT RETRIEVED CONTEXT")
    print("=" * 72)
    print(answer_without_context(query))

    unrelated = "Who won the 2022 FIFA World Cup?"
    print("\n" + "=" * 72)
    print("4) UNRELATED QUESTION + SIMILARITY THRESHOLD")
    print("=" * 72)
    print(f"Question: {unrelated}")
    print(answer_with_threshold(unrelated, chunks))


def chatbot(chunks):
    """Conversation history persists, but retrieved context is fresh every turn."""
    history = []

    print("\nRAG chatbot. Type 'quit' to exit.")
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"quit", "exit"}:
            print("Goodbye!")
            return
        if not query:
            continue

        results = retrieve(query, chunks)
        print_results(results)

        best_score = results[0]["score"] if results else 0.0
        if best_score < SIMILARITY_THRESHOLD:
            answer = "I don't know based on the provided documents."
        else:
            recent_history = "\n".join(
                f"{message['role'].upper()}: {message['content']}"
                for message in history[-6:]
            )
            prompt = f"""{SYSTEM_RULE}

PREVIOUS CONVERSATION:
{recent_history or '(none)'}

FRESH RETRIEVED CONTEXT FOR THIS TURN:
{build_context(results)}

LATEST QUESTION:
{query}
"""
            answer = generate_text(prompt)

        print(f"\nAssistant: {answer}")

        # Store only conversation history. Do not permanently append retrieved chunks.
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})


def main():
    chunks = ensure_embeddings()
    demo(chunks)
    chatbot(chunks)


if __name__ == "__main__":
    main()
