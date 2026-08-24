# Week Three — Basic RAG in Python

A deliberately small Retrieval-Augmented Generation (RAG) pipeline built from scratch.

The goal is to see the mechanics of RAG instead of hiding them behind LangChain, FAISS, or a vector database.

## What this project does

1. Loads five short `.txt` documents.
2. Chunks each document with a simple word-count function.
3. Generates Gemini embeddings for each chunk.
4. Stores `{source, chunk_id, text, embedding}` as a plain Python list and caches it in `embeddings.json`.
5. Embeds a user query with Gemini.
6. Computes cosine similarity against every stored chunk with NumPy.
7. Prints the top 3 matches and their similarity scores.
8. Sends those top chunks to Gemini as context.
9. Compares an answer with retrieved context against an answer without it.
10. Tests an unrelated question.
11. Applies a similarity threshold so irrelevant questions return `I don't know` instead of forcing an answer.
12. Runs an interactive chatbot where retrieval is performed fresh on every turn.

## Stack

- Python 3.10+
- NumPy
- Gemini API via Python's standard `urllib`
- No LangChain
- No FAISS
- No vector database
- No Gemini Python SDK

The only third-party Python dependency is NumPy. Gemini calls are raw HTTP requests so the retrieval mechanics remain visible.

## Project structure

```text
week-three-basic-rag-python/
├── docs/
│   ├── chatbot.txt
│   ├── embeddings.txt
│   ├── python.txt
│   ├── rag.txt
│   └── vector_search.txt
├── rag.py
├── requirements.txt
├── setup.bat
├── run.bat
├── rebuild_embeddings.bat
├── .env.example
└── README.md
```

`embeddings.json` and `.env` are local files and are ignored by Git. `embeddings.json` is generated from the documents; `.env` contains your secret API key.

## Windows quick start

### 1. Get a Gemini API key

Create a Gemini API key in Google AI Studio:

https://aistudio.google.com/app/apikey

### 2. Create your local `.env`

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Then open `.env` and replace the placeholder:

```text
GEMINI_API_KEY=your-real-gemini-api-key
```

You only need to do this **once**. The Python program loads `.env` automatically every time it starts. No environment variable command is required for normal use.

**Never commit `.env` to GitHub. It is already listed in `.gitignore`.**

### 3. Run setup

Double-click:

```text
setup.bat
```

This creates `.venv` and installs NumPy.

### 4. Run the project

Double-click:

```text
run.bat
```

The first run generates document embeddings and saves them to `embeddings.json`. Later runs load that file instead of regenerating document embeddings.

### 5. Rebuild embeddings when documents change

After editing or adding files in `docs/`, run:

```text
rebuild_embeddings.bat
```

Then run `run.bat` normally.

## What to expect

The program first demonstrates retrieval:

```text
Top matches:
------------------------------------------------------------------------
1. score=0.xxxx source=rag.txt chunk=0
   Retrieval-Augmented Generation...

2. score=0.xxxx source=embeddings.txt chunk=0
   Vector embeddings...

3. score=0.xxxx source=vector_search.txt chunk=0
   Vector search...
```

The exact scores depend on the embedding model and should not be treated as universal constants.

It then shows:

- an answer using retrieved context;
- an answer without retrieved context;
- an unrelated question with the threshold applied;
- an interactive RAG chatbot.

## Grounding

Grounding means giving the language model specific external information that it should use as the basis for its answer. In this project, the embedding model does not answer the question. It converts each document chunk and the user's query into vectors so we can find chunks that are semantically close to the query. Those retrieved chunks are then inserted into the Gemini prompt. The model is instructed to answer only from that supplied context. This makes the answer grounded in the selected documents instead of asking the model to rely only on its general learned knowledge. The similarity threshold adds another guardrail: if the best matching chunk is not similar enough, the application refuses to answer from irrelevant context.

This simple implementation does not scale because every query loops over every stored embedding and calculates cosine similarity. That is fine for a handful of chunks, but a collection containing millions of vectors would require millions of comparisons for each query. The JSON file is also only a basic persistence mechanism; it has no efficient vector index, metadata filtering, concurrency model, or production database features. Systems such as FAISS and vector databases use specialized nearest-neighbor indexes and storage systems to make vector retrieval much faster and more manageable at large scale. They can avoid comparing a query against every vector and can provide features for updating, filtering, persistence, and distributed workloads. The important lesson here is that RAG itself is not mysterious: chunk the data, embed it, retrieve the nearest chunks, put them in the prompt, and generate an answer. Production systems mainly improve the storage and retrieval part of that pipeline.

## Fresh retrieval vs. chat history

The chatbot keeps a small amount of conversation history, but it does **not** permanently append retrieved chunks to that history.

Each turn follows this pattern:

```text
new question
    ↓
query embedding
    ↓
cosine similarity against all chunks
    ↓
top 3 fresh chunks
    ↓
Gemini prompt
    ↓
answer
```

This matters because old retrieved context can become irrelevant to a later question. Conversation history and retrieval context are different kinds of state.

## Changing the threshold

The default threshold is:

```python
SIMILARITY_THRESHOLD = 0.35
```

There is no universal correct threshold. Run the demo, inspect the printed scores, and adjust it for the document collection and embedding model you are using.

## Adding your own documents

Put `.txt` files into `docs/`. The program automatically loads them. If you change the documents, rebuild the cached embeddings with `rebuild_embeddings.bat`.

## Notes

This is intentionally educational code. It uses a brute-force in-memory search so the core RAG steps remain visible. It is not intended to replace a production retrieval system.
