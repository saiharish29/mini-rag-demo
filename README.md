# Mini RAG — Retrieval-Augmented Generation Demo

A small, fully working RAG (Retrieval-Augmented Generation) pipeline built from
scratch to understand every moving part: chunking, embeddings, vector storage,
semantic search, and grounded answer generation — wrapped in a Gradio UI.

> New to RAG? Open **`rag_explainer.html`** in any browser first. It explains
> everything here in plain language with interactive visuals — no code
> reading required.

---

## What this project does

You ask a question in plain English. The system:

1. Converts your question into a vector (a list of 384 numbers that capture its meaning)
2. Compares that vector against a database of pre-embedded text chunks
3. Pulls back the chunks that are semantically closest to your question
4. Feeds those chunks + your question to a language model
5. Returns a grounded answer — or honestly says "I don't know" if nothing in the database is relevant

This is the same architecture used in production RAG systems (customer support
bots, internal document search, "chat with your PDF" tools) — just with small,
free, local models instead of paid APIs.

---

## Architecture

```
                    ┌─────────────────┐
   Documents  ───▶  │  Chunking        │
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │  Embedding       │  BAAI/bge-small-en-v1.5
                    │  (text → vector) │  (384 dimensions)
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │  ChromaDB        │  Vector database
                    │  (vector store)  │  (persisted to disk)
                    └────────┬─────────┘
                             │
   User query  ───▶  embed query  ───▶  cosine similarity search
                             │
                             ▼
                    ┌─────────────────┐
                    │  Top-K chunks    │  + similarity threshold guardrail
                    └────────┬─────────┘
                             ▼
                    ┌─────────────────┐
                    │  flan-t5-base    │  Generates grounded answer
                    │  (generation)    │  from retrieved context
                    └────────┬─────────┘
                             ▼
                       Final answer
                    (via Gradio UI)
```

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Embedding model | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | Small (~130MB), fast, strong retrieval quality for its size |
| Vector database | [ChromaDB](https://www.trychroma.com/) | Lightweight, no server setup, persists to local disk |
| Generation model | [`google/flan-t5-base`](https://huggingface.co/google/flan-t5-base) | Free, local, instruction-tuned, no API key required |
| UI | [Gradio](https://www.gradio.app/) | Fastest way to wrap a Python function in a shareable web UI |
| Deployment | [Render](https://render.com) | Free tier, plain CPU hosting, no GPU-only restrictions |

---

## Project structure

```
.
├── app.py              # Full pipeline + Gradio interface (deploy this to Spaces)
├── requirements.txt     # Python dependencies
├── README.md             # This file
└── rag_explainer.html   # Standalone interactive explainer (open in any browser)
```

---

## Code walkthrough

### 1. Chunking
Source text is split into paragraphs (one topic per chunk in this demo).
For real documents, sentence-aware chunking with overlap is more robust —
fixed-size character chunking can cut a sentence in half and hurt retrieval.

### 2. Embedding
```python
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
embeddings = model.encode(chunks, normalize_embeddings=True)
```
Each chunk becomes a 384-number vector. `normalize_embeddings=True` scales
every vector to unit length, which is required for cosine similarity to work
correctly.

**Important BGE-specific detail:** queries get a special instruction prefix,
documents do not:
```python
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```
This is how the model was trained, and skipping it measurably hurts retrieval
accuracy.

### 3. Storage
```python
collection.add(ids=..., embeddings=..., documents=chunks)
```
ChromaDB stores the vector *and* the original text together, so a similarity
search on vectors can return readable text.

### 4. Retrieval
```python
results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
```
Chroma returns *distance* (lower = more similar). We convert with
`similarity = 1 - distance` for a more intuitive "higher = better" scale.

### 5. Guardrail
```python
if best_similarity < similarity_threshold:
    return "I don't have information about that in my documents."
```
Retrieval always returns its "closest" matches — even when none of them are
actually relevant. Without this check, the model will confidently answer
questions using totally unrelated context (we saw this happen live: asking
"who invented the car?" pulled back a Renaissance paragraph and the model
guessed "Johannes Gutenberg"). The threshold value (0.3 here) was chosen by
eyeballing scores on a few in-domain vs. out-of-domain test queries — in a
production system you'd tune this more rigorously against a labeled test set.

### 6. Generation
```python
input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
output_ids = t5_model.generate(input_ids, max_new_tokens=100)
```
The retrieved chunks are stitched into a prompt and passed to flan-t5-base,
which generates a natural-language answer grounded in that context — instead
of just handing back raw retrieved text.

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Gradio will print a local URL to open in your browser.

## Deploying permanently (Render)

1. Push `app.py` and `requirements.txt` to a public GitHub repo
2. On [render.com](https://render.com), create a new **Web Service** and connect that repo
3. Configure:
   - **Runtime**: Python 3
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `python app.py`
   - **Instance type**: Free
4. Deploy — Render gives you a permanent public URL

No GPU required — both models are small enough to run on Render's free CPU tier.
The free tier sleeps after ~15 minutes of inactivity and takes ~30–50 seconds to
wake back up on the next visit — expected behavior on the free plan, not a bug.

> **Note:** We originally attempted Hugging Face Spaces, but new free accounts
> are currently defaulted to ZeroGPU hardware (GPU-only), which had an
> allocation issue at the time ("No CUDA GPUs are available") and couldn't be
> downgraded to CPU-only hardware without a PRO subscription. Render was used
> instead since this app doesn't need a GPU at all.

---

## Known limitations (by design, for learning purposes)

- **Tiny knowledge base** — only 5 sample paragraphs. Swap in real documents
  to make this useful for an actual use case.
- **flan-t5-base gives terse answers** — it's a small model. A larger model
  (flan-t5-large/xl) or a hosted API (Claude, GPT) would produce more natural,
  detailed answers using the exact same retrieval pipeline.
- **Fixed similarity threshold** — not tuned on a real evaluation set.
- **No conversation memory** — each question is answered independently.

---

## Credits

Built step by step as a learning project: chunking → embedding → vector
storage → retrieval → generation → guardrails → UI → deployment.
