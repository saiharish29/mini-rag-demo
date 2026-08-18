"""
Mini RAG Demo — Retrieval-Augmented Generation with a similarity guardrail.

Pipeline: chunk -> embed (BAAI/bge-small-en-v1.5) -> store (ChromaDB)
          -> retrieve (cosine similarity) -> guardrail -> generate (flan-t5-base)
          -> Gradio UI
"""

import os
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# 1. Sample data (swap this out for your own documents)
# ---------------------------------------------------------------------------
SAMPLE_TEXT = """
The Great Barrier Reef is the world's largest coral reef system, located off the coast of Queensland, Australia. It stretches over 2,300 kilometers and is home to thousands of species of marine life, including fish, mollusks, and sharks. The reef is under threat from climate change, which causes coral bleaching when ocean temperatures rise.

Python is a high-level programming language known for its readability and simplicity. It was created by Guido van Rossum and first released in 1991. Python is widely used in web development, data science, artificial intelligence, and automation. Its extensive library ecosystem makes it a popular choice for beginners and professionals alike.

The Renaissance was a period of European cultural, artistic, and intellectual revival that began in Italy in the 14th century. It marked the transition from the medieval to the modern world. Notable figures include Leonardo da Vinci, Michelangelo, and Galileo Galilei. The invention of the printing press by Johannes Gutenberg helped spread Renaissance ideas across Europe.

Photosynthesis is the process by which plants convert sunlight into chemical energy. Chlorophyll in plant cells absorbs light energy, which is used to convert carbon dioxide and water into glucose and oxygen. This process is essential for life on Earth, as it produces the oxygen most organisms need to survive.

The stock market is a collection of exchanges where shares of publicly held companies are bought and sold. Prices are driven by supply and demand, which are influenced by company performance, economic indicators, and investor sentiment. Major stock exchanges include the NYSE and NASDAQ in the United States.
"""

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SIMILARITY_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def chunk_text(text):
    return [p.strip() for p in text.strip().split("\n\n") if p.strip()]

chunks = chunk_text(SAMPLE_TEXT)

# ---------------------------------------------------------------------------
# 3. Embedding model + vector DB
# ---------------------------------------------------------------------------
embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

client = chromadb.PersistentClient(path="./rag_db")
collection = client.get_or_create_collection(name="my_docs")

if collection.count() == 0:
    embeddings = embed_model.encode(chunks, normalize_embeddings=True).tolist()
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
    )

# ---------------------------------------------------------------------------
# 4. Generation model
# ---------------------------------------------------------------------------
t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# ---------------------------------------------------------------------------
# 5. RAG function
# ---------------------------------------------------------------------------
def rag_answer_ui(query, top_k=2, similarity_threshold=SIMILARITY_THRESHOLD):
    if not query.strip():
        return "Please enter a question."

    query_embedding = embed_model.encode(
        QUERY_PREFIX + query, normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=int(top_k),
        include=["documents", "distances"],
    )
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    best_similarity = 1 - distances[0]

    if best_similarity < similarity_threshold:
        return (
            f"**Answer:** I don't have information about that in my documents.\n\n"
            f"*(best match similarity: {best_similarity:.3f}, below threshold {similarity_threshold})*"
        )

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}
Answer:"""

    input_ids = t5_tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output_ids = t5_model.generate(input_ids, max_new_tokens=100)
    answer = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    sources = "\n\n---\n\n".join(retrieved_chunks)
    return f"**Answer:** {answer}\n\n---\n\n**Retrieved sources:**\n\n{sources}"

# ---------------------------------------------------------------------------
# 6. Gradio UI
# ---------------------------------------------------------------------------
demo = gr.Interface(
    fn=rag_answer_ui,
    inputs=[
        gr.Textbox(label="Ask a question", placeholder="e.g. Who invented the printing press?"),
        gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Number of chunks to retrieve (top_k)"),
    ],
    outputs=gr.Markdown(label="Answer"),
    title="Mini RAG Demo",
    description=(
        "Ask a question — answered using BAAI/bge-small-en-v1.5 for retrieval "
        "and flan-t5-small for generation. Answers outside the sample knowledge "
        "base are declined rather than guessed."
    ),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
