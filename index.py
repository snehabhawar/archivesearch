"""
index.py
Two jobs:
  1. Embed transcript chunks and build a FAISS index for semantic search.
  2. Assign controlled-vocabulary topic tags via Llama-8B (Groq) — each clip
     gets tags ONLY from the fixed CONTROLLED_VOCAB list, never free-form.

Why controlled vocab?  Archivists use fixed taxonomies (like LCSH) so metadata
is consistent across decades of holdings.  This mirrors that practice with AI.
"""

import os
import json
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from groq import Groq
except ImportError:
    raise SystemExit("Run: pip install faiss-cpu sentence-transformers groq")

TRANSCRIPT_DIR = "transcripts"
INDEX_FILE = "archive.index"
CHUNKS_FILE = "chunks.json"
TAGGED_FILE = "tagged_clips.json"

EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good for sentence similarity
CHUNK_SIZE = 150                     # words per chunk — balances context and precision
CHUNK_OVERLAP = 30

# ── Controlled vocabulary ───────────────────────────────────────────────────
# Fixed set of topic tags.  The LLM may ONLY choose from these — never invent.
# Modeled after broadcast journalism topic taxonomies.
CONTROLLED_VOCAB = [
    "politics", "economy", "war", "science", "technology",
    "environment", "health", "education", "society", "arts",
    "sports", "international", "government", "civil-rights", "space"
]
# ────────────────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def tag_clip(client: Groq, title: str, year: int, text: str) -> list[str]:
    """Ask Llama-8B to assign topic tags from CONTROLLED_VOCAB only."""
    vocab_str = ", ".join(CONTROLLED_VOCAB)
    prompt = f"""You are an archivist tagging broadcast audio content for a searchable library.

Audio title: {title} ({year})
Transcript excerpt (first 400 words):
{" ".join(text.split()[:400])}

Assign 1–4 topic tags to this audio clip.
You MUST only use tags from this controlled vocabulary list: {vocab_str}
Do NOT invent new tags. Do NOT use tags not in the list.

Respond with ONLY a JSON array of strings, no explanation.
Example: ["politics", "economy"]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,   # low temp = more deterministic tag selection
        max_tokens=60,
    )

    raw = response.choices[0].message.content.strip()
    try:
        tags = json.loads(raw)
        # Enforce constraint: only accept tags from vocab
        tags = [t for t in tags if t in CONTROLLED_VOCAB]
        return tags if tags else ["society"]   # fallback
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse tags for {title}: {raw}")
        return ["society"]


def main():
    with open("transcript_index.json") as f:
        clips = json.load(f)

    # ── Step 1: Chunk all transcripts ───────────────────────────────────────
    print("Chunking transcripts...")
    all_chunks = []
    for clip in clips:
        with open(clip["transcript_txt"]) as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "clip_id": clip["id"],
                "title": clip["title"],
                "year": clip["year"],
                "chunk_index": i,
                "text": chunk,
            })

    with open(CHUNKS_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)
    print(f"  {len(all_chunks)} chunks from {len(clips)} clips")

    # ── Step 2: Embed and index ─────────────────────────────────────────────
    print(f"\nEmbedding with {EMBED_MODEL}...")
    embedder = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # L2-normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    print(f"  FAISS index saved: {INDEX_FILE} ({index.ntotal} vectors, dim={dim})")

    # ── Step 3: Controlled-vocab tagging via Groq ───────────────────────────
    if not GROQ_API_KEY:
        print("\n[SKIP] Set GROQ_API_KEY env var to run tagging step")
    else:
        print("\nTagging clips with controlled vocabulary via Llama-8B (Groq)...")
        client = Groq(api_key=GROQ_API_KEY)
        tagged = []

        for clip in clips:
            with open(clip["transcript_txt"]) as f:
                text = f.read()
            tags = tag_clip(client, clip["title"], clip["year"], text)
            print(f"  {clip['id']} ({clip['year']}): {tags}")
            tagged.append({**clip, "controlled_vocab_tags": tags})

        with open(TAGGED_FILE, "w") as f:
            json.dump(tagged, f, indent=2)
        print(f"\nTagging complete. Results saved to {TAGGED_FILE}")

    print("\nNext: run python search.py")


if __name__ == "__main__":
    main()