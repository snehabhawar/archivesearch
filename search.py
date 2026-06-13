"""
search.py
Semantic search over the FAISS-indexed transcript library.
Embeds the query and retrieves the top-k most similar transcript chunks,
with the source clip title, year, and timestamp context.

Usage:
    python search.py "moon landing space exploration"
    python search.py "economic recovery banking" --top_k 5
    python search.py "military conflict declaration" --filter_tag war
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise SystemExit("Run: pip install faiss-cpu sentence-transformers")

INDEX_FILE = "archive.index"
CHUNKS_FILE = "chunks.json"
TAGGED_FILE = "tagged_clips.json"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_tag_map() -> dict:
    """Returns {clip_id: [tags]} if tagged_clips.json exists, else empty."""
    if not os.path.exists(TAGGED_FILE):
        return {}
    with open(TAGGED_FILE) as f:
        tagged = json.load(f)
    return {c["id"]: c.get("controlled_vocab_tags", []) for c in tagged}


def search(query: str, top_k: int = 5, filter_tag: str = None) -> list[dict]:
    # Load index and chunks
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    tag_map = load_tag_map()

    # Embed query
    embedder = SentenceTransformer(EMBED_MODEL)
    q_vec = embedder.encode([query], normalize_embeddings=True)
    q_vec = np.array(q_vec, dtype="float32")

    # Retrieve more than top_k so we can filter by tag if needed
    k = min(top_k * 4, index.ntotal)
    scores, indices = index.search(q_vec, k)

    results = []
    seen_clips = set()   # deduplicate: one result per clip

    for score, idx in zip(scores[0], indices[0]):
        chunk = chunks[idx]
        clip_id = chunk["clip_id"]

        # Optional tag filter
        if filter_tag:
            tags = tag_map.get(clip_id, [])
            if filter_tag not in tags:
                continue

        # Deduplicate clips (show best-scoring chunk per clip)
        if clip_id in seen_clips:
            continue
        seen_clips.add(clip_id)

        results.append({
            "clip_id": clip_id,
            "title": chunk["title"],
            "year": chunk["year"],
            "score": round(float(score), 4),
            "tags": tag_map.get(clip_id, []),
            "excerpt": chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""),
        })

        if len(results) >= top_k:
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Semantic search over archive transcripts")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--filter_tag", type=str, help="Restrict to clips with this tag")
    args = parser.parse_args()

    if not args.query:
        # Interactive mode
        print("ArchiveSearch — semantic search over historical broadcast audio")
        print("Type a query and press Enter. Ctrl+C to exit.\n")
        while True:
            try:
                query = input("Query: ").strip()
                if not query:
                    continue
                tag_filter = input("Filter by tag (leave blank for none): ").strip() or None
                results = search(query, top_k=5, filter_tag=tag_filter)
                _print_results(query, results)
            except KeyboardInterrupt:
                print("\nBye.")
                break
    else:
        results = search(args.query, top_k=args.top_k, filter_tag=args.filter_tag)
        _print_results(args.query, results)


def _print_results(query: str, results: list[dict]):
    print(f"\nQuery: \"{query}\"")
    print(f"Found {len(results)} result(s)\n" + "─" * 60)
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        tags_str = ", ".join(r["tags"]) if r["tags"] else "untagged"
        print(f"{i}. [{r['score']:.3f}] {r['title']} ({r['year']})")
        print(f"   Tags: {tags_str}")
        print(f"   {r['excerpt']}\n")


if __name__ == "__main__":
    main()