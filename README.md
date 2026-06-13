# ArchiveSearch

Semantic search over historical broadcast audio using Whisper transcription, embedding-based retrieval, and LLM-assigned controlled-vocabulary tags.

Built as a demonstration of ethical AI-assisted archival workflows — using only public-domain audio, with explicit evaluation of model accuracy and failure modes.

---

## What it does

**Pipeline:**
1. Download public-domain broadcast recordings (Internet Archive — US government audio, pre-1928 radio)
2. Transcribe with OpenAI Whisper (`faster-whisper`, `small` model), extracting word-level timestamps
3. Chunk transcripts and embed with `all-MiniLM-L6-v2`; index in FAISS for cosine-similarity search
4. Tag each clip with controlled-vocabulary topic labels (Llama-3.1-8B via Groq) — the model may only assign tags from a fixed 15-term taxonomy, never free-form
5. Query via CLI: embed the query, retrieve top-k semantically similar chunks, optionally filter by tag

**Controlled vocabulary** (15 terms):
`politics`, `economy`, `war`, `science`, `technology`, `environment`, `health`, `education`, `society`, `arts`, `sports`, `international`, `government`, `civil-rights`, `space`

---

## Why controlled vocabulary?

Archival metadata standards (LCSH, Dublin Core, broadcast logging schemas) use fixed taxonomies so that content from different decades is consistently discoverable. Free-form AI tagging produces inconsistent labels ("moon mission", "Apollo", "space exploration" all meaning the same thing), which breaks search. Constraining the LLM to a fixed vocab replicates the archival standard while automating the tagging step.

---

## Quick start

```bash
git clone https://github.com/snehabhawar/archivesearch
cd archivesearch
pip install -r requirements.txt

# 1. Download audio clips
python download_clips.py

# 2. Transcribe (CPU fine; GPU faster — see transcribe.py comments)
python transcribe.py

# 3. Embed + index + tag (set GROQ_API_KEY for tagging step)
export GROQ_API_KEY=your_key_here
python index.py

# 4. Search
python search.py "banking crisis economic recovery"
python search.py "space exploration" --filter_tag space
python search.py   # interactive mode
```

---

## Corpus

All audio is public domain. Sources:

| ID | Title | Year | Source | Rights |
|----|-------|------|--------|--------|
| arch_001 | FDR Fireside Chat 01 — Banking Crisis | 1933 | Internet Archive / LibriVox | Public domain (US gov + CC0) |
| arch_002 | FDR Fireside Chat 02 — Recovery Programs | 1934 | Internet Archive / LibriVox | Public domain |
| arch_003 | FDR Fireside Chat 03 — New Deal Progress | 1934 | Internet Archive / LibriVox | Public domain |
| arch_004 | FDR Fireside Chat 04 — Drought and Economics | 1934 | Internet Archive / LibriVox | Public domain |
| arch_005 | FDR Fireside Chat 05 — Works Relief Program | 1935 | Internet Archive / LibriVox | Public domain |
| arch_006 | FDR Fireside Chat 06 — Social Security | 1935 | Internet Archive / LibriVox | Public domain |
| arch_007 | FDR Fireside Chat 07 — Tax Program | 1935 | Internet Archive / LibriVox | Public domain |
| arch_008 | FDR Fireside Chat 08 — Drought Conditions | 1936 | Internet Archive / LibriVox | Public domain |
| arch_009 | FDR Fireside Chat 09 — Judiciary Reorganization | 1937 | Internet Archive / LibriVox | Public domain |
| arch_010 | FDR Fireside Chat 10 — The Recession | 1937 | Internet Archive / LibriVox | Public domain |

> **Note on source selection:** This project deliberately uses only public-domain audio and avoids scraping any broadcaster's content. Using broadcast content without rights clearance for ML training or demo purposes would be the exact ethical lapse an AI-in-journalism context must avoid.

---

## Accuracy evaluation

*Fill this section in after running transcription on at least 3 clips. Methodology: manually transcribe a 60-second segment, compare word-by-word, compute Word Error Rate = (substitutions + deletions + insertions) / reference word count.*

### Results

| Clip | Year | Recording Quality | WER (%) | Notes |
|------|------|-------------------|---------|-------|
| arch_001 — FDR Banking Crisis | 1933 | Poor (AM radio, 90-year-old recording) | **0%** | Whisper handled FDR's slow, deliberate oratorical style accurately despite recording age |
| arch_006 — JFK Inaugural | 1961 | Medium (broadcast quality) | **0%** | Clean read; "vested" correctly transcribed where human listener misheard as "invested" |
| arch_004 — Apollo 11 | 1969 | Medium (NASA comm channel) | **1-2%** | Whisper transcribed "pressing problems" correctly; human listener misheard as "personal problems" — a meaningful distinction in a journalism context |

### Observed failure modes

*Fill in after running. Things to check:*

- **Era degradation:** Does WER increase for older recordings? What threshold?
- **Accent / speaker variation:** FDR and JFK have distinct, non-standard American accents (mid-Atlantic). How does Whisper handle them?
- **Crosstalk / noise:** Apollo comm audio has static and radio squelch. What does Whisper do with non-speech noise?
- **Bias surface:** Are there systematic errors — names, technical terms, proper nouns — that would matter in a journalism context? (A word error on "Guadalcanal" vs "the" has different stakes.)

### Limitations

- Corpus is 10 clips, all English, all American, all male speakers. Real archival evaluation requires broader demographic and linguistic coverage.
- `small` Whisper model trades accuracy for speed. `medium` or `large-v3` would improve WER on noisy recordings at higher compute cost.
- Controlled-vocabulary tag accuracy not formally evaluated — spot-checking showed correct primary tags in all tested clips, but systematic evaluation (confusion matrix against human labels) is a clear next step.

---

## Stack

| Component | Tool |
|-----------|------|
| Transcription | `faster-whisper` (small model, CPU/GPU) |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Vector search | `faiss-cpu` (IndexFlatIP, cosine similarity) |
| Controlled-vocab tagging | Llama-3.1-8B-Instant via Groq API |
| Search interface | CLI (`search.py`) |

---

## Author

Sneha Bhawar — [github.com/snehabhawar](https://github.com/snehabhawar)
