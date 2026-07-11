# prototype/

Streamlit chat application and RAG backend for mī lyte System 1: an evidence-based mindfulness skills recommendation engine.

> [!IMPORTANT]
> This prototype is a work in progress and not intended for distribution. The _i-MBI_ knowledge base is copyright-protected and not publicly available.

## Files

| File | Description |
|------|-------------|
| `demo.py` | Streamlit chat app for demos. Loads FAISS index, streams LLM responses with reasoning masked, renders the mī lyte UI. |
| `diagnostic.ipynb` | Backend notebook for RAG diagnostics: PDF loading, chunking, FAISS vectorization, and query testing with exposed reasoning traces and source citations. |

Both files import configuration and streaming logic from `src/`:
- `src/config.py` — LLM parameters, retriever settings, prompt template
- `src/dialogue_stream.py` — Unified streaming function with `mode` parameter

---

## Architecture

The `query_and_stream()` function in `src/dialogue_stream.py` supports two modes:

| Mode | Output | Use Case |
|------|--------|----------|
| `mode='diagnostic'` | Prints reasoning traces (`<think>...</think>`) and source excerpts to stdout | Jupyter notebook inspection |
| `mode='ui'` | Returns generator yielding masked tokens with sentinel events | Streamlit UI streaming |

This unified design keeps retrieval and prompt logic in one place while supporting both diagnostic inspection and user-facing output.

---

## Prerequisites

1. **Ollama** running locally with required models:
   ```bash
   ollama pull deepseek-r1:14b
   ollama pull nomic-embed-text
   ollama serve
   ```

2. **Python dependencies** — Install from project root:
   ```bash
   pip install -r requirements.txt
   pip install langchain-ollama  # Required for non-deprecated Ollama integration
   ```

3. **Environment variables** — Create a `.env` file in the project root with the path to your knowledge base PDFs:
   ```
   KNOW_DIR=/path/to/your/knowledge/base/pdfs
   ```
   The `.env` file is gitignored and never committed.

4. **FAISS index** — The vector index must be generated before running the Streamlit app. Run the notebook first (see Usage below).

---

## Usage

### Build FAISS Index (first time or after knowledge base changes)

1. Open `diagnostic.ipynb` in Jupyter
2. Run Section 1 (Setup) and Section 2 (Build FAISS Index)
3. Index saves to `src/faiss_index/`

### Run Demo

```bash
cd prototype
streamlit run demo.py
```

### Diagnostic Testing

1. Open `diagnostic.ipynb` in Jupyter
2. Run Section 1 (Setup) and Section 3 (Query and Inspect)
3. Observe reasoning traces and source citations in output

---

## Configuration

LLM and retriever settings are centralized in `src/config.py`. Edit to adjust:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `model` | `LLM_PARAMS` | Ollama model tag (default: `deepseek-r1:14b`) |
| `temperature` | `LLM_PARAMS` | Sampling temperature (default: 0.6) |
| `num_predict` | `LLM_PARAMS` | Max tokens to generate (default: 768) |
| `num_ctx` | `LLM_PARAMS` | Context window size (default: 2048) |
| `EMBEDDING_MODEL` | `config.py` | Embedding model for vectorization (default: `nomic-embed-text`) |
| `k` | `RETRIEVER_PARAMS` | Number of chunks to retrieve per query (default: 4) |

The system prompt is defined in `src/system_prompt.py`.

---

_Last updated: 07-11-2026_
