# prototype/

Streamlit chat application and RAG backend for mī lyte System 1: an evidence-based mindfulness skills recommendation engine.

> [!IMPORTANT]
> This prototype is a work in progress and not intended for distribution. The _i-MBI_ knowledge base is copyright-protected and not publicly available.

| File | Description |
|------|-------------|
| `mi_lyte_sys_1_toy.ipynb` | Backend notebook for RAG initialization: PDF loading, chunking, FAISS vectorization, embedding, and query testing via Ollama. |
| `mi_lyte_sys_1_prototype.py` | Streamlit chat app. Loads the FAISS index, streams LLM responses with `<think>` tag filtering, and renders the mī lyte conversational UI. |
| `demo.py` | Snapshot of the prototype optimized for quick inference demos (reduced `num_predict`, faster streaming). |

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
   ```

3. **Environment variables** — Create a `.env` file in the project root with the path to your knowledge base PDFs:
   ```
   KNOW_DIR=/path/to/your/knowledge/base/pdfs
   ```
   The `.env` file is gitignored and never committed.

4. **FAISS index** — The vector index must be generated before running the Streamlit app. Run the notebook first (see Usage below).

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