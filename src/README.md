# src/

Source code for mī lyte System 1.

| File | Description |
|------|-------------|
| `config.py` | Shared configuration: system prompt, LLM parameters, retriever settings, and prompt template. Single source of truth imported by both the notebook and the prototype. |
| `dialogue_stream.py` | Standalone `query_and_stream()` function that replicates LangChain's `RetrievalQA` (stuff chain) with token-by-token streaming and optional source metadata display. Used by the notebook. |
| `mi_lyte_system01_config.ipynb` | Backend notebook for RAG initialization: PDF loading, chunking, FAISS vectorization, embedding, and query testing via Ollama. |
| `mi_lyte_system01_prototype.py` | Streamlit chat app. Loads the FAISS index, streams LLM responses with `<think>` tag filtering, and renders the mī lyte conversational UI. Run via `streamlit run mi_lyte_system01_prototype.py`. |
| `demo.py` | Snapshot of the prototype optimized for quick inference demos (reduced `num_predict`, faster streaming). |

