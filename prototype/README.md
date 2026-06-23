# src/

Source code for mī lyte System 1.

| File | Description |
|------|-------------|
| `mi_lyte_sys_1_toy.ipynb` | Backend notebook for RAG initialization: PDF loading, chunking, FAISS vectorization, embedding, and query testing via Ollama. |
| `mi_lyte_sys_1_prototype.py` | Streamlit chat app. Loads the FAISS index, streams LLM responses with `<think>` tag filtering, and renders the mī lyte conversational UI. Run via `streamlit run mi_lyte_system01_prototype.py`. |
| `demo.py` | Snapshot of the prototype optimized for quick inference demos (reduced `num_predict`, faster streaming). |

