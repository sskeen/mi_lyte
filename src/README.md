# src/

Source code for mī lyte System 1.

| File | Description |
|------|-------------|
| `config.py` | Shared configuration: system prompt, LLM parameters, retriever settings, and prompt template. Single source of truth imported by both the notebook and the prototype. |
| `dialogue_stream.py` | Standalone `query_and_stream()` function that replicates LangChain's `RetrievalQA` (stuff chain) with token-by-token streaming and optional source metadata display. Used by `mi_lyte_sys_1_toy.ipynb`. |

