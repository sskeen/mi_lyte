# data/

Input files for crisis query simulation.

| File | Description |
|------|-------------|
| `personas.tsv` | Persona definitions with columns: `persona_id`, `persona_name`, `age`, `gender`, `current_suicide_risk_level`. |
| `seed_phrases.tsv` | Seed phrases for query grounding with columns: `seed_id`, `persona_id`, `seed_phrase`. Each persona has 5 associated seeds. |
| `persona_context.txt` | Per-persona context for grounding. Sections delimited by `{persona_id = PXXX}` headers. |
