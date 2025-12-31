## Changelog

### 2025-10-10

- Add CSV import for RAGAS-style queries via `trcli questions load_csv`.
  - CSVs can be placed anywhere under `data/ragas_queries/`. The loader searches by filename within this folder, so you can pass just the filename on the CLI.
  - New examples and instructions added to `README.md` under "Import RAGAS CSV queries".
- Add prompt breakdown to `trcli db stats`

#### Developer actions / configuration changes

- Ensure `data/ragas_queries/` exists and contains your CSV files. Any substructure is fine.
- If you rely on filename lookup, keep unique filenames across subfolders to avoid ambiguity. If duplicates exist, pass a full path under `data/ragas_queries/` instead.
- Use a dedicated prompt tag to isolate imported queries, e.g. `--prompt_name=ragas_squad`. This helps filtering during embedding and evaluation. I suggest to set prompt-name to the name of file we are using.
- After loading, embed imported queries filtered by the same LLM tag used on load, e.g. `trcli embed queries --dataset=squad --llm=ragas`.
- If you changed dataset names, ensure your DB already contains docs/chunks for `--dataset` you pass to `load_csv` so mapping by `original_row_idx`/`source_chunk_idx` can succeed.
