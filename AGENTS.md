# AGENTS.md

Purpose: keep future changes to this repo consistent, testable, and easy to extend.

## 1) Repository intent

This repo teaches a **continuous 4-part RAG progression** and a final side-by-side comparison:

1. Tutorial 1: Dense baseline + fixed chunking
2. Tutorial 2: Dense baseline + semantic chunking
3. Tutorial 3: Dense + semantic + reranking
4. Tutorial 4: Hybrid retrieval (dense + BM25)
5. Tutorial 5: Benchmark notebook comparing 1–4 under identical conditions

All tutorials should preserve continuity:
- Same handbook domain and schema
- Same canonical text source (`data/handbook_manual.txt`) as ingestion input
- Same evaluation dimensions
- Same baseline assumptions unless a tutorial explicitly changes one component

## 2) Environment and dependency policy

- Dependency manager: **uv** (required)
- Python: **3.11.13**
- Main runtime configuration: `.env`
- Primary commands:
  - `uv sync`
  - `uv lock`
  - `uv run jupyter lab`

Do not introduce alternate package managers in docs unless explicitly requested.

## 3) Source layout (single source of truth)

- `src/rag_tutorials/`
  - `data_generation.py`: synthetic corpus/query generation
  - `io_utils.py`: jsonl load/save helpers
  - `schema.py`: dataclasses for document/chunk/query/result
  - `chunking.py`: fixed and semantic chunkers
  - `embeddings.py`: OpenAI embedding + cosine similarity helper
  - `vector_store.py`: Chroma index build/query
  - `retrieval.py`: BM25 + reciprocal rank fusion
  - `reranking.py`: cross-encoder reranker
  - `qa.py`: context-aware answer generation
  - `evaluation.py`: recall/mrr/latency/groundedness summary
  - `pipeline.py`: shared pipeline constructors and helpers
- `tutorials/`: notebooks (01–05)
- `scripts/`: utility scripts (`generate_data.py`, `smoke_imports.py`)
- `data/`: canonical handbook text + generated jsonl datasets
- `artifacts/`: persisted local artifacts (e.g., Chroma)

Rule: avoid duplicating logic in notebooks when it belongs in `src/rag_tutorials`.

## 4) Notebook contract

Each tutorial notebook should:
1. Load env and validate required keys early
2. Use shared modules from `rag_tutorials`
3. Include at least one novice-focused embedding/retrieval trace
4. Include an inline diagram (Mermaid or equivalent)
5. Include a learning checkpoint that states:
  - what is working,
  - what is not working,
  - why the next tutorial is needed
6. Show measurable outputs (table or metrics dict)

Tutorial-specific change isolation:
- T1 changes: none (baseline)
- T2 changes: chunking only
- T3 changes: adds reranking stage
- T4 changes: adds hybrid retrieval stage
- T5 changes: no new method; comparison only

Tutorial transition narrative contract:
- T1 -> T2: move because baseline chunk boundaries lose policy context.
- T2 -> T3: move because ranking quality is still inconsistent.
- T3 -> T4: move because dense retrieval may miss exact lexical signals.
- T4 -> T5: move because measured tradeoffs must guide final architecture choice.

## 5) Evaluation contract

Core metrics used across variants:
- `recall_at_k`
- `mrr`
- `groundedness`
- `latency_ms`

When adding a variant, ensure:
- Same query slice is used for comparison
- Same `top_k` unless intentionally studied
- Any metric definition changes are reflected in all tutorials and README

## 6) Safe-change checklist (for future contributors/agents)

Before coding:
- Confirm whether change is tutorial-local or shared-module level
- Prefer updating shared module, then notebook usage

During coding:
- Keep changes scoped to requested behavior
- Do not silently alter dataset schema without migration notes
- Maintain deterministic defaults where possible (seeded generation)

After coding:
- Run `uv sync` if dependencies changed
- Run `uv lock` if dependency graph changed
- Run `uv run python scripts/smoke_imports.py`
- If notebook behavior changed, re-run affected notebook cells end-to-end
- Update `README.md` and this file if workflow or architecture changed

## 7) Adding a new RAG variant (playbook)

1. Implement variant logic in `src/rag_tutorials` (not only notebook code).
2. Add/extend retrieval function with same return shape (`RetrievalResult`).
3. Reuse current evaluation harness (`evaluate_single`, `summarize`).
4. Add a notebook (or section) with:
   - what changed vs prior tutorial
   - one novice retrieval trace
   - one quantitative comparison table
5. Include variant in `tutorials/05_rag_comparison.ipynb` benchmark table.
6. Update README run order and progression diagram if needed.

## 8) Known assumptions and constraints

- OpenAI API is used for embeddings and generation.
- Chroma is local/persistent vector store.
- Reranking uses local cross-encoder model from sentence-transformers.
- Synthetic HR/international work policy data is used for continuity and reproducibility.

## 9) Non-goals unless explicitly requested

- Replacing stack with unrelated frameworks
- Introducing extra tutorials/pages/features not tied to the progression
- Changing tutorial order or metric semantics without documented rationale

## 10) Quick recovery commands

- Recreate dataset:
  - `uv run python scripts/generate_data.py`
- Smoke-check imports and basic pipeline wiring:
  - `uv run python scripts/smoke_imports.py`
- Open notebooks:
  - `uv run jupyter lab`
