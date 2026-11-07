*This project has been created as part of the 42 curriculum by adraji.*

# RAG Against the Machine

## Description

Standalone Large Language Models (LLMs) cannot answer questions about private or frequently updated codebases without hallucinating. This project implements a fast, lightweight **Retrieval-Augmented Generation (RAG)** system built for the `vllm-0.10.1` repository (1,965 files, 14,079 chunks).

The pipeline splits Python, Markdown, and text files into syntax-aware chunks, builds an incremental index using both lexical (`BM25`) and semantic (`Dense Embeddings`) search, combines results via **Reciprocal Rank Fusion (RRF)**, and generates grounded answers using `Qwen/Qwen3-0.6B` with disk-backed caching.

### Project Structure

```text
src/
├── __main__.py               # CLI entrypoint
├── models.py                 # Pydantic schemas (MinimalSource, MinimalAnswer, etc.)
├── api/                      # FastAPI web server (/health, /search, /answer)
├── chunking/                 # AST-based Python & Markdown chunkers + Text fallback
├── indexing/                 # BM25, Dense Embeddings, Manifest, and Incremental Indexer
├── retrieval/                # BM25Retriever, HybridRetriever (RRF), and QueryCache
├── generation/               # AnswerGenerator (Qwen3-0.6B) and PromptBuilder
├── evaluation/               # Character-level IoU and Recall@k Evaluator
└── cli/                      # CLI commands powered by Google Fire
```

---

## Instructions

### 1. Setup

```bash
make install
```

### 2. CLI Commands

#### A. Indexing
```bash
# Fast lexical indexing: BM25 + chunks + manifest only (incremental)
uv run python -m src index

# Build dense vector embeddings as well (required for --mode hybrid)
uv run python -m src index --embed

# Force a full re-index from scratch
uv run python -m src index --force

# Force a full re-index from scratch with embeddings
uv run python -m src index --force --embed
```

#### B. Search & Dataset Search
```bash
# Single query search (mode: bm25 or hybrid)
uv run python -m src search --query "What is PagedAttention?" --k 5 --mode bm25

# Batch dataset search
uv run python -m src search_dataset \
    --dataset_path data/public/AnsweredQuestions/dataset_docs_public.json \
    --k 5 \
    --mode bm25 \
    --save_directory data/output/search_results
```

#### C. Evaluation
```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --dataset_path data/public/AnsweredQuestions/dataset_docs_public.json
```

#### D. Answer Generation
```bash
# Single question answer
uv run python -m src answer --query "How does chunking work?" --k 5 --mode bm25

# Batch dataset answer generation
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer
```

#### E. REST API Server
```bash
uv run python -m src serve --host 127.0.0.1 --port 8000
```

```bash
# Health check
curl -X GET http://127.0.0.1:8000/health

# Search request
curl -X POST http://127.0.0.1:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "What is PagedAttention?", "k": 5, "mode": "bm25"}'

# Answer request
curl -X POST http://127.0.0.1:8000/answer \
     -H "Content-Type: application/json" \
     -d '{"query": "What is PagedAttention?", "k": 5, "mode": "bm25"}'
```

### 3. Quality & Cleanup (`Makefile`)

```bash
make lint          # Run flake8 and mypy
make lint-strict   # Run flake8 and mypy --strict
make clean         # Remove caches and .venv
```

---

## Core Algorithms & Math (Simplified Breakdown)

### 1. BM25 Lexical Search (`src/indexing/bm25/index.py`)

BM25 scores how well a chunk $d$ matches a search query $Q$ by looking at word frequency and rarity:

$$
\text{BM25}(d, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{\vert{}d\vert{}}{\text{avgdl}}\right)}
$$

**What the symbols mean:**
* $\sum_{t \in Q}$: Loop through every unique word $t$ in the user's query $Q$ and sum their scores.
* $f(t, d)$: **Term Frequency** (`freq`) — how many times word $t$ appears in chunk $d$.
* $\vert{}d\vert{}$: **Document Length** (`doc_len`) — total number of words in chunk $d$.
* $\text{avgdl}$: **Average Document Length** across all chunks in the database.
* $k_1 = 1.5, b = 0.75$: Standard BM25 tuning constants ($k_1$ limits score inflation from repeated words; $b$ penalizes very long chunks).
* $\text{IDF}(t)$: **Inverse Document Frequency** — gives higher weight to rare words (like `PagedAttention`) and lower weight to common words (like `def` or `the`):

$$
\text{IDF}(t) = \ln\left(1.0 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)
$$

*(where $N = 14,079$ is the total number of chunks, and $n(t)$ is how many chunks contain word $t$).*

**Speed Optimization:**
Instead of recalculating $k_1$ and $b$ on every search, `BM25Index.__init__` precomputes three constants once:
* `__k1_plus_1` $= k_1 + 1.0$
* `__k1_base` $= k_1 \cdot (1 - b)$
* `__b_factor` $= (k_1 \cdot b) / \text{avgdl}$

During search, we only check chunks that actually contain the query words (via an inverted index) and pick the top $k$ results using a min-heap (`heapq.nlargest`) in $\mathcal{O}(M \log k)$ time instead of sorting all $N$ chunks.

---

### 2. Code-Aware Tokenizer (`src/indexing/bm25/tokenizer.py`)

Standard tokenizers fail on code identifiers like `AsyncLLMEngine` or `vllm_v1`. `Tokenizer.tokenize` uses regex rules to split `camelCase`, `PascalCase`, `snake_case`, and numbers, and emits **both**:
1. The full lowercase token (`asyncllmengine`).
2. Its individual parts (`async`, `llm`, `engine`).

Pure alphabetic words take a fast path to skip unnecessary regex operations.

---

### 3. Syntax-Aware Chunking (`src/chunking/`)

* **`LineMap` (`line_map.py`):** Stores newline positions once and uses binary search (`bisect_right`) to convert line numbers to character indices in $\mathcal{O}(\log L)$ time (where $L$ is the number of lines).
* **`PythonChunker` (`python/`):** Uses `ast.parse` to keep classes and functions intact. If a class or function is larger than `max_chunk_size` ($2,000$ chars), it splits at method/statement boundaries and fills any gaps (`__fill_gaps`) so $100\%$ of the file is covered. Falls back to `TextChunker` on `SyntaxError`.
* **`MarkdownChunker` (`markdown/`):** Uses `markdown-it-py` to split documents by heading hierarchy (`h1` $\rightarrow$ `h6`) while keeping code blocks and tables unbroken (`BlockIsolator`).
* **`RecursiveTextSplitter` (`text/`):** Splits plain text hierarchically: Paragraphs (`\n\n`) $\rightarrow$ Sentences (`.?!`) $\rightarrow$ Lines (`\n`).
* **`Emitter` (`emitter.py`):** Merges adjacent small spans (`pack`) up to $2,000$ characters and splits oversized spans cleanly at the nearest newline or space (`rfind`).

---

### 4. Dense Embeddings & Fast Vector Search (`src/indexing/embeddings.py`)

`EmbeddingModel` converts text into $384$-number vectors using `sentence-transformers/all-MiniLM-L6-v2`:

1. **Masked Mean Pooling (`__mean_pooling`):**
   The model outputs a vector for every word in a batch. To get a single vector $\mathbf{v}_i$ for chunk $i$, we average only the real words and ignore empty padding tokens (where mask $\mathbf{M}_{i,t} = 0$):

$$
\mathbf{v}_i = \frac{\sum_{t=1}^{T} \mathbf{H}_{i, t} \cdot \mathbf{M}_{i, t}}{\max\left(1.0, \sum_{t=1}^{T} \mathbf{M}_{i, t}\right)}
$$

**What the symbols mean:**
* $\mathbf{H}_{i, t}$: The $384$-float vector for token $t$ in chunk $i$.
* $\mathbf{M}_{i, t}$: Attention mask value ($1$ = real word, $0$ = padding).
* $T$: Max tokens per chunk ($256$).

2. **$L_2$ Normalization:**
   We divide each vector by its length $\|\mathbf{v}_i\|_2$ (`normalize(p=2, dim=1)`) so every vector has a length of exactly $1.0$:

$$
\mathbf{\hat{v}}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|_2}
$$

3. **Dot-Product Cosine Similarity & $\mathcal{O}(N)$ Top-K (`VectorIndex.search`):**
   Because all document vectors $\mathbf{e}_i$ and the query vector $\mathbf{q}$ have length $1.0$, Cosine Similarity simplifies to a single dot product (`np.dot`):

$$
\text{Similarity}(\mathbf{e}_i, \mathbf{q}) = \mathbf{e}_i \cdot \mathbf{q}
$$

Instead of sorting all $14,079$ scores, we use `np.argpartition(scores, -top_k)[-top_k:]` to grab the top $k$ matches in linear $\mathcal{O}(N)$ time.

---

### 5. Hybrid Search via Reciprocal Rank Fusion (`src/retrieval/hybrid.py`)

BM25 scores (e.g., $25.4$) and Cosine similarity scores (e.g., $0.72$) are on different scales and cannot be added directly. `HybridRetriever` combines their **rankings** (1st place, 2nd place, etc.) using **Reciprocal Rank Fusion (RRF)**:

$$
\text{RRF\_Score}(d) = \sum_{m \in \{\text{bm25}, \text{vector}\}} \frac{1}{60.0 + \text{rank}_m(d)}
$$

**What the symbols mean:**
* $\text{rank}_m(d)$: The position of chunk $d$ in retriever $m$'s list ($1$ for 1st place, $2$ for 2nd place, etc.).
* $60.0$: Standard smoothing constant (`rrf_k`) so a chunk that ranks well in **both** BM25 and Vector search beats a chunk that only ranks well in one.

---

### 6. Incremental Indexing (`src/indexing/manifest.py`, `incremental.py`)

To avoid re-indexing unchanged files, `ManifestBuilder.check_status` uses a two-step check:
1. **Step 1 ($\mathcal{O}(1)$ `mtime` check):** If the file's modification time (`os.path.getmtime`) matches `manifest.pkl`, it is skipped immediately without reading the file.
2. **Step 2 (`SHA-256` content check):** If `mtime` changed, we hash the file content with `hashlib.sha256`. If the hash matches the saved hash, the content didn't actually change and we reuse its existing chunks and embeddings.

---

### 7. Disk-Synced Query & Answer Cache (`src/retrieval/cache.py`, `generator.py`)

* **`QueryCache`:** Saves search results and generated answers in `cache.pkl` (up to `max_size = 512` with FIFO eviction). Before reading, `__reload_if_stale` checks `cache.pkl`'s `mtime` so the CLI and FastAPI server stay in sync.
* **Answer Cache Key (`__build_cache_key`):** Hashes the question together with the exact retrieved source spans using `SHA-256`:

$$
\text{Key} = \text{SHA-256}(\text{question} + \text{file\_path:start:end of each source})
$$

If the question and retrieved sources haven't changed, the cached answer is returned instantly without loading `Qwen/Qwen3-0.6B`.

---

### 8. Evaluation Metrics (`src/evaluation/metrics.py`)

A retrieved source chunk $R = [s_1, e_1]$ matches a ground-truth source $G = [s_2, e_2]$ if they are from the same file and their character **Intersection over Union (IoU)** is at least $0.05$ ($5\%$):

$$
\text{IoU}(R, G) = \frac{\text{Overlap Length}}{\text{Combined Length}} = \frac{\max(0, \min(e_1, e_2) - \max(s_1, s_2))}{(e_1 - s_1) + (e_2 - s_2) - \text{Overlap Length}}
$$

**Recall@k** measures the percentage of ground-truth sources successfully found within the top $k$ retrieved chunks, averaged across all evaluated questions.

---

## Performance Analysis

Evaluated on public and private datasets over `vllm-0.10.1` (1,965 files, 14,079 chunks):

| Dataset | Questions | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dataset_docs_public.json` | 100 | **60.0%** | **77.0%** | **80.0%** | **85.0%** |
| `dataset_docs_private.json` | 100 | **61.0%** | **74.0%** | **83.0%** | **86.0%** |
| `dataset_code_public.json` | 99 | **42.4%** | **59.6%** | **66.7%** | **77.8%** |
| `dataset_code_private.json` | 100 | **46.0%** | **62.0%** | **64.0%** | **73.0%** |

### Strengths & Limitations
* **BM25 vs. Hybrid:** BM25 is strongest on exact class and function names, while Hybrid search helps on conceptual questions where wording differs from the code.
* **Main Failure Cases:**
  1. **Large Ground-Truth Classes:** When a ground-truth span covers a huge 300-line class, a $2,000$-character chunk of a single method inside that class may have $<5\%$ character overlap ($\text{IoU} < 0.05$).
  2. **Very Short Queries:** Generic 2-word queries match many files (`worker.py`, `engine.py`).
  3. **Context Window Limit:** To keep CPU generation fast, `PromptBuilder.build_context` caps context at $6,000$ characters.

---

## Design Decisions

1. **CPU & RAM Optimization:**
   * **`dtype=torch.bfloat16` & `low_cpu_mem_usage=True`:** Cuts model RAM usage in half (2 bytes per weight instead of 4).
   * **`enable_thinking=False`:** Disables Qwen3's `<think>` chain-of-thought mode (with a safe `TypeError` fallback for other models) so CPU generation finishes quickly within `max_new_tokens = 300`.
   * **100% Lazy Loading:** Heavy PyTorch models are only loaded when `answer` or `hybrid` search is actually called.
2. **Clean Exception Handling:**
   * **Internal Classes (`indexing/`, `retrieval/`):** Validate inputs and raise clear exceptions (`ValueError`, `FileNotFoundError`, `RuntimeError`, `OSError`) instead of failing silently.
   * **CLI & API Layer (`cli/`, `api/`):** Catch exceptions and print clear messages to `sys.stderr` (or return `HTTP 400 Bad Request` in FastAPI) so the program never crashes with raw tracebacks.
3. **Strict Code Quality:**
   * All 52 Python files enforce OOP encapsulation, functions $\le 30$ lines, sorted imports, and zero `flake8` or `mypy --strict` errors.

---

## Challenges Faced

* **Code Identifier Mismatch in Lexical Search (`CamelCase` / `snake_case`):** Standard regex tokenizers either keep `AsyncLLMEngine` intact (missing natural-language queries like `"async llm engine"`) or split it destructively (losing exact identifier matches). Solved via a dual-emission `Tokenizer` (`src/indexing/bm25/tokenizer.py`) that emits both the lowered full identifier and its decomposed sub-tokens, with a `raw.islower() and raw.isalpha()` fast-path.
* **Preserving Syntactic & Structural Integrity Under `max_chunk_size = 2000`:** Naive fixed-size character slicing breaks Python functions and Markdown tables mid-block. Solved using AST-guided span extraction (`ast.parse` in `PythonChunker` with `__fill_gaps`, and `markdown-it-py` in `MarkdownChunker` with `BlockIsolator`) combined with `Emitter` greedy span packing and backward newline/space boundary splitting.
* **Indexing Speed vs. CPU Embedding Overhead:** Computing 384-d embeddings for 14,079 chunks on a CPU-only machine takes minutes, whereas BM25 indexing takes <5 seconds. Solved by decoupling embedding generation behind `--embed`, using `torch.bfloat16` under `torch.inference_mode()`, and implementing two-stage incremental indexing (`mtime` + `SHA-256`) with `zip(chunks, embs, strict=True)`.
* **CPU Latency & Reasoning Token Exhaustion on `Qwen/Qwen3-0.6B`:** By default, `Qwen3-0.6B` generates long `<think>...</think>` blocks that consume the `max_new_tokens = 300` budget on CPU before outputting the final answer. Solved by passing `enable_thinking=False` to `apply_chat_template` (with a `TypeError` fallback for alternative models) and stripping residual `<think>` tags in `PromptBuilder`.

---

## Resources & AI Usage

### References
* [BM25 & Robertson-Spärck Jones IDF](https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25.pdf)
* [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormack-sigir09-rrf.pdf)
* [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) & [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)

### AI Usage
AI tools were used as a pair-programming assistant to format LaTeX equations, structure README tables, and verify static type annotations for `flake8` and `mypy --strict`. All core algorithms, chunking logic, and system architecture were implemented and verified by `adraji`.
