# Mini-RAG: Ask Questions About a Machine Learning Book

A fully local Retrieval-Augmented Generation (RAG) system, served as a **FastAPI** service and packaged as a **Docker** image,
that answers questions about *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (560+ page PDF).
Every answer comes with the book pages it was based on, so it can be checked.
Everything runs on your machine with [Ollama](https://ollama.com): no API keys, and no data leaves your computer.

## How it works

```
PDF -> PyPDFLoader -> clean text -> RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
    -> nomic-embed-text embeddings -> Chroma vector store            (ingest.py)

POST /ask -> retriever (top-k chunks) -> grounded prompt -> local LLM (Ollama)
          -> answer + source pages                                   (app.py, rag.py)
```

The prompt tells the LLM to use **only** the retrieved context, and to reply
*"I don't know based on the provided book context."* when the answer is not there.

## Tech stack

| Component | Tool |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| Orchestration | LangChain |
| PDF loading | PyPDFLoader (pypdf) |
| Vector store | Chroma |
| Embeddings | Ollama `nomic-embed-text` |
| LLM | Ollama (`llama3` by default, configurable) |
| Packaging | Docker (image on Docker Hub) |

## Project structure

```
app.py             FastAPI app (/ask, /health)
rag.py             retriever, prompt, and answer chain
ingest.py          builds the Chroma index from the PDF
requirements.txt
Dockerfile
.dockerignore
.gitignore
```

## Quick start (local)

### 1. Install Ollama and pull the models
```bash
ollama pull nomic-embed-text
ollama pull llama3
```

### 2. Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

### 3. Build the index (once)
The book is not included in this repository. Put your own copy of the PDF in a `data/` folder, then run:
```bash
python ingest.py --pdf data/book.pdf
```
This cleans the text, splits it into chunks, embeds them, and saves them to `chroma_db/`.
It takes a few minutes. Use `--reset` to rebuild the index from scratch.

### 4. Run the API
```bash
uvicorn app:app --reload
```
Open http://localhost:8000/docs for the interactive Swagger UI.

## API

### `POST /ask`
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is overfitting?"}'
```

Example response (shortened):
```json
{
  "question": "What is overfitting?",
  "answer": "Overfitting is when the model performs well on the training data, but it does not generalize well. It means that the model is too complex relative to the amount and noisiness of the training data.",
  "sources": [
    {"page": "28", "snippet": "As you might guess, underfitting is the opposite of overfitting: ..."},
    {"page": "27", "snippet": "..."}
  ]
}
```

### `GET /health`
Returns the status (`ok` or `empty_index`), the LLM model name, and the number of indexed chunks.

### Error codes

| Status | Meaning |
|---|---|
| 422 | Invalid input (the question must be 3 to 1000 characters) |
| 502 | Ollama returned an error (for example the model is not pulled) |
| 503 | Ollama is unreachable, or the index is empty (run `ingest.py`) |

## Configuration

Set through environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is running |
| `LLM_MODEL` | `llama3:latest` | Chat model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_DIR` | `chroma_db` | Vector store folder |
| `COLLECTION_NAME` | `hands_on_ml_book` | Chroma collection |
| `TOP_K` | `5` | Chunks retrieved per question |
| `PDF_PATH` | `data/book.pdf` | Default PDF for `ingest.py` |

## Docker

The image contains the API code only. The PDF, the index, and Ollama stay on your machine:
the index is mounted as a volume, and the container reaches Ollama through `host.docker.internal`.

```bash
docker pull mohammedwaleederfaan/mini-rag
```
Docker Hub: https://hub.docker.com/r/mohammedwaleederfaan/mini-rag

**1. Build the index once** (Windows cmd; use `$(pwd)` on macOS/Linux or `${PWD}` in PowerShell):
```bash
docker run --rm ^
  -v "%cd%\data:/app/data" -v "%cd%\chroma_db:/app/chroma_db" ^
  mohammedwaleederfaan/mini-rag python ingest.py --pdf data/book.pdf
```

**2. Run the API:**
```bash
docker run -p 8000:8000 -v "%cd%\chroma_db:/app/chroma_db" mohammedwaleederfaan/mini-rag
```

To use a different model: add `-e LLM_MODEL=qwen2.5-coder:7b`. If port 8000 is taken, use `-p 8001:8000`.
On Linux, add `--add-host=host.docker.internal:host-gateway` and start Ollama with `OLLAMA_HOST=0.0.0.0`.

To build the image yourself: `docker build -t mini-rag .`

## Evaluation

I wrote a test set of 22 questions: 20 covering chapters 1-8 and 11 of the book (ML basics, classification,
training models, SVMs, decision trees, ensembles, PCA, deep nets), plus 2 out-of-scope questions
(e.g. "Who won the 2018 FIFA World Cup?") to check that the system refuses when the context has no answer.
Each answer was compared by hand against a reference answer, and the retrieved page numbers were checked.

The runs below used `qwen2.5-coder:7b` as the LLM, before switching the default to `llama3`.

**Findings**
- Retrieval was accurate: the relevant page was among the top results for nearly every question.
- Both out-of-scope questions were correctly refused.
- Most answers matched the book. Weak spots: some answers added details that were not in the retrieved context
  (for example, extra techniques in the vanishing gradients answer), and one answer mixed up the fixes for
  overfitting and underfitting.
- Moving from `k=3` to `k=5` and tightening the prompt reduced incomplete and off-context answers.

## Known limitations
- Smaller local models can still go beyond the provided context despite the prompt.
- The top-k results sometimes include unrelated chunks; the LLM usually ignores them, but they can add noise.
- Text extracted from the PDF has some spacing artifacts (for example "da ta"), which slightly weakens the embeddings.
- Chunk boundaries can split a definition across two chunks.
- Evaluation was done manually and has not yet been repeated with `llama3`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `/health` shows `empty_index` | Run `ingest.py`, or mount the folder that contains the index (`-v ...:/app/chroma_db`) |
| `Cannot reach Ollama` | Start Ollama; in Docker check `OLLAMA_BASE_URL` |
| `model ... not found` (502) | `ollama pull <model>` or set `LLM_MODEL` to a model from `ollama list` |
| `port is already allocated` | Stop the other container (`docker ps`, `docker stop <id>`) or use another host port |
| `ingest.py` says the collection already has chunks | Rerun with `--reset` |

## Author
**Mohammed Waleed Erfaan** | [GitHub](https://github.com/mohammedwaleederfaan-coder) | [LinkedIn](https://linkedin.com/in/mohammed-waleed-0065b9409)