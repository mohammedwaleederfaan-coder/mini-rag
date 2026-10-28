# Mini-RAG: Ask Questions About a Machine Learning Book

A fully local Retrieval-Augmented Generation (RAG) system that answers questions about
*Hands-On Machine Learning with Scikit-Learn and TensorFlow* (500+ page PDF).
Everything runs on your machine with [Ollama](https://ollama.com), with no API keys and no data leaving your computer.

## How it works

```
PDF -> PyPDFLoader -> clean text -> RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
    -> nomic-embed-text embeddings -> Chroma vector store
Question -> retriever (top-k chunks) -> prompt -> local LLM (Ollama) -> answer + page references
```

1. **Load & clean**: the PDF is loaded page by page and the text is normalized (Unicode NFKC, whitespace cleanup).
2. **Chunk**: pages are split into 1000-character chunks with 200 overlap, keeping page metadata.
3. **Embed & store**: chunks are embedded with `nomic-embed-text` and stored in a persistent Chroma collection, in batches.
4. **Retrieve & answer**: the top-k most similar chunks are injected into a prompt that tells the LLM to answer
   **only** from the context, or say *"I don't know based on the provided book context."*

## Tech stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| PDF loading | PyPDFLoader (pypdf) |
| Vector store | Chroma |
| Embeddings | Ollama `nomic-embed-text` |
| LLM | Ollama (`llama3`, `qwen2.5-coder:7b`, or `llama3.2`) |

## Setup

### 1. Install Ollama and pull the models
```bash
ollama pull nomic-embed-text
ollama pull llama3
```

### 2. Install Python dependencies
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Add the book
Place the PDF in a `data/` folder and update the `path` variable in the notebook.
The book is not included in this repository.

### 4. Run
Open the notebook and run the cells in order:
1. Load and clean the PDF
2. Split into chunks
3. Build the Chroma vector store (done once; it is saved to disk)
4. Ask questions with `rag_chain.invoke("What is overfitting?")`

## Evaluation

I wrote a test set of 22 questions: 20 covering chapters 1-8 and 11 of the book (ML basics, classification,
training models, SVMs, decision trees, ensembles, PCA, deep nets), plus 2 out-of-scope questions
(e.g. "Who won the 2018 FIFA World Cup?") to check that the system refuses to answer when the context has no answer.

Each answer was compared against a reference answer, and the retrieved page numbers were checked.

**What I found**
- Retrieval was accurate: the relevant page was among the top results for nearly every question.
- Both out-of-scope questions were correctly refused.
- Most answers matched the book. Weak spots: answers sometimes added details not in the retrieved context
  (e.g. extra techniques for the vanishing gradients question), and one answer mixed up the fixes for overfitting and underfitting.

**What helped**: increasing `k` from 3 to 5 and tightening the prompt reduced incomplete answers.

## Known limitations
- Smaller local models occasionally go beyond the provided context despite the prompt.
- Chunk boundaries can split a definition across two chunks.
- No automatic scoring yet; evaluation was done manually.


## Author
**Mohammed Waleed Erfaan** | [GitHub](https://github.com/mohammedwaleederfaan-coder) | [LinkedIn](https://linkedin.com/in/mohammed-waleed-0065b9409)