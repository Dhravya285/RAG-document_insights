# 📄 Smart Document Insights — Agentic RAG System

> An intelligent document Q&A system powered by a local LLM routing agent that dynamically selects between FAISS vector search, DuckDuckGo web search, and Wikipedia — so you're never limited to just your uploaded document.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-black?logo=ollama)](https://ollama.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-brightgreen)](https://github.com/facebookresearch/faiss)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://langchain.com)

---

## 🧠 How It Works

The system uses a **two-stage agentic pipeline**:

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│        Routing Agent            │
│  Rule-based + LLM fallback      │
└────────────┬────────────────────┘
             │
     ┌───────┼───────────┬──────────────┐
     ▼       ▼           ▼              ▼
 📄 FAISS  🌐 DuckDuckGo  📖 Wikipedia  🔀 All Three
 Document  Web Search     General KB    (hybrid_all)
     │       │               │              │
     └───────┴───────────────┴──────────────┘
                         │
                         ▼
               ┌─────────────────┐
               │  Local LLM      │
               │  (llama3.2)     │
               │  Grounded Answer│
               └─────────────────┘
```

### Routing Logic

| Query Type | Route | Example |
|---|---|---|
| About uploaded PDF | `document` | *"Summarize this file"* |
| Current events / prices | `web_search` | *"Latest AI news today"* |
| History / science / concepts | `wikipedia` | *"What is the French Revolution?"* |
| Cross-source questions | `hybrid_all` | *"How does this doc relate to current trends?"* |

---

## ✨ Features

- **🧭 Agentic Routing** — Rule-based keyword matching + LLM fallback picks the best tool per query automatically
- **📄 PDF RAG** — Upload any PDF; it's chunked, embedded, and indexed into FAISS for fast semantic retrieval
- **🌐 Web Search** — DuckDuckGo integration for live, current, real-world answers
- **📖 Wikipedia** — Instant factual summaries for general knowledge queries
- **🔀 Hybrid Mode** — Combines all three sources when a query spans document + world knowledge
- **🎯 Three RAG Strategies** — Semantic, Multi-Query, and Hybrid keyword-boosted retrieval
- **📊 MMR Reranking** — Maximal Marginal Relevance ensures diverse, non-redundant retrieved chunks
- **📈 Confidence Score** — Grounding metric shows how well the answer is anchored in retrieved context
- **💬 Multi-turn Chat** — Full conversation history with role-based chat UI
- **🗑️ Clear Chat** — Reset conversation anytime from the sidebar

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama (llama3.2) — runs fully locally |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace |
| Vector Store | FAISS |
| RAG Framework | LangChain + LangChain Community |
| Web Search | DuckDuckGo (`ddgs`) |
| General Knowledge | Wikipedia API |
| PDF Loader | PyPDFLoader |
| UI | Streamlit |

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 2. Pull the LLM

```bash
ollama pull llama3.2
```

### 3. Clone the repo

```bash
git clone https://github.com/Dhravya285/RAG-document_insights
cd RAG-document_insights
```

### 4. Install dependencies

```bash
pip install streamlit langchain langchain-community langchain-ollama
pip install langchain-text-splitters sentence-transformers faiss-cpu
pip install pypdf ddgs wikipedia
pip install "numpy<2"   # required for scipy/sklearn compatibility
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 📦 Project Structure

```
RAG-document_insights/
├── app.py          # Main Streamlit app — all logic lives here
└── README.md       # This file
```

---

## 💡 Example Queries

Try these after launching the app:

| Query | Expected Route |
|---|---|
| `"What is quantum computing?"` | 📖 Wikipedia |
| `"Latest AI news today"` | 🌐 Web Search |
| `"Summarize the uploaded document"` | 📄 Document |
| `"Who founded Microsoft?"` | 📖 Wikipedia |
| `"Current Bitcoin price"` | 🌐 Web Search |
| `"What does this document say about X?"` | 📄 Document |
| `"How does this compare to industry trends?"` | 🔀 Hybrid |

---

## ⚙️ Configuration

You can tweak these parameters directly in `app.py`:

```python
# LLM settings
ChatOllama(
    model="llama3.2",     # swap with any Ollama model
    temperature=0.0,       # 0 = deterministic
    num_predict=600,       # max output tokens
    num_ctx=4096           # context window
)

# Chunking
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=60
)

# MMR reranking
vectorstore.max_marginal_relevance_search(
    query, k=4, fetch_k=12, lambda_mult=0.65
)
```

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `numpy` version conflicts | `pip install "numpy<2"` |
| `No module named ddgs` | `pip uninstall duckduckgo-search && pip install ddgs` |
| `MultiQueryRetriever` import error | Use `from langchain.retrievers.multi_query import MultiQueryRetriever` |
| Ollama connection refused | Make sure Ollama is running: `ollama serve` |
| Slow first query | Normal — HuggingFace model downloads on first run |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with 🦙 Ollama · ⚡ FAISS · 🔗 LangChain · 🎈 Streamlit</p>
