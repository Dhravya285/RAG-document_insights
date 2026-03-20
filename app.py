import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.tools import DuckDuckGoSearchRun
import wikipedia
import tempfile
import re


st.set_page_config(page_title="Smart Document Insights", layout="wide")
st.title("📄 Smart Document Insights — Agentic RAG")
st.caption("Routing Agent + FAISS + Web Search + Wikipedia + Local LLM")


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.0,
        num_predict=600,
        num_ctx=4096
    )

llm = load_llm()


# ─────────────────────────────────────────────
# Agentic Tools
# ─────────────────────────────────────────────

@st.cache_resource
def load_search_tool():
    return DuckDuckGoSearchRun()

search_tool = load_search_tool()


def web_search(query: str) -> str:
    """Run a DuckDuckGo web search and return results."""
    try:
        result = search_tool.run(query)
        return result if result else "No results found."
    except Exception as e:
        return f"Web search failed: {e}"


def wikipedia_search(query: str) -> str:
    """Fetch a Wikipedia summary for a query."""
    try:
        summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # Pick the first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=5)
            return f"[Disambiguation → {e.options[0]}]\n{summary}"
        except Exception:
            return f"Wikipedia disambiguation failed for: {query}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for: {query}"
    except Exception as e:
        return f"Wikipedia search failed: {e}"


# ─────────────────────────────────────────────
# Routing Agent — decides where to get the answer
# ─────────────────────────────────────────────

def route_query(query: str, has_document: bool) -> str:
    """
    Decide the best source to answer the query.
    Returns one of: 'document', 'web_search', 'wikipedia', 'hybrid_all'
    """
    q = query.lower()

    # Rule-based fast routing — reliable, no LLM needed
    live_keywords = ["today", "latest", "current", "price", "news", "2024", "2025", "2026",
                     "right now", "stock", "weather", "live", "trending", "recently"]
    if any(kw in q for kw in live_keywords):
        return "web_search"

    wiki_keywords = ["what is", "who is", "who was", "what was", "explain", "define",
                     "history of", "how does", "tell me about", "meaning of", "revolution",
                     "war", "invention", "founder", "capital of", "born in"]
    if any(kw in q for kw in wiki_keywords) and not has_document:
        return "wikipedia"

    if has_document:
        doc_keywords = ["document", "pdf", "this file", "mentioned", "according to",
                        "in the", "from the", "summarize"]
        if any(kw in q for kw in doc_keywords):
            return "document"

    # LLM fallback for ambiguous queries
    doc_context = "A PDF document is loaded." if has_document else "No document is loaded."
    prompt = f"""You are a routing agent. Pick ONE source for this query.

Context: {doc_context}

Options:
- document   → query is specifically about the uploaded PDF
- web_search → recent news, prices, live data, current events
- wikipedia  → general knowledge, history, science, definitions, concepts
- hybrid_all → needs both the document AND external knowledge

Reply with ONLY one of these exact words: document, web_search, wikipedia, hybrid_all

Query: {query}
Answer:"""

    res = llm.invoke(prompt)
    txt = res.content.strip().lower().split()[0]

    if "web" in txt:
        return "web_search"
    elif "wiki" in txt:
        return "wikipedia"
    elif "hybrid" in txt:
        return "hybrid_all"
    elif has_document and "doc" in txt:
        return "document"
    else:
        return "wikipedia" if any(kw in q for kw in wiki_keywords) else "web_search"


# ─────────────────────────────────────────────
# RAG Strategy Selector
# ─────────────────────────────────────────────

def choose_strategy(query: str) -> str:
    prompt = f"""
Classify query complexity.

Return ONLY one:
semantic OR multi-query OR hybrid

Rules:
- semantic    → simple fact lookup
- multi-query → compare / analyze / multi-topic
- hybrid      → technical term or exact phrase match needed

Query: {query}
"""
    res = llm.invoke(prompt)
    txt = res.content.lower()

    if "multi" in txt:
        return "multi-query"
    elif "hybrid" in txt:
        return "hybrid"
    else:
        return "semantic"


# ─────────────────────────────────────────────
# Hybrid (keyword-boosted) Search
# ─────────────────────────────────────────────

def hybrid_search(vectorstore, query: str, k: int = 6):
    docs = vectorstore.similarity_search(query, k=k)
    keywords = query.lower().split()
    ranked = sorted(
        docs,
        key=lambda d: sum(w in d.page_content.lower() for w in keywords),
        reverse=True,
    )
    return ranked[:4]


# ─────────────────────────────────────────────
# Retrieve from vectorstore (with strategy)
# ─────────────────────────────────────────────

def retrieve_from_document(vectorstore, query: str, strategy: str) -> list:
    if strategy == "semantic":
        docs = vectorstore.similarity_search(query, k=4)
    elif strategy == "multi-query":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        multi = MultiQueryRetriever.from_llm(retriever, llm)
        docs = multi.invoke(query)
    elif strategy == "hybrid":
        docs = hybrid_search(vectorstore, query)
    else:
        docs = vectorstore.similarity_search(query, k=4)

    # Always re-rank with MMR for diversity
    docs = vectorstore.max_marginal_relevance_search(
        query, k=4, fetch_k=12, lambda_mult=0.65
    )
    return docs


# ─────────────────────────────────────────────
# Build prompt with context
# ─────────────────────────────────────────────

def build_prompt(query: str, context: str, source_label: str) -> str:
    return f"""You are a helpful QA assistant. Answer the question using the context below.

Rules:
- Use the context as your primary source
- Be concise and informative
- Cite the source (e.g. [Wikipedia], [Web], [Source 1])
- Only say not found if the context is truly empty or completely unrelated

Context ({source_label}):
{context}

Question: {query}

Answer:"""


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ─────────────────────────────────────────────
# Sidebar — Tool Status
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("🛠️ Agentic Tools")
    st.success("✅ Web Search (DuckDuckGo)")
    st.success("✅ Wikipedia")
    st.info("📄 Document RAG (FAISS + MMR)")
    st.markdown("---")
    st.caption("The routing agent picks the best tool automatically.")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────
# PDF Upload
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader("📄 Upload PDF (optional — tools work without it too)", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
        chunks = splitter.split_documents(raw_docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.vectorstore = vectorstore

    st.success(f"✅ PDF processed — {len(chunks)} chunks indexed")


# ─────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask anything — from your document or the world...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Routing query to best tool..."):

            has_document = st.session_state.vectorstore is not None

            # ── Step 1: Route ──
            route = route_query(query, has_document)

            tool_icons = {
                "document":   "📄 Document",
                "web_search": "🌐 Web Search",
                "wikipedia":  "📖 Wikipedia",
                "hybrid_all": "🔀 Document + Web + Wikipedia",
            }
            st.caption(f"🧭 Routed to: **{tool_icons.get(route, route)}**")

            context = ""
            source_label = "context"
            docs_used = []

            # ── Step 2: Fetch from tool(s) ──

            if route == "document" and has_document:
                strategy = choose_strategy(query)
                st.caption(f"🧠 RAG Strategy: **{strategy}**")
                docs_used = retrieve_from_document(st.session_state.vectorstore, query, strategy)
                parts = [f"[Source {i}]\n{d.page_content}" for i, d in enumerate(docs_used, 1)]
                context = "\n\n---\n\n".join(parts)[:2500]
                source_label = "document"

            elif route == "web_search":
                st.caption("🌐 Searching the web...")
                web_result = web_search(query)
                context = f"[Web Search Results]\n{web_result}"
                source_label = "web search"

            elif route == "wikipedia":
                st.caption("📖 Fetching from Wikipedia...")
                wiki_result = wikipedia_search(query)
                context = f"[Wikipedia]\n{wiki_result}"
                source_label = "Wikipedia"

            elif route == "hybrid_all":
                parts = []

                if has_document:
                    strategy = choose_strategy(query)
                    st.caption(f"🧠 RAG Strategy: **{strategy}**")
                    docs_used = retrieve_from_document(st.session_state.vectorstore, query, strategy)
                    doc_parts = [f"[Doc Source {i}]\n{d.page_content}" for i, d in enumerate(docs_used, 1)]
                    parts.append("\n\n".join(doc_parts))

                st.caption("🌐 Searching the web...")
                parts.append(f"[Web Search]\n{web_search(query)}")

                st.caption("📖 Fetching from Wikipedia...")
                parts.append(f"[Wikipedia]\n{wikipedia_search(query)}")

                context = "\n\n---\n\n".join(parts)[:3500]
                source_label = "document + web + Wikipedia"

            else:
                # Fallback: no doc, use web
                st.caption("🌐 No document — falling back to web search...")
                web_result = web_search(query)
                context = f"[Web Search Results]\n{web_result}"
                source_label = "web search"

            # ── Step 3: Generate answer ──
            prompt = build_prompt(query, context, source_label)
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)

            st.write(answer)

            # ── Step 4: Confidence (only meaningful for document answers) ──
            if route in ("document", "hybrid_all") and docs_used:
                answer_words = set(answer.lower().split())
                context_lower = context.lower()
                common = {"the","a","an","and","or","in","on","at","to","for","of"}
                content_words = [w for w in answer_words if w not in common and len(w) > 3]
                grounded = sum(1 for w in content_words if w in context_lower)
                confidence = int((grounded / max(len(content_words), 1)) * 100)

                st.progress(confidence / 100)
                st.caption(f"📊 Grounding confidence: {confidence}%")

            # ── Step 5: Show sources ──
            with st.expander("📚 Retrieved Context"):
                if docs_used:
                    for i, d in enumerate(docs_used, 1):
                        st.markdown(f"**Doc Source {i}:**")
                        st.write(d.page_content)
                else:
                    st.text(context[:1500])

    st.session_state.messages.append({"role": "assistant", "content": answer})