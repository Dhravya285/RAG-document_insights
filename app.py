import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
import tempfile
import re

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(page_title="Smart Document Insights", layout="wide")

st.title("📄 Smart Document Insights — RAG")
st.caption("Upload a PDF and ask grounded questions. Powered by FAISS + Local LLM")

# -----------------------------------
# Load LLM
# -----------------------------------
@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.0,
        num_predict=600,
        num_ctx=4096
    )

llm = load_llm()

# -----------------------------------
# Session State for Chat
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# -----------------------------------
# PDF Upload
# -----------------------------------
uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save temp PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Load + Split
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=60
        )
        chunks = splitter.split_documents(docs)

        # Embed + FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore

    st.success(f"PDF processed — {len(chunks)} chunks indexed")

# -----------------------------------
# Chat UI
# -----------------------------------
if st.session_state.vectorstore:

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    query = st.chat_input("Ask something about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # ---------------------------------
                # MMR Retrieval
                # ---------------------------------
                docs = st.session_state.vectorstore.max_marginal_relevance_search(
                    query,
                    k=4,
                    fetch_k=12,
                    lambda_mult=0.65
                )

                if not docs:
                    docs = st.session_state.vectorstore.similarity_search(query, k=4)

                # Build Context
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    context_parts.append(f"[Source {i}]\n{doc.page_content}")

                context = "\n\n---\n\n".join(context_parts)
                context = context[:2500]

                # ---------------------------------
                # Prompt
                # ---------------------------------
                prompt = f"""
You are a strict document QA assistant.

RULES:
- Use ONLY given sources
- If answer missing → say "Not in document"
- Cite [Source X] for every claim
- Do not guess

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

                # Generate
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)

                st.write(answer)

                # ---------------------------------
                # Confidence (Grounding)
                # ---------------------------------
                answer_words = set(answer.lower().split())
                context_lower = context.lower()

                common_words = {"the","a","an","and","or","in","on","at","to","for","of"}
                content_words = [w for w in answer_words if w not in common_words and len(w) > 3]

                grounded = sum(1 for w in content_words if w in context_lower)
                confidence = int((grounded / max(len(content_words),1)) * 100)

                st.progress(confidence / 100)
                st.caption(f"Confidence (grounded): {confidence}%")

                # ---------------------------------
                # Sources
                # ---------------------------------
                with st.expander("📚 Retrieved Sources"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.write(d.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})
