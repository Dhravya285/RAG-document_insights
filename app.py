import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import tempfile


st.set_page_config(page_title="Smart Document Insights", layout="wide")
st.title("📄 Smart Document Insights — Agentic RAG")
st.caption("Routing Agent + FAISS + Local LLM (Semantic / Multi-Query / Hybrid)")


@st.cache_resource
def load_llm():
    return ChatOllama(
        model="llama3.2",
        temperature=0.0,
        num_predict=600,
        num_ctx=4096
    )

llm = load_llm()


def choose_strategy(query):

    prompt = f"""
Classify query complexity.

Return ONLY one:
semantic OR multi-query OR hybrid

Rules:
- semantic → simple fact
- multi-query → compare / analyze / multi-topic
- hybrid → technical / exact term

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



def hybrid_search(vectorstore, query, k=6):

    docs = vectorstore.similarity_search(query, k=k)
    keywords = query.lower().split()

    ranked = sorted(
        docs,
        key=lambda d: sum(word in d.page_content.lower() for word in keywords),
        reverse=True
    )

    return ranked[:4]



if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None



uploaded_file = st.file_uploader("📄 Upload PDF", type="pdf")

if uploaded_file:

    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=60
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        st.session_state.vectorstore = vectorstore

    st.success(f"PDF processed — {len(chunks)} chunks indexed")



if st.session_state.vectorstore:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask something about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                strategy = choose_strategy(query)
                st.caption(f"🧠 Strategy: **{strategy}**")

                vectorstore = st.session_state.vectorstore

              
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

                
                docs = vectorstore.max_marginal_relevance_search(
                    query,
                    k=4,
                    fetch_k=12,
                    lambda_mult=0.65
                )

             
                context_parts = []
                for i, doc in enumerate(docs, 1):
                    context_parts.append(f"[Source {i}]\n{doc.page_content}")

                context = "\n\n---\n\n".join(context_parts)
                context = context[:2500]

           
                prompt = f"""
You are a strict document QA assistant.

Rules:
- Use ONLY given sources
- If answer missing → say "Not in document"
- Cite [Source X]
- Do not guess

Context:
{context}

Question:
{query}

Answer:
"""

                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, "content") else str(response)

                st.write(answer)

                # -----------------------------------
                # Confidence (Grounding)
                # -----------------------------------
                answer_words = set(answer.lower().split())
                context_lower = context.lower()

                common_words = {"the","a","an","and","or","in","on","at","to","for","of"}
                content_words = [w for w in answer_words if w not in common_words and len(w) > 3]

                grounded = sum(1 for w in content_words if w in context_lower)
                confidence = int((grounded / max(len(content_words),1)) * 100)

                st.progress(confidence / 100)
                st.caption(f"Confidence (grounded): {confidence}%")

               
                with st.expander("📚 Retrieved Sources"):
                    for i, d in enumerate(docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.write(d.page_content)

        st.session_state.messages.append({"role": "assistant", "content": answer})
