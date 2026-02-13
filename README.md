📄 Smart Document Insights

An AI-powered system that allows users to chat with documents (PDF) using Retrieval-Augmented Generation (RAG). The application performs semantic search over document content and generates context-aware answers grounded in the uploaded file.

🚀 Features

📂 Load and analyze PDF documents

🔍 Semantic search using embeddings

🤖 Context-aware Question Answering (RAG)

⚡ Fast similarity search using FAISS vector database

🧠 Uses HuggingFace embeddings (no training required)

🌐 Simple interactive UI with Streamlit

📊 Accurate, grounded answers from document (reduces hallucination)

🧠 How It Works (RAG Pipeline)

PDF → Extract text

Split text into small chunks

Convert chunks → Embeddings (vector representation)

Store vectors in FAISS (Vector Database)

User asks question

System retrieves most relevant chunks

LLM generates answer using retrieved context

Flow:
Question → Retrieval → Context Injection → Answer Generation

🏗️ Tech Stack
Technology	Purpose
Python	Core programming language
LangChain	RAG pipeline & orchestration
HuggingFace Embeddings	Convert text → vectors
FAISS	Vector database for similarity search
PyPDF	PDF text extraction
Streamlit	Web UI for interaction
📂 Project Structure
rag-doc-ai/
│── app.py              # Streamlit UI + QA
│── ingest.py           # PDF → FAISS index creation
│── sample.pdf          # Input document
│── faiss_index/        # Saved vector database
│── requirements.txt
│── README.md

⚙️ Installation
1. Clone repo
git clone https://github.com/yourusername/rag-doc-ai.git
cd rag-doc-ai

2. Install dependencies
pip install -r requirements.txt


If requirements.txt not present:

pip install langchain langchain-community sentence-transformers faiss-cpu pypdf streamlit

▶️ Usage
Step 1 — Create Vector Database
python ingest.py

Step 2 — Run Application
streamlit run app.py


Open browser → http://localhost:8501

📸 Example

Ask:

What is the main topic of this document?

AI responds using actual document context.

🎯 Applications

Chat with PDFs

Research paper analysis

Legal/finance document QA

Knowledge base assistants

Enterprise document intelligence
