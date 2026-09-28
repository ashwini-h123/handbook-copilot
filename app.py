import os
import streamlit as st
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# -------------------- CONFIG --------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in .env file. Please add it to your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(
    page_title="Handbook Copilot",
    page_icon="📚",
    layout="wide",
)

# -------------------- GOOGLE EMBEDDING FUNCTION --------------------
def google_embed(texts, task_type="retrieval_document"):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type=task_type
    )
    return result["embedding"]


class GoogleEmbeddingFunction:
    def __init__(self, task_type="retrieval_document"):
        self.task_type = task_type

    def __call__(self, input):
        return [google_embed(text, self.task_type) for text in input]

    def embed_documents(self, texts):
        return [google_embed(text, "retrieval_document") for text in texts]

    def embed_query(self, input):
        if isinstance(input, str):
            return google_embed(input, "retrieval_query")
        return [google_embed(text, "retrieval_query") for text in input]

    def name(self):
        return "google-gemini-embedding-001"


# -------------------- VECTOR DB --------------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(
        name="handbook",
        embedding_function=GoogleEmbeddingFunction(),
    )

collection = get_collection()


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.title("📚 Handbook Copilot")
    st.caption("RAG-powered Q&A over your PDF")
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. PDF is split into chunks")
    st.markdown("2. Chunks are embedded into ChromaDB")
    st.markdown("3. Your question retrieves top 5 chunks")
    st.markdown("4. Gemini generates a cited answer")
    st.divider()
    st.caption("Built with RAG + Gemini + ChromaDB")


# -------------------- MAIN UI --------------------
st.title("📚 Institutional Handbook Copilot")
st.caption("Ask natural language questions. Get answers with page citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about the handbook..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching the handbook..."):
            results = collection.query(query_texts=[prompt], n_results=5)
            docs = results["documents"][0]
            metas = results["metadatas"][0]

            context_parts = []
            for d, m in zip(docs, metas):
                context_parts.append(f"[Page {m['page']}]\n{d}")
            context = "\n\n---\n\n".join(context_parts)

            model = genai.GenerativeModel("gemini-3.8-flash")
            response = model.generate_content(
                f"""You are a helpful assistant for the CMR University Student Handbook.

Rules:
- Answer the user's question using ONLY the context below.
- If the answer is not in the context, say: "I couldn't find that in the handbook."
- ALWAYS cite the page number(s) like this: (Page 12)
- Be concise and accurate.

CONTEXT:
{context}

QUESTION: {prompt}

ANSWER:"""
            )

            answer = response.text
            st.markdown(answer)

            with st.expander("📖 View Sources"):
                for d, m in zip(docs, metas):
                    st.markdown(f"**Page {m['page']}**")
                    st.caption(d[:400] + ("..." if len(d) > 400 else ""))
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})