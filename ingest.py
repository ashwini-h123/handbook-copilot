import os
import chromadb
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/handbook.pdf"
CHROMA_PATH = "./chroma_db"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)


def google_embed(texts):
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=texts,
        task_type="retrieval_document"
    )
    return result["embedding"]


class GoogleEmbeddingFunction:
    def __call__(self, input):
        return [google_embed(text) for text in input]

    def name(self):
        return "google-text-embedding-004"


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for p in pages:
        for c in splitter.split_text(p["text"]):
            chunks.append({"page": p["page"], "text": c.strip()})
    return chunks


def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at: {PDF_PATH}")
        return

    print("📖 Reading PDF...")
    pages = load_pdf(PDF_PATH)
    print(f"   ✓ Loaded {len(pages)} pages")

    print("✂️  Splitting into chunks...")
    chunks = chunk_pages(pages)
    print(f"   ✓ Created {len(chunks)} chunks")

    print("🧠 Creating embeddings with Google API...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = GoogleEmbeddingFunction()

    try:
        client.delete_collection("handbook")
        print("   ✓ Cleared previous collection")
    except Exception:
        pass

    collection = client.create_collection(
        name="handbook",
        embedding_function=emb_fn,
    )

    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[{"page": c["page"]} for c in batch],
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )
        print(f"   ✓ Added {i + len(batch)}/{len(chunks)} chunks")

    print(f"\n✅ Done! Vector DB saved to '{CHROMA_PATH}'")
    print("   Now run: streamlit run app.py")


if __name__ == "__main__":
    main()