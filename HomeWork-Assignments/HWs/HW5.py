# HW5.py — Short-term Memory Chatbot with Vector Search
import os, tempfile
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
from openai import OpenAI as OpenAIClient

# ======== Vector Search Helper ========
def get_relevant_info(query: str, collection, n_results=3) -> str:
    """Retrieve relevant info for query from Chroma collection."""
    results = collection.query(query_texts=[query], n_results=n_results)
    if not results or not results["documents"]:
        return ""
    return "\n\n".join(results["documents"][0])

# ======== Chroma Builder ========
def build_chroma_from_uploads(uploaded_files, persist_dir):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name="HW5Collection", embedding_function=embed_fn
    )

    if collection.count() > 0:
        return collection

    for f in uploaded_files:
        tmp_path = os.path.join(tempfile.gettempdir(), f.name)
        with open(tmp_path, "wb") as out:
            out.write(f.getbuffer())
        with open(tmp_path, "r", encoding="utf-8") as doc:
            soup = BeautifulSoup(doc, "html.parser")
            text = soup.get_text(" ", strip=True)

        # Split into 2 halves
        mid = len(text) // 2
        chunks = [text[:mid], text[mid:]]
        for i, chunk in enumerate(chunks):
            cid = f"{f.name}_part{i+1}"
            collection.add(
                documents=[chunk],
                ids=[cid],
                metadatas=[{"filename": f.name, "chunk": i+1}],
            )
    return collection

# ======== Main Streamlit App ========
def app():
    st.set_page_config(page_title="HW5 — Short-term Memory Chatbot", layout="wide")
    st.title("HW5 — Short-term Memory Chatbot")

    with st.sidebar:
        st.header("🔑 Keys")
        openai_key = st.text_input("OPENAI_API_KEY", type="password")
        uploaded_files = st.file_uploader("Upload HTML files", type="html", accept_multiple_files=True)

        if st.button("🗑️ Clear Chat"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("collection", None)
            st.rerun()

    # Build DB
    if uploaded_files and "collection" not in st.session_state:
        persist_dir = os.path.join(tempfile.gettempdir(), "hw5_chroma_store")
        st.session_state.collection = build_chroma_from_uploads(uploaded_files, persist_dir)
        st.success(f"Vector DB built with {len(uploaded_files)} docs")

    # Chatbot UI
    if "collection" in st.session_state:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Show past messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if query := st.chat_input("Ask me about the orgs/courses..."):
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"): st.markdown(query)

            # Vector search
            context = get_relevant_info(query, st.session_state.collection, n_results=3)
            system_prompt = f"Use only the context below to answer clearly:\n{context}\n\nQuestion: {query}"

            # Keep last 5 turns only
            history = st.session_state.chat_history[-5:]

            reply = ""
            if openai_key:
                try:
                    client = OpenAIClient(api_key=openai_key)
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=history + [{"role": "system", "content": system_prompt}]
                    )
                    reply = resp.choices[0].message.content
                except Exception as e:
                    reply = f"⚠️ Error: {e}"
            else:
                reply = "⚠️ Please add your OpenAI key."

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"): st.markdown(reply)
    else:
        st.info("👆 Upload HTMLs to build the vector DB first.")

# Runner
if __name__ == "__main__":
    app()
