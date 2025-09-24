# # HW4.py — Student Org Chatbot with RAG + Benchmarking
# import os, sys, time, tempfile
# import streamlit as st
# import pandas as pd
# from openai import OpenAI as OpenAIClient
# import google.generativeai as genai
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
# import chromadb
# from chromadb.utils import embedding_functions
# from bs4 import BeautifulSoup
# import tiktoken

# # --- Fix for ChromaDB sqlite on Streamlit Cloud ---
# try:
#     __import__("pysqlite3")
#     sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# except Exception:
#     pass

# # ================= Helpers =================
# def normalize(val, vmin, vmax, invert=False):
#     if vmax == vmin:
#         return 1.0
#     x = (val - vmin) / (vmax - vmin)
#     return 1 - x if invert else x

# def count_tokens(text: str, model="gpt-4o-mini"):
#     try:
#         enc = tiktoken.encoding_for_model(model)
#     except Exception:
#         enc = tiktoken.get_encoding("cl100k_base")
#     return len(enc.encode(text))

# def truncate_by_tokens(history, model, max_tokens):
#     truncated, total = [], 0
#     for msg in reversed(history):
#         toks = count_tokens(msg["content"], model)
#         if total + toks <= max_tokens:
#             truncated.insert(0, msg)
#             total += toks
#         else:
#             break
#     return truncated

# # ================= Local LLaMA setup =================
# LOCAL_LLAMA_ID = "meta-llama/Llama-3.1-1B-Instruct"

# def _bnb_config(cuda_ok: bool):
#     return BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.bfloat16 if cuda_ok else torch.float32,
#     )

# @st.cache_resource(show_spinner=False)
# def load_llama_local_pipeline(model_id: str):
#     cuda_ok = torch.cuda.is_available()
#     tok = AutoTokenizer.from_pretrained(model_id)
#     mdl = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         torch_dtype=torch.bfloat16 if cuda_ok else torch.float32,
#         quantization_config=_bnb_config(cuda_ok),
#         device_map="auto",
#     )
#     if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
#         mdl.config.pad_token_id = tok.eos_token_id
#     pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
#     return tok, pipe

# def run_llama_local(prompt: str, max_new_tokens=256):
#     tok, pipe = load_llama_local_pipeline(LOCAL_LLAMA_ID)
#     msg = [{"role": "user", "content": prompt}]
#     input_text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
#     out = pipe(input_text, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
#     return out[0]["generated_text"][len(input_text):].strip()

# # ================= Vector DB (HTML Student Org Docs) =================
# def build_chroma_from_uploads(uploaded_files, persist_dir):
#     """Build ChromaDB from uploaded HTML files.
#     Chunking strategy: split each doc into 2 halves (simple + balanced).
#     """
#     os.makedirs(persist_dir, exist_ok=True)
#     client = chromadb.PersistentClient(path=persist_dir)

#     embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )

#     collection = client.get_or_create_collection(
#         name="HW4Collection", embedding_function=embed_fn
#     )

#     if collection.count() > 0:
#         return collection

#     for f in uploaded_files:
#         tmp_path = os.path.join(tempfile.gettempdir(), f.name)
#         with open(tmp_path, "wb") as out:
#             out.write(f.getbuffer())

#         with open(tmp_path, "r", encoding="utf-8") as doc:
#             soup = BeautifulSoup(doc, "html.parser")
#             text = soup.get_text(" ", strip=True)

#         # ✅ Chunking by halves to keep context manageable
#         mid = len(text) // 2
#         chunks = [text[:mid], text[mid:]]
#         for i, chunk in enumerate(chunks):
#             cid = f"{f.name}_part{i+1}"
#             collection.add(
#                 documents=[chunk],
#                 ids=[cid],
#                 metadatas=[{"filename": f.name, "chunk": i+1}],
#             )
#     return collection

# # ================= Main Streamlit App =================
# def app():
#     st.set_page_config(page_title="HW4 — iSchool Org Chatbot", layout="wide")
#     st.title("HW4 — iSchool Student Org Chatbot (RAG + Benchmark)")

#     # Sidebar
#     with st.sidebar:
#         st.header("🔑 API Keys")
#         openai_key = st.text_input("OPENAI_API_KEY", type="password")
#         gemini_key = st.text_input("GEMINI_API_KEY", type="password")

#         st.header("📂 Upload HTMLs")
#         uploaded_files = st.file_uploader(
#             "Drag & drop Student Org HTML files", type="html", accept_multiple_files=True
#         )

#         st.header("⚙️ Chatbot Settings")
#         backend = st.radio("Backend", ["OpenAI (API)", "Gemini (API)", "Local LLaMA"], index=0)

#         if backend == "OpenAI (API)":
#             openai_model = st.selectbox("OpenAI model", ["gpt-5", "gpt-4.1", "gpt-4o-mini"], index=0)
#         elif backend == "Gemini (API)":
#             gemini_model = st.selectbox("Gemini model", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
#         else:
#             max_tokens_llama = st.slider("Max new tokens (local)", 128, 768, 256, 32)

#         memory_type = st.selectbox("Memory type", ["Buffer of 5", "Conversation Summary", "Token Buffer (2000)"])

#         if st.button("🗑️ Clear Chat"):
#             st.session_state.pop("chat_history", None)
#             st.session_state.pop("summary", None)
#             st.session_state.pop("collection", None)
#             st.rerun()

#         st.header("📊 Benchmark Weights")
#         w_quality = st.slider("Quality", 0.0, 1.0, 0.5, 0.05)
#         w_speed   = st.slider("Speed",   0.0, 1.0, 0.3, 0.05)
#         w_cost    = st.slider("Cost",    0.0, 1.0, 0.2, 0.05)
#         s = w_quality + w_speed + w_cost
#         if s == 0:
#             w_quality, w_speed, w_cost = 0.5, 0.3, 0.2
#         else:
#             w_quality, w_speed, w_cost = (w_quality/s, w_speed/s, w_cost/s)
#         st.caption(f"Normalized → Q={w_quality:.2f}, S={w_speed:.2f}, C={w_cost:.2f}")

#     # ================= Build DB =================
#     if uploaded_files and "collection" not in st.session_state:
#         persist_dir = os.path.join(tempfile.gettempdir(), "hw4_chroma_store")
#         st.session_state.collection = build_chroma_from_uploads(uploaded_files, persist_dir)
#         st.success(f"Vector DB built with {len(uploaded_files)} docs")

#     # ================= Chatbot Section =================
#     if "collection" in st.session_state:
#         st.header("Chatbot")
#         if "chat_history" not in st.session_state:
#             st.session_state.chat_history = []
#         if "summary" not in st.session_state:
#             st.session_state.summary = ""

#         for msg in st.session_state.chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         if prompt := st.chat_input("Ask me about iSchool student orgs..."):
#             st.session_state.chat_history.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             results = st.session_state.collection.query(query_texts=[prompt], n_results=3)
#             context_docs = "\n\n".join(d for d in results["documents"][0]) if results else ""
#             rag_note = f"(Using RAG from {len(results['documents'][0])} docs)" if results else "(No RAG context)"

#             if memory_type == "Buffer of 5":
#                 history = st.session_state.chat_history[-5:]
#             elif memory_type == "Conversation Summary":
#                 history = [{"role": "system", "content": f"Summary: {st.session_state.summary}"}] + st.session_state.chat_history[-2:]
#             else:
#                 history = truncate_by_tokens(st.session_state.chat_history, "gpt-4o-mini", 2000)

#             rag_augmented = f"QUESTION: {prompt}\n\nCONTEXT:\n{context_docs}\n\nAnswer clearly. {rag_note}"

#             reply = ""
#             try:
#                 if backend == "OpenAI (API)":
#                     if openai_key:
#                         client = OpenAIClient(api_key=openai_key)
#                         resp = client.chat.completions.create(
#                             model=openai_model,
#                             messages=history + [{"role": "system", "content": rag_augmented}],
#                         )
#                         reply = resp.choices[0].message.content
#                     else:
#                         reply = "⚠️ Please add your OpenAI key."
#                 elif backend == "Gemini (API)":
#                     if gemini_key:
#                         genai.configure(api_key=gemini_key)
#                         model = genai.GenerativeModel(gemini_model)
#                         resp = model.generate_content(rag_augmented)
#                         reply = resp.text or "⚠️ No response."
#                     else:
#                         reply = "⚠️ Please add your Gemini key."
#                 else:
#                     reply = run_llama_local(rag_augmented, max_new_tokens=max_tokens_llama)
#             except Exception as e:
#                 reply = f"⚠️ Error: {e}"

#             st.session_state.chat_history.append({"role": "assistant", "content": reply})
#             with st.chat_message("assistant"):
#                 st.markdown(reply)

#             if memory_type == "Conversation Summary":
#                 st.session_state.summary += f"\nUser: {prompt}\nAssistant: {reply}\n"

#     # ================= Benchmark Section =================
#     if "collection" in st.session_state:
#         st.header("Benchmark Evaluation")
#         if st.button("Run Benchmark Tests"):
#             questions = [
#                 "What student organizations are available at the iSchool?",
#                 "How can I join the Information Security Club?",
#                 "What events are organized by Women in Technology?",
#                 "Who leads the Data Science Club?",
#                 "How do I start a new organization at the iSchool?"
#             ]
#             vendors = [
#                 ("OpenAI", "gpt-5", openai_key),
#                 ("Gemini", "gemini-2.5-pro", gemini_key),
#                 ("Local", LOCAL_LLAMA_ID, None),
#             ]
#             results_list = []
#             for vendor, model, key in vendors:
#                 for q in questions:
#                     docs = st.session_state.collection.query(query_texts=[q], n_results=3)
#                     ctx = "\n\n".join(d for d in docs["documents"][0]) if docs else ""
#                     prompt = f"You are a tutor for 10-year-olds.\n\nQUESTION: {q}\n\nCONTEXT:\n{ctx}"

#                     t0 = time.perf_counter()
#                     try:
#                         if vendor == "OpenAI" and key:
#                             client = OpenAIClient(api_key=key)
#                             resp = client.responses.create(model=model, input=prompt)
#                             ans = resp.output_text
#                         elif vendor == "Gemini" and key:
#                             genai.configure(api_key=key)
#                             ans = genai.GenerativeModel(model).generate_content(prompt).text or ""
#                         else:
#                             ans = run_llama_local(prompt, max_new_tokens=256)
#                     except Exception as e:
#                         ans = f"⚠️ Error: {e}"
#                     latency = time.perf_counter() - t0
#                     quality = min(1.0, len(ans.split()) / 250.0)
#                     pricing = {"OpenAI": 0.002, "Gemini": 0.0015, "Local": 0.0}
#                     cost = pricing[vendor] * len(ans.split())

#                     results_list.append({
#                         "Vendor": vendor, "Model": model, "Question": q,
#                         "Answer": ans[:250] + "...", "Latency": round(latency, 2),
#                         "Quality": round(quality, 2), "Cost": round(cost, 4)
#                     })

#             df = pd.DataFrame(results_list)
#             for i, row in df.iterrows():
#                 sq = row["Quality"]
#                 ss = normalize(row["Latency"], df["Latency"].min(), df["Latency"].max(), invert=True)
#                 sc = normalize(row["Cost"], df["Cost"].min(), df["Cost"].max(), invert=True)
#                 df.loc[i, "Score"] = round(sq*w_quality + ss*w_speed + sc*w_cost, 2)

#             st.subheader("Benchmark Results")
#             st.dataframe(df)

#             csv = df.to_csv(index=False).encode("utf-8")
#             st.download_button("📥 Download Results as CSV", csv, "benchmark_results.csv", "text/csv")

#     else:
#         st.info("👆 Please upload HTML files first to build the vector DB.")

# # Allow running standalone
# if __name__ == "__main__":
#     app()

# HW4.py — Student Org Chatbot with RAG + Benchmarking
import os, sys, time, tempfile
import streamlit as st
import pandas as pd
from openai import OpenAI as OpenAIClient
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import chromadb
from chromadb.utils import embedding_functions
from bs4 import BeautifulSoup
import tiktoken
from huggingface_hub import login as hf_login

# --- Fix for ChromaDB sqlite on Streamlit Cloud ---
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# ================= Helpers =================
def normalize(val, vmin, vmax, invert=False):
    if vmax == vmin:
        return 1.0
    x = (val - vmin) / (vmax - vmin)
    return 1 - x if invert else x

def count_tokens(text: str, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def truncate_by_tokens(history, model, max_tokens):
    truncated, total = [], 0
    for msg in reversed(history):
        toks = count_tokens(msg["content"], model)
        if total + toks <= max_tokens:
            truncated.insert(0, msg)
            total += toks
        else:
            break
    return truncated

# ================= Local LLaMA setup =================
LOCAL_LLAMA_ID = "meta-llama/Llama-3.2-1B-Instruct"

def _bnb_config(cuda_ok: bool):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cuda_ok else torch.float32,
    )

@st.cache_resource(show_spinner=False)
def load_llama_local_pipeline(model_id: str, hf_token: str | None = None):
    if not hf_token:
        raise RuntimeError("❌ HF_TOKEN is required to load local LLaMA models.")
    hf_login(token=hf_token)

    cuda_ok = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if cuda_ok else torch.float32,
        quantization_config=_bnb_config(cuda_ok),
        device_map="auto",
        use_auth_token=hf_token,
    )
    if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
        mdl.config.pad_token_id = tok.eos_token_id
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
    return tok, pipe

def run_llama_local(prompt: str, hf_token: str | None = None, max_new_tokens=256):
    tok, pipe = load_llama_local_pipeline(LOCAL_LLAMA_ID, hf_token)
    msg = [{"role": "user", "content": prompt}]
    input_text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    out = pipe(input_text, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    return out[0]["generated_text"][len(input_text):].strip()

# ================= Vector DB (HTML Student Org Docs) =================
def build_chroma_from_uploads(uploaded_files, persist_dir):
    """Build ChromaDB from uploaded HTML files.
    Chunking strategy: split each doc into 2 halves (simple + balanced).
    """
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name="HW4Collection", embedding_function=embed_fn
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

        # ✅ Chunking by halves to keep context manageable
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

# ================= Main Streamlit App =================
def app():
    st.set_page_config(page_title="HW4 — iSchool Org Chatbot", layout="wide")
    st.title("HW4 — iSchool Student Org Chatbot (RAG + Benchmark)")

    # Sidebar
    with st.sidebar:
        st.header("🔑 API Keys")
        openai_key = st.text_input("OPENAI_API_KEY", type="password")
        gemini_key = st.text_input("GEMINI_API_KEY", type="password")
        hf_token = st.text_input("HF_TOKEN (Hugging Face Hub)", type="password")

        st.header("📂 Upload HTMLs")
        uploaded_files = st.file_uploader(
            "Drag & drop Student Org HTML files", type="html", accept_multiple_files=True
        )

        st.header("⚙️ Chatbot Settings")
        backend = st.radio("Backend", ["OpenAI (API)", "Gemini (API)", "Local LLaMA"], index=0)

        if backend == "OpenAI (API)":
            openai_model = st.selectbox("OpenAI model", ["gpt-5", "gpt-4.1", "gpt-4o-mini"], index=0)
        elif backend == "Gemini (API)":
            gemini_model = st.selectbox("Gemini model", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
        else:
            max_tokens_llama = st.slider("Max new tokens (local)", 128, 768, 256, 32)

        memory_type = st.selectbox("Memory type", ["Buffer of 5", "Conversation Summary", "Token Buffer (2000)"])

        if st.button("🗑️ Clear Chat"):
            st.session_state.pop("chat_history", None)
            st.session_state.pop("summary", None)
            st.session_state.pop("collection", None)
            st.rerun()

        st.header("📊 Benchmark Weights")
        w_quality = st.slider("Quality", 0.0, 1.0, 0.5, 0.05)
        w_speed   = st.slider("Speed", 0.0, 1.0, 0.3, 0.05)
        w_cost    = st.slider("Cost", 0.0, 1.0, 0.2, 0.05)
        s = w_quality + w_speed + w_cost
        if s == 0:
            w_quality, w_speed, w_cost = 0.5, 0.3, 0.2
        else:
            w_quality, w_speed, w_cost = (w_quality/s, w_speed/s, w_cost/s)
        st.caption(f"Normalized → Q={w_quality:.2f}, S={w_speed:.2f}, C={w_cost:.2f}")

    # ================= Build DB =================
    if uploaded_files and "collection" not in st.session_state:
        persist_dir = os.path.join(tempfile.gettempdir(), "hw4_chroma_store")
        st.session_state.collection = build_chroma_from_uploads(uploaded_files, persist_dir)
        st.success(f"Vector DB built with {len(uploaded_files)} docs")

    # ================= Chatbot Section =================
    if "collection" in st.session_state:
        st.header("Chatbot")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "summary" not in st.session_state:
            st.session_state.summary = ""

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask me about iSchool student orgs..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            results = st.session_state.collection.query(query_texts=[prompt], n_results=3)
            context_docs = "\n\n".join(d for d in results["documents"][0]) if results else ""
            rag_note = f"(Using RAG from {len(results['documents'][0])} docs)" if results else "(No RAG context)"

            if memory_type == "Buffer of 5":
                history = st.session_state.chat_history[-5:]
            elif memory_type == "Conversation Summary":
                history = [{"role": "system", "content": f"Summary: {st.session_state.summary}"}] + st.session_state.chat_history[-2:]
            else:
                history = truncate_by_tokens(st.session_state.chat_history, "gpt-4o-mini", 2000)

            rag_augmented = f"QUESTION: {prompt}\n\nCONTEXT:\n{context_docs}\n\nAnswer clearly. {rag_note}"

            reply = ""
            try:
                if backend == "OpenAI (API)":
                    if openai_key:
                        client = OpenAIClient(api_key=openai_key)
                        resp = client.chat.completions.create(
                            model=openai_model,
                            messages=history + [{"role": "system", "content": rag_augmented}],
                        )
                        reply = resp.choices[0].message.content
                    else:
                        reply = "⚠️ Please add your OpenAI key."
                elif backend == "Gemini (API)":
                    if gemini_key:
                        genai.configure(api_key=gemini_key)
                        model = genai.GenerativeModel(gemini_model)
                        resp = model.generate_content(rag_augmented)
                        reply = resp.text or "⚠️ No response."
                    else:
                        reply = "⚠️ Please add your Gemini key."
                else:
                    if not hf_token:
                        reply = "⚠️ HF_TOKEN required for Local LLaMA."
                    else:
                        reply = run_llama_local(rag_augmented, hf_token=hf_token, max_new_tokens=max_tokens_llama)
            except Exception as e:
                reply = f"⚠️ Error: {e}"

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)

            if memory_type == "Conversation Summary":
                st.session_state.summary += f"\nUser: {prompt}\nAssistant: {reply}\n"

    # ================= Benchmark Section =================
    if "collection" in st.session_state:
        st.header("Benchmark Evaluation")
        if st.button("Run Benchmark Tests"):
            questions = [
                "What student organizations are available at the iSchool?",
                "How can I join the Information Security Club?",
                "What events are organized by Women in Technology?",
                "Who leads the Data Science Club?",
                "How do I start a new organization at the iSchool?"
            ]
            vendors = [
                ("OpenAI", "gpt-5", openai_key),
                ("Gemini", "gemini-2.5-pro", gemini_key),
                ("Local", LOCAL_LLAMA_ID, hf_token),
            ]
            results_list = []
            for vendor, model, key in vendors:
                for q in questions:
                    docs = st.session_state.collection.query(query_texts=[q], n_results=3)
                    ctx = "\n\n".join(d for d in docs["documents"][0]) if docs else ""
                    prompt = f"You are a tutor for 10-year-olds.\n\nQUESTION: {q}\n\nCONTEXT:\n{ctx}"

                    t0 = time.perf_counter()
                    try:
                        if vendor == "OpenAI" and key:
                            client = OpenAIClient(api_key=key)
                            resp = client.responses.create(model=model, input=prompt)
                            ans = resp.output_text
                        elif vendor == "Gemini" and key:
                            genai.configure(api_key=key)
                            ans = genai.GenerativeModel(model).generate_content(prompt).text or ""
                        else:
                            if not hf_token:
                                ans = "⚠️ HF_TOKEN required for Local LLaMA."
                            else:
                                ans = run_llama_local(prompt, hf_token=hf_token, max_new_tokens=256)
                    except Exception as e:
                        ans = f"⚠️ Error: {e}"
                    latency = time.perf_counter() - t0
                    quality = min(1.0, len(ans.split()) / 250.0)
                    pricing = {"OpenAI": 0.002, "Gemini": 0.0015, "Local": 0.0}
                    cost = pricing[vendor] * len(ans.split())

                    results_list.append({
                        "Vendor": vendor, "Model": model, "Question": q,
                        "Answer": ans[:250] + "...", "Latency": round(latency, 2),
                        "Quality": round(quality, 2), "Cost": round(cost, 4)
                    })

            df = pd.DataFrame(results_list)
            for i, row in df.iterrows():
                sq = row["Quality"]
                ss = normalize(row["Latency"], df["Latency"].min(), df["Latency"].max(), invert=True)
                sc = normalize(row["Cost"], df["Cost"].min(), df["Cost"].max(), invert=True)
                df.loc[i, "Score"] = round(sq*w_quality + ss*w_speed + sc*w_cost, 2)

            st.subheader("Benchmark Results")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results as CSV", csv, "benchmark_results.csv", "text/csv")

    else:
        st.info("👆 Please upload HTML files first to build the vector DB.")

# Allow running standalone
if __name__ == "__main__":
    app()
