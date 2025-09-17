

# hw3_app.py — Streaming Chatbot with Memory + Benchmark
import os, time, requests
import streamlit as st
from bs4 import BeautifulSoup
from huggingface_hub import login as hf_login
from openai import OpenAI as OpenAIClient
import google.generativeai as genai
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import tiktoken

# --- Hugging Face login ---
HF_TOKEN = os.getenv("HF_HUB_TOKEN")
if HF_TOKEN:
    hf_login(token=HF_TOKEN)

# ================= URL Reader =================
def read_url_content(url: str) -> str | None:
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        return BeautifulSoup(r.content, "html.parser").get_text(separator="\n")
    except Exception as e:
        st.error(f"Error reading {url}: {e}")
        return None

# ================= Local LLaMA =================
LOCAL_LLAMA_ID = "meta-llama/Llama-3.2-1B-Instruct"

def _bnb_config(cuda_ok: bool):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if cuda_ok else torch.float32,
    )

@st.cache_resource(show_spinner=False)
def load_llama_local_pipeline(model_id: str):
    cuda_ok = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if cuda_ok else torch.float32,
        quantization_config=_bnb_config(cuda_ok),
        device_map="auto",
        use_auth_token=HF_TOKEN,
    )
    if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
        mdl.config.pad_token_id = tok.eos_token_id
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
    return tok, pipe

def run_llama_local(prompt: str, max_new_tokens=256):
    tok, pipe = load_llama_local_pipeline(LOCAL_LLAMA_ID)
    msg = [{"role": "user", "content": prompt}]
    input_text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    out = pipe(input_text, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    return out[0]["generated_text"][len(input_text):].strip()

# ================= Token Utils =================
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
            truncated.insert(0, msg); total += toks
        else:
            break
    return truncated

def normalize(val, vmin, vmax, invert=False):
    if vmax == vmin: return 1.0
    x = (val - vmin) / (vmax - vmin)
    return 1 - x if invert else x

# ================= Main App =================
def app():
    st.set_page_config(page_title="HW3 — Chatbot & Benchmark", layout="wide")
    st.title("HW3 — Streaming Chatbot with URLs & Memory")

    # Sidebar
    with st.sidebar:
        st.header("API Keys")
        openai_key = st.text_input("OPENAI_API_KEY", type="password")
        gemini_key = st.text_input("GEMINI_API_KEY", type="password")

        st.header("Chatbot Settings")
        url1 = st.text_input("Enter first URL")
        url2 = st.text_input("Enter second URL (optional)")
        backend = st.radio("Backend", ["OpenAI (API)", "Gemini (API)", "Local LLaMA"], index=0)

        if backend == "OpenAI (API)":
            openai_model = st.selectbox("OpenAI model", ["gpt-5", "gpt-4.1", "gpt-4o-mini"], index=0)
        elif backend == "Gemini (API)":
            gemini_model = st.selectbox("Gemini model", ["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
        else:
            max_tokens_llama = st.slider("Max new tokens (local)", 128, 768, 256, 32)

        memory_type = st.selectbox("Memory type", ["Buffer of 6", "Conversation Summary", "Token Buffer (2000)"])

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history, st.session_state.summary = [], ""
            st.rerun()

    # Init
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "summary" not in st.session_state: st.session_state.summary = ""

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about the URLs..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        # Prepare docs
        docs_text = "\n\n".join([read_url_content(u) or "" for u in [url1, url2] if u])

        # Memory
        if memory_type == "Buffer of 6":
            history = st.session_state.chat_history[-6:]
        elif memory_type == "Conversation Summary":
            history = [{"role": "system", "content": f"Summary so far: {st.session_state.summary}"}] + st.session_state.chat_history[-2:]
        else:
            history = truncate_by_tokens(st.session_state.chat_history, "gpt-4o-mini", 2000)

        messages = history + [{"role": "system", "content": f"Relevant docs:\n{docs_text}"}]

        # Run backend
        reply = ""
        if backend == "OpenAI (API)":
            if not openai_key: st.error("Missing OpenAI key"); return
            client = OpenAIClient(api_key=openai_key)
            stream = client.chat.completions.create(model=openai_model, messages=messages, stream=True)
            with st.chat_message("assistant"):
                ph = st.empty()
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    reply += delta; ph.markdown(reply)
        elif backend == "Gemini (API)":
            if not gemini_key: st.error("Missing Gemini key"); return
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(gemini_model)
            resp = model.generate_content(prompt + "\n\nDocs:\n" + docs_text, stream=True)
            with st.chat_message("assistant"):
                ph = st.empty()
                for chunk in resp:
                    delta = getattr(chunk, "text", "") or ""
                    reply += delta; ph.markdown(reply)
        else:
            reply = run_llama_local(prompt + "\n\nDocs:\n" + docs_text, max_new_tokens=max_tokens_llama)
            with st.chat_message("assistant"): st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        if memory_type == "Conversation Summary":
            st.session_state.summary += f"\nUser: {prompt}\nAssistant: {reply}\n"

    # ================= Benchmark =================
    st.header("Benchmark Evaluation")
    if st.button("Run Benchmark Test"):
        url1 = "https://www.howbaseballworks.com/TheBasics.htm"
        url2 = "https://www.pbs.org/kenburns/baseball/baseball-for-beginners"
        ctx1, ctx2 = read_url_content(url1) or "", read_url_content(url2) or ""
        both_ctx = ctx1 + "\n\n" + ctx2

        questions = [
            "Explain how an inning works in baseball.",
            "What is the role of the pitcher?",
            "How does a team score runs?",
        ]
        vendors = [
            ("OpenAI", "gpt-5", openai_key),
            ("Gemini", "gemini-2.5-pro", gemini_key),
            ("Local", LOCAL_LLAMA_ID, None),
        ]
        memory_types = [
            ("Buffer of 6", lambda hist, model: hist[-6:]),
            ("Conversation Summary", lambda hist, model: (
                [{"role": "system", "content": f"Summary so far: {''.join(m['content'] for m in hist)}"}] + hist[-2:]
            )),
            ("Token Buffer (2000)", lambda hist, model: truncate_by_tokens(hist, model, 2000)),
        ]

        results = []
        for vendor, model, key in vendors:
            for docs_label, ctx in [("Doc1 only", ctx1), ("Both docs", both_ctx)]:
                for mem_label, mem_func in memory_types:
                    hist, summary = [], ""
                    for q in questions:
                        hist.append({"role": "user", "content": q})
                        history = mem_func(hist, model)
                        prompt = f"You are a tutor for 10-year-olds.\n\nCONTEXT:\n{ctx[:3000]}\n\nQUESTION: {q}"

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
                                ans = run_llama_local(prompt)
                        except Exception as e:
                            ans = f"⚠️ Error: {e}"
                        secs = time.perf_counter() - t0

                        hist.append({"role": "assistant", "content": ans})
                        if mem_label == "Conversation Summary":
                            summary += f"\nUser: {q}\nAssistant: {ans}\n"

                        qscore = min(1.0, len(ans.split()) / 250.0)
                        results.append({
                            "Vendor": vendor, "Model": model,
                            "Docs": docs_label, "Memory": mem_label,
                            "Question": q,
                            "Answer": ans[:300] + "...",
                            "Latency": round(secs, 2),
                            "Quality": round(qscore, 2)
                        })

        st.subheader("Benchmark Results")
        st.dataframe(results)

# Run
if __name__ == "__main__":
    app()
