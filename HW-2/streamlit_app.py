import os
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ——— Hugging Face Hub login ———
from huggingface_hub import login as hf_login

# Log in to HF Hub using the environment variable
HF_TOKEN = os.getenv("HF_HUB_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set the HF_HUB_TOKEN environment variable before running.")
hf_login(token=HF_TOKEN)


# ----------------------------- Providers -----------------------------
# OpenAI API
from openai import OpenAI as OpenAIClient
# Google Gemini API
import google.generativeai as genai

# Local HF inference for LLaMA 3.2 1B
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)


# ============================ URL Reader =============================
def read_url_content(url: str) -> str | None:
    try:
        resp = requests.get(url, timeout=25)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        return soup.get_text(separator="\n")
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None


# =========================== Prompt Builder ==========================
def build_prompt(page_text: str, summary_style: str, out_lang: str) -> str:
    return (
        "You are a careful summarizer. Work only with the provided page text.\n"
        "If content seems missing, state that briefly.\n\n"
        f"Required output language: {out_lang}\n"
        f"Summary style: {summary_style}\n\n"
        "PAGE TEXT:\n"
        f"--- START ---\n{page_text}\n--- END ---\n\n"
        "Now produce the summary."
    )


# ======================== Local LLaMA Loader =========================
LOCAL_LLAMA_ID = "meta-llama/Llama-3.2-1B-Instruct"  # HF repo name

def _bnb_config(cuda_available: bool):
    compute_dtype = torch.bfloat16 if cuda_available else torch.float32
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

def _max_memory_map():
    return {"cuda:0": "6GiB", "cpu": "22GiB"}

try:
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

@st.cache_resource(show_spinner=False)
def load_llama_local_pipeline(model_id: str):
    """
    Load LLaMA 3.2 1B locally with 4-bit quant (if bitsandbytes available).
    Uses GPU if present; falls back to CPU FP32 otherwise.
    """
    cuda_ok = torch.cuda.is_available()
    attn_impl = "sdpa"

    tok = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=False,
        use_auth_token=HF_TOKEN
    )

    try:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if cuda_ok else torch.float32,
            attn_implementation=attn_impl,
            quantization_config=_bnb_config(cuda_ok),
            device_map="auto",
            max_memory=_max_memory_map(),
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            use_auth_token=HF_TOKEN
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            use_auth_token=HF_TOKEN
        )

    if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
        mdl.config.pad_token_id = tok.eos_token_id

    pipe = pipeline(task="text-generation", model=mdl, tokenizer=tok)
    return tok, pipe

def run_llama_local(prompt: str, max_new_tokens: int = 256):
    tok, pipe = load_llama_local_pipeline(LOCAL_LLAMA_ID)

    try:
        messages = [{"role": "user", "content": prompt}]
        input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        input_text = prompt

    t0 = time.perf_counter()
    out = pipe(
        input_text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tok.eos_token_id,
        num_return_sequences=1,
        pad_token_id=tok.eos_token_id,
    )
    secs = time.perf_counter() - t0

    generated = out[0]["generated_text"]
    answer = generated[len(input_text):].strip() if generated.startswith(input_text) else generated.strip()
    return answer, secs


# ============================== OpenAI ===============================
def run_openai_summary(model_id: str, api_key: str, prompt: str):
    client = OpenAIClient(api_key=api_key)
    t0 = time.perf_counter()
    resp = client.responses.create(model=model_id, input=prompt)
    secs = time.perf_counter() - t0
    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "input_tokens", None)
    out_tok = getattr(usage, "output_tokens", None)
    return resp.output_text, secs, in_tok, out_tok


# ============================== Gemini ===============================
def run_gemini_summary(model_id: str, api_key: str, prompt: str):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_id)
    t0 = time.perf_counter()
    resp = model.generate_content(prompt)
    secs = time.perf_counter() - t0
    text = resp.text or ""
    return text, secs, None, None


# ================================ UI =================================
st.set_page_config(page_title="HW 2 – URL Summarizer", layout="centered")
st.title("HW 2 – URL Summarizer (OpenAI API, Gemini API, Local LLaMA 3.2 1B)")

# URL input at top
url = st.text_input("Enter a web page URL")

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    summary_style = st.selectbox(
        "Summary type",
        [
            "Concise TL;DR (3–5 sentences)",
            "Bulleted key takeaways (5 bullets)",
            "Headline and 3 bullets",
            "Short abstract (academic tone)",
            "Detailed outline (sections and subpoints)",
        ],
        index=0,
    )

    out_lang = st.selectbox(
        "Output language",
        ["English", "French", "Spanish", "German"],
        index=0,
    )

    backend = st.radio(
        "Backend",
        ["OpenAI (API)", "Gemini (API)", "Local LLaMA 3.2 1B"],
        index=2,
    )

    if backend == "OpenAI (API)":
        openai_model = st.selectbox(
            "OpenAI model",
            ["gpt-5", "gpt-5-nano", "gpt-4.1"],
            index=0,
        )
        openai_key = st.text_input("OPENAI_API_KEY", type="password")

    elif backend == "Gemini (API)":
        gemini_model = st.selectbox(
            "Gemini model",
            ["gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
        )
        gemini_key = st.text_input("GEMINI_API_KEY", type="password")

    else:  # Local LLaMA 3.2 1B
        max_tokens = st.slider("Max new tokens (local)", 96, 768, 256, 32)

# Action button
run_btn = st.button("Summarize")

# ============================= Run App ================================
if run_btn:
    if not url:
        st.error("Please enter a URL.")
        st.stop()

    page_text = read_url_content(url)
    if not page_text:
        st.stop()

    prompt = build_prompt(page_text, summary_style, out_lang)

    if backend == "OpenAI (API)":
        if not openai_key:
            st.error("Please provide your OPENAI_API_KEY in the sidebar.")
            st.stop()
        try:
            ans, secs, in_toks, out_toks = run_openai_summary(openai_model, openai_key, prompt)
            st.subheader(f"Summary — OpenAI {openai_model}")
            st.write(ans)
            st.caption(f"Latency: {secs:.2f}s | Tokens in/out: {in_toks or '—'}/{out_toks or '—'}")
        except Exception as e:
            st.error(f"OpenAI error: {e}")

    elif backend == "Gemini (API)":
        if not gemini_key:
            st.error("Please provide your GEMINI_API_KEY in the sidebar.")
            st.stop()
        try:
            ans, secs, _, _ = run_gemini_summary(gemini_model, gemini_key, prompt)
            st.subheader(f"Summary — Gemini {gemini_model}")
            st.write(ans)
            st.caption(f"Latency: {secs:.2f}s")
        except Exception as e:
            st.error(f"Gemini error: {e}")

    else:  # Local LLaMA 3.2 1B
        try:
            ans, secs = run_llama_local(prompt, max_new_tokens=max_tokens)
            st.subheader("Summary — Local LLaMA 3.2 1B Instruct")
            st.write(ans)
            st.caption(f"Latency: {secs:.2f}s")
            st.info(
                "Local inference uses your machine resources. "
                "This setup supports 4-bit quantization with CPU offloading for low VRAM."
            )
        except Exception as e:
            st.error(
                f"Local LLaMA error: {e}\n"
                "Tips: ensure torch (CUDA if available), transformers, accelerate, and bitsandbytes are installed; "
                "accept the model license on Hugging Face; reduce 'Max new tokens' if needed."
            )


