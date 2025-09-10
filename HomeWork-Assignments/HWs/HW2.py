# # HWs/HW2.py
# import os
# import time
# import requests
# import streamlit as st
# from bs4 import BeautifulSoup
# from openai import OpenAI as OpenAIClient
# import google.generativeai as genai

# # Optional: Cohere (second non-OpenAI API)
# try:
#     import cohere
# except Exception:
#     cohere = None

# # ---------- URL Reader ----------
# def read_url_content(url: str) -> str | None:
#     try:
#         resp = requests.get(url, timeout=25)
#         resp.raise_for_status()
#         soup = BeautifulSoup(resp.content, "html.parser")
#         return soup.get_text(separator="\n")
#     except requests.RequestException as e:
#         st.error(f"Error reading {url}: {e}")
#         return None

# # ---------- Prompt Builder ----------
# def build_prompt(page_text: str, summary_style: str, out_lang: str) -> str:
#     return (
#         "You are a careful summarizer. Work only with the provided page text.\n"
#         "If content seems missing, say so briefly.\n\n"
#         f"Required output language: {out_lang}\n"
#         f"Summary style: {summary_style}\n\n"
#         "PAGE TEXT:\n"
#         f"--- START ---\n{page_text}\n--- END ---\n\n"
#         "Now produce the summary."
#     )

# # ---------- Scoring Helpers ----------
# def normalize(val, vmin, vmax, invert=False):
#     if vmax == vmin:
#         return 1.0
#     x = (val - vmin) / (vmax - vmin)
#     return 1 - x if invert else x

# PRICING = {
#     # $/1M tokens (illustrative; adjust to your plan)
#     "gpt-5":         {"in": 1.25, "out": 10.00},
#     "gpt-5-nano":    {"in": 0.05, "out": 0.40},
#     "gpt-4.1":       {"in": 5.00,  "out": 15.00},
#     "gemini-1.5-pro":   {"in": 1.25, "out": 5.00},
#     "gemini-1.5-flash": {"in": 0.35, "out": 0.70},
#     "command-r-plus":   {"in": 3.00,  "out": 15.00},
#     "command-r":        {"in": 0.50,  "out": 1.50},
# }

# QUALITY_PRIOR = {
#     # Prior subjective quality (1..4) — adjust after observing outputs
#     "gpt-5": 4, "gpt-4.1": 3, "gpt-5-nano": 2,
#     "gemini-1.5-pro": 3, "gemini-1.5-flash": 2,
#     "command-r-plus": 3, "command-r": 2,
# }

# # ---------- Backends ----------
# def run_openai(model_id: str, api_key: str, prompt: str):
#     client = OpenAIClient(api_key=api_key)
#     t0 = time.perf_counter()
#     resp = client.responses.create(model=model_id, input=prompt)
#     secs = time.perf_counter() - t0
#     usage = getattr(resp, "usage", None)
#     in_tok = getattr(usage, "input_tokens", None)
#     out_tok = getattr(usage, "output_tokens", None)
#     return resp.output_text, secs, in_tok, out_tok

# def run_gemini(model_id: str, api_key: str, prompt: str):
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel(model_id)
#     t0 = time.perf_counter()
#     resp = model.generate_content(prompt)
#     secs = time.perf_counter() - t0
#     return (resp.text or ""), secs, None, None

# def run_cohere(model_id: str, api_key: str, prompt: str):
#     if cohere is None:
#         raise RuntimeError("cohere SDK not installed. pip install cohere")
#     client = cohere.Client(api_key)
#     t0 = time.perf_counter()
#     # Use Chat endpoint for simplicity
#     resp = client.chat(model=model_id, message=prompt)
#     secs = time.perf_counter() - t0
#     text = getattr(resp, "text", None) or getattr(resp, "output_text", "") or ""
#     return text, secs, None, None

# # ---------- App ----------
# def app():
#     st.set_page_config(page_title="HW 2 – URL Summarizer", layout="centered")
#     st.title("HW 2 – URL Summarizer (Multi‑LLM + Weighted Scoring)")

#     # 4) URL at top
#     url = st.text_input("Enter a web page URL")

#     # Sidebar controls
#     with st.sidebar:
#         st.header("Controls")
#         # 5) Summary menu (like Lab2)
#         summary_style = st.selectbox(
#             "Summary type",
#             [
#                 "Concise TL;DR (3–5 sentences)",
#                 "Bulleted key takeaways (5 bullets)",
#                 "Headline and 3 bullets",
#                 "Short abstract (academic tone)",
#                 "Detailed outline (sections and subpoints)",
#             ],
#             index=0,
#         )
#         # 8) Output language (≥3 options)
#         out_lang = st.selectbox("Output language", ["English", "French", "Spanish", "German"], index=0)

#         # 10) LLM picker (+ legacy "use advanced model" style)
#         st.subheader("Single‑Run LLM")
#         use_advanced = st.checkbox("Use advanced model", value=True)
#         single_llm = st.selectbox(
#             "LLM API",
#             [
#                 "OpenAI",  # gpt‑5 / gpt‑5‑nano
#                 "Gemini",  # 1.5‑pro / 1.5‑flash
#                 "Cohere",  # command‑r‑plus / command‑r
#             ],
#             index=0,
#         )
#         # 11) API keys for ≥2 other APIs besides OpenAI
#         openai_key  = st.text_input("OPENAI_API_KEY",  type="password")
#         gemini_key  = st.text_input("GEMINI_API_KEY",  type="password")
#         cohere_key  = st.text_input("COHERE_API_KEY",  type="password")

#     # Buttons
#     run_single = st.button("Summarize (Selected LLM)")
#     run_eval   = st.button("Evaluate Models (Advanced & Cheaper)")

#     if not (run_single or run_eval):
#         return

#     if not url:
#         st.error("Please enter a URL.")
#         return

#     page_text = read_url_content(url)
#     if not page_text:
#         return

#     prompt = build_prompt(page_text, summary_style, out_lang)

#     # Helper for cost
#     def est_cost(model: str, in_tok, out_tok):
#         if in_tok is None or out_tok is None or model not in PRICING:
#             return 0.0
#         return (in_tok/1e6)*PRICING[model]["in"] + (out_tok/1e6)*PRICING[model]["out"]

#     # 12–13) Evaluation sets
#     ADVANCED = [
#         ("gpt-5", "openai"),
#         ("gemini-1.5-pro", "gemini"),
#         ("command-r-plus", "cohere"),
#     ]
#     CHEAPER = [
#         ("gpt-5-nano", "openai"),
#         ("gemini-1.5-flash", "gemini"),
#         ("command-r", "cohere"),
#     ]

#     # ---- Single run (respects use_advanced + single_llm) ----
#     if run_single:
#         try:
#             if single_llm == "OpenAI":
#                 model = "gpt-5" if use_advanced else "gpt-5-nano"
#                 if not openai_key:
#                     st.error("OPENAI_API_KEY required.")
#                     return
#                 text, secs, in_tok, out_tok = run_openai(model, openai_key, prompt)
#                 st.subheader(f"Summary — OpenAI {model}")
#                 st.write(text)
#                 st.caption(f"Latency: {secs:.2f}s • Tokens in/out: {in_tok or '—'}/{out_tok or '—'} • Cost≈ ${est_cost(model, in_tok, out_tok):.4f}")

#             elif single_llm == "Gemini":
#                 model = "gemini-1.5-pro" if use_advanced else "gemini-1.5-flash"
#                 if not gemini_key:
#                     st.error("GEMINI_API_KEY required.")
#                     return
#                 text, secs, in_tok, out_tok = run_gemini(model, gemini_key, prompt)
#                 st.subheader(f"Summary — Gemini {model}")
#                 st.write(text)
#                 st.caption(f"Latency: {secs:.2f}s • Tokens in/out: —/— • Cost≈ ${est_cost(model, in_tok, out_tok):.4f}")

#             else:  # Cohere
#                 model = "command-r-plus" if use_advanced else "command-r"
#                 if not cohere_key:
#                     st.error("COHERE_API_KEY required.")
#                     return
#                 text, secs, in_tok, out_tok = run_cohere(model, cohere_key, prompt)
#                 st.subheader(f"Summary — Cohere {model}")
#                 st.write(text)
#                 st.caption(f"Latency: {secs:.2f}s • Tokens in/out: —/— • Cost≈ ${est_cost(model, in_tok, out_tok):.4f}")
#         except Exception as e:
#             st.error(f"Single-run error: {e}")

#     # ---- Evaluation run (advanced & cheaper) ----
#     if run_eval:
#         sets = [("Advanced Models", ADVANCED), ("Cheaper Models", CHEAPER)]
#         for title, models in sets:
#             st.header(title)
#             results = {}

#             for model, vendor in models:
#                 try:
#                     with st.spinner(f"Querying {vendor}:{model}…"):
#                         if vendor == "openai":
#                             if not openai_key:
#                                 raise RuntimeError("OpenAI key missing")
#                             text, secs, in_tok, out_tok = run_openai(model, openai_key, prompt)
#                         elif vendor == "gemini":
#                             if not gemini_key:
#                                 raise RuntimeError("Gemini key missing")
#                             text, secs, in_tok, out_tok = run_gemini(model, gemini_key, prompt)
#                         else:
#                             if not cohere_key:
#                                 raise RuntimeError("Cohere key missing")
#                             text, secs, in_tok, out_tok = run_cohere(model, cohere_key, prompt)

#                     results[model] = {
#                         "answer": text,
#                         "latency": secs,
#                         "in": in_tok,
#                         "out": out_tok,
#                         "cost": est_cost(model, in_tok, out_tok),
#                         "q_prior": QUALITY_PRIOR.get(model, 2)
#                     }
#                 except Exception as e:
#                     results[model] = {"answer": f"<error: {e}>", "latency": float("inf"), "in": None, "out": None, "cost": float("inf"), "q_prior": 1}

#             # Scoring
#             latencies = [v["latency"] for v in results.values() if v["latency"] != float("inf")]
#             costs     = [v["cost"]    for v in results.values() if v["cost"]    != float("inf")]
#             lmin, lmax = (min(latencies), max(latencies)) if latencies else (0, 1)
#             cmin, cmax = (min(costs),     max(costs))     if costs     else (0, 1)

#             # Sidebar weights (reuse HW1 pattern)
#             with st.sidebar:
#                 st.subheader(f"Weights for {title}")
#                 wq = st.slider(f"Quality weight — {title}", 0.0, 1.0, 0.50, 0.05, key=f"wq_{title}")
#                 ws = st.slider(f"Speed weight — {title}",   0.0, 1.0, 0.30, 0.05, key=f"ws_{title}")
#                 wc = st.slider(f"Cost weight — {title}",    0.0, 1.0, 0.20, 0.05, key=f"wc_{title}")
#                 ssum = wq + ws + wc
#                 wq, ws, wc = (wq/ssum, ws/ssum, wc/ssum) if ssum else (0.5, 0.3, 0.2)

#             for m, r in results.items():
#                 r["score_quality"] = (r["q_prior"] - 1) / 3.0
#                 r["score_speed"]   = normalize(r["latency"], lmin, lmax, invert=True)
#                 r["score_cost"]    = normalize(r["cost"],    cmin, cmax, invert=True)
#                 r["composite"]     = r["score_quality"]*wq + r["score_speed"]*ws + r["score_cost"]*wc
#                 r["contrib_quality"] = r["score_quality"]*wq
#                 r["contrib_speed"]   = r["score_speed"]*ws
#                 r["contrib_cost"]    = r["score_cost"]*wc

#             # Tabs with answers
#             tabs = st.tabs(list(results.keys()))
#             for i, m in enumerate(results.keys()):
#                 with tabs[i]:
#                     st.subheader(m)
#                     st.write(results[m]["answer"])
#                     st.caption(
#                         f"Latency: {results[m]['latency']:.2f}s • "
#                         f"In/Out tokens: {results[m]['in'] or '—'}/{results[m]['out'] or '—'} • "
#                         f"Estimated cost: ${results[m]['cost']:.4f}"
#                     )

#             # Table + best
#             st.subheader(f"Weighted Criteria & Ranking — {title}")
#             rows = []
#             for m, r in results.items():
#                 rows.append({
#                     "Model": m,
#                     "Quality (0–1)": f"{r['score_quality']:.2f}",
#                     "Speed (0–1)":   f"{r['score_speed']:.2f}",
#                     "Cost (0–1)":    f"{r['score_cost']:.2f}",
#                     "Composite":     f"{r['composite']:.2f}",
#                 })
#             st.table(rows)

#             best = max(results.keys(), key=lambda k: results[k]["composite"])
#             st.success(f"Best Overall (by your weights): {best}")

#             # 12b / 13a-b Explanations
#             br = results[best]
#             contribs = {"quality": br["contrib_quality"], "speed": br["contrib_speed"], "cost": br["contrib_cost"]}
#             main_driver = max(contribs, key=contribs.get)
#             st.markdown(
#                 f"**Why best?** {best} scores {br['composite']:.2f}. The biggest contribution is **{main_driver}** "
#                 f"({br[f'contrib_{main_driver}']:.2f}). Adjust weights to test different priorities."
#             )

#             # Criterion winners (contrast)
#             best_quality = max(results.keys(), key=lambda k: results[k]["score_quality"])
#             best_speed   = max(results.keys(), key=lambda k: results[k]["score_speed"])
#             best_cost    = max(results.keys(), key=lambda k: results[k]["score_cost"])
#             st.caption(
#                 f"Leaders — quality: {best_quality}, speed: {best_speed}, cost: {best_cost}."
#             )






# HWs/HW2.py — URL Summarizer with Auto-Scoring & Manual Modes
import os
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Providers
from openai import OpenAI as OpenAIClient
import google.generativeai as genai

# Local HF inference for LLaMA 3.2 1B
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

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
LOCAL_LLAMA_ID = "meta-llama/Llama-3.2-1B-Instruct"  # HF repo

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
def load_llama_local_pipeline(model_id: str, hf_token: str):
    cuda_ok = torch.cuda.is_available()
    attn_impl = "sdpa"

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=False, token=hf_token
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
            token=hf_token,
        )
    except Exception:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation=attn_impl,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            token=hf_token,
        )

    if mdl.config.pad_token_id is None and tok.eos_token_id is not None:
        mdl.config.pad_token_id = tok.eos_token_id

    pipe = pipeline(task="text-generation", model=mdl, tokenizer=tok)
    return tok, pipe

def run_llama_local(prompt: str, hf_token: str, max_new_tokens: int = 256):
    tok, pipe = load_llama_local_pipeline(LOCAL_LLAMA_ID, hf_token)
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

# ============================== Scoring ==============================
def normalize(val, vmin, vmax, invert=False):
    if vmax == vmin:
        return 1.0
    x = (val - vmin) / (vmax - vmin)
    return 1 - x if invert else x

# $ per 1M tokens (illustrative)
PRICING = {
    "gpt-5":              {"in": 1.25, "out": 10.00},
    "gemini-2.5-pro":     {"in": 1.25, "out": 5.00},
    "gemini-2.5-flash":   {"in": 0.35, "out": 0.70},
    "llama-3.2-1b":       {"in": 0.00, "out": 0.00},  # local
}
QUALITY_PRIOR = {
    "gpt-5": 4,
    "gemini-2.5-pro": 4,
    "gemini-2.5-flash": 3,
    "llama-3.2-1b": 2,
}

def estimate_cost(model: str, in_tok, out_tok):
    if in_tok is None or out_tok is None or model not in PRICING:
        return 0.0
    return (in_tok/1e6)*PRICING[model]["in"] + (out_tok/1e6)*PRICING[model]["out"]

# ================================ UI =================================
def app():
    st.title("HW 2 – URL Summarizer (Auto-Scoring & Manual)")

    # URL input at top
    url = st.text_input("Enter a web page URL")

    # Sidebar: shared controls
    with st.sidebar:
        st.header("Summary Controls")
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
        out_lang = st.selectbox("Output language", ["English", "French", "Spanish", "German"], index=0)

        st.markdown("---")
        st.header("API Keys / Local Options")
        openai_key = st.text_input("OPENAI_API_KEY", type="password")
        gemini_key = st.text_input("GEMINI_API_KEY", type="password")
        include_local = st.checkbox("Include Local LLaMA 3.2 1B", value=False)
        hf_token = ""
        max_tokens = 256
        if include_local:
            hf_token = st.text_input("HF_HUB_TOKEN", type="password", value=os.getenv("HF_HUB_TOKEN") or "")
            max_tokens = st.slider("Max new tokens (local)", 96, 768, 256, 32)

        st.markdown("---")
        st.header("Auto-Scoring Weights (HW1 style)")
        w_quality = st.slider("Quality weight", 0.0, 1.0, 0.50, 0.05, key="as_wq")
        w_speed   = st.slider("Speed weight",   0.0, 1.0, 0.30, 0.05, key="as_ws")
        w_cost    = st.slider("Cost weight",    0.0, 1.0, 0.20, 0.05, key="as_wc")
        s = w_quality + w_speed + w_cost
        w_quality, w_speed, w_cost = (w_quality/s, w_speed/s, w_cost/s) if s else (0.5, 0.3, 0.2)
        st.caption(f"Normalized: quality={w_quality:.2f}, speed={w_speed:.2f}, cost={w_cost:.2f}")

        st.markdown("---")
        st.header("Manual Mode Options")
        backend = st.radio(
            "Manual: Backend",
            ["OpenAI (API)", "Gemini (API)", "Local LLaMA 3.2 1B"],
            index=0,
            key="manual_backend"
        )
        openai_model = gemini_model = None
        if backend == "OpenAI (API)":
            # Keep manual OpenAI simple per your request
            openai_model = st.selectbox(
                "OpenAI model",
                ["gpt-5"],
                index=0,
                key="manual_openai_model"
            )
        elif backend == "Gemini (API)":
            # Updated to 2.5 models
            gemini_model = st.selectbox(
                "Gemini model",
                ["gemini-2.5-pro", "gemini-2.5-flash"],
                index=0,
                key="manual_gemini_model"
            )
        # Local LLaMA uses hf_token/max_tokens above

    # Mode buttons
    colA, colB = st.columns(2)
    with colA:
        run_auto = st.button("Auto-Score (gpt-5 • gemini-2.5-pro • llama-3.2-1b)", type="primary")
    with colB:
        run_manual = st.button("Manual Summarize (selected backend)")

    if not (run_auto or run_manual):
        return
    if not url:
        st.error("Please enter a URL.")
        return

    page_text = read_url_content(url)
    if not page_text:
        return
    prompt = build_prompt(page_text, summary_style, out_lang)

    # ---------- Manual mode ----------
    if run_manual:
        try:
            if backend == "OpenAI (API)":
                if not openai_key:
                    st.error("Please provide your OPENAI_API_KEY in the sidebar."); return
                ans, secs, in_toks, out_toks = run_openai_summary(openai_model, openai_key, prompt)
                st.subheader(f"Summary — OpenAI {openai_model}")
                st.write(ans)
                st.caption(f"Latency: {secs:.2f}s | Tokens in/out: {in_toks or '—'}/{out_toks or '—'}")
            elif backend == "Gemini (API)":
                if not gemini_key:
                    st.error("Please provide your GEMINI_API_KEY in the sidebar."); return
                ans, secs, _, _ = run_gemini_summary(gemini_model, gemini_key, prompt)
                st.subheader(f"Summary — Gemini {gemini_model}")
                st.write(ans)
                st.caption(f"Latency: {secs:.2f}s")
            else:
                if not include_local or not hf_token:
                    st.error("Enable Local LLaMA and provide HF_HUB_TOKEN in the sidebar."); return
                ans, secs = run_llama_local(prompt, hf_token, max_new_tokens=max_tokens)
                st.subheader("Summary — Local LLaMA 3.2 1B Instruct")
                st.write(ans)
                st.caption(f"Latency: {secs:.2f}s")
                st.info("Local inference uses your machine resources.")
        except Exception as e:
            st.error(f"Manual mode error: {e}")
        return

    # ---------- Auto-Scoring mode ----------
    # EXACT set you asked for (only if available):
    model_list = []
    if openai_key:
        model_list.append("gpt-5")
    if gemini_key:
        model_list.append("gemini-2.5-pro")
    if include_local and hf_token:
        model_list.append("llama-3.2-1b")

    if not model_list:
        st.info("No models available to auto-score. Add an API key or enable Local LLaMA.")
        return

    results = {}
    for m in model_list:
        with st.spinner(f"Querying {m}…"):
            try:
                if m == "gpt-5":
                    text, secs, in_tok, out_tok = run_openai_summary(m, openai_key, prompt)
                elif m.startswith("gemini-"):
                    text, secs, in_tok, out_tok = run_gemini_summary(m, gemini_key, prompt)
                else:
                    text, secs = run_llama_local(prompt, hf_token, max_new_tokens=max_tokens)
                    in_tok = out_tok = None
                latency = secs
                cost = estimate_cost(m, in_tok, out_tok)
                results[m] = {
                    "answer": text, "latency": latency,
                    "tokens_in": in_tok, "tokens_out": out_tok, "cost": cost
                }
            except Exception as e:
                results[m] = {
                    "answer": f"<error: {e}>", "latency": float('inf'),
                    "tokens_in": None, "tokens_out": None, "cost": float('inf')
                }

    # Scoring (HW1 style)
    latencies = [v["latency"] for v in results.values() if v["latency"] != float("inf")]
    costs     = [v["cost"]    for v in results.values() if v["cost"]    != float("inf")]
    lmin, lmax = (min(latencies), max(latencies)) if latencies else (0, 1)
    cmin, cmax = (min(costs),     max(costs))     if costs     else (0, 1)

    for m in model_list:
        r = results[m]
        q_raw = QUALITY_PRIOR.get(m, 2)                     # 1..4
        r["score_quality"] = (q_raw - 1) / 3.0              # → 0..1
        r["score_speed"]   = normalize(r["latency"], lmin, lmax, invert=True)
        r["score_cost"]    = normalize(r["cost"],    cmin, cmax, invert=True)
        r["composite"]     = r["score_quality"]*w_quality + r["score_speed"]*w_speed + r["score_cost"]*w_cost
        r["contrib_quality"] = r["score_quality"]*w_quality
        r["contrib_speed"]   = r["score_speed"]*w_speed
        r["contrib_cost"]    = r["score_cost"]*w_cost

    # Answers in tabs
    tabs = st.tabs(model_list)
    for i, m in enumerate(model_list):
        with tabs[i]:
            st.subheader(m)
            st.write(results[m]["answer"])
            st.caption(
                f"Latency: {results[m]['latency']:.2f}s • "
                f"In/Out tokens: {results[m]['tokens_in'] or '—'}/{results[m]['tokens_out'] or '—'} • "
                f"Estimated cost: ${results[m]['cost']:.4f}"
            )

    # Weighted table & ranking
    st.header("Weighted Criteria & Ranking")
    st.table([
        {
            "Model": m,
            "Quality (0–1)": f"{results[m]['score_quality']:.2f}",
            "Speed (0–1)":   f"{results[m]['score_speed']:.2f}",
            "Cost (0–1)":    f"{results[m]['score_cost']:.2f}",
            "Composite":     f"{results[m]['composite']:.2f}",
        } for m in model_list
    ])

    best = max(model_list, key=lambda k: results[k]["composite"])
    ranked = sorted(model_list, key=lambda k: results[k]["composite"], reverse=True)
    st.success(f"Best Overall (by your weights): {best}")

    # Worded rationale
    br = results[best]
    contribs = {"quality": br["contrib_quality"], "speed": br["contrib_speed"], "cost": br["contrib_cost"]}
    main_driver = max(contribs, key=contribs.get)
    best_quality = max(model_list, key=lambda k: results[k]["score_quality"])
    best_speed   = max(model_list, key=lambda k: results[k]["score_speed"])
    best_cost    = max(model_list, key=lambda k: results[k]["score_cost"])

    lines = []
    lines.append(
        f"With your weights (quality {w_quality:.0%}, speed {w_speed:.0%}, cost {w_cost:.0%}), "
        f"{best} ranks first with a composite score of {br['composite']:.2f}."
    )
    lines.append(
        f"The largest contribution to the winning score comes from {main_driver} "
        f"({br[f'contrib_{main_driver}']:.2f} of the total)."
    )
    is_compromise = (best != best_quality) and (best != best_speed) and (best != best_cost)
    if is_compromise:
        lines.append(
            f"{best} is not the top model on any single criterion "
            f"(best quality: {best_quality}, best speed: {best_speed}, best cost: {best_cost}), "
            "but it offers the strongest overall balance given your weights."
        )
    if len(ranked) > 1:
        runner = ranked[1]
        diff = results[best]["composite"] - results[runner]["composite"]
        lines.append(
            f"The runner-up is {runner} with a composite of {results[runner]['composite']:.2f}, "
            f"{diff:.2f} points behind."
        )

    st.subheader("Summary")
    st.write(" ".join(lines))
