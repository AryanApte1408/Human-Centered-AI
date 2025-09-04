import streamlit as st
import fitz  # PyMuPDF
import time, os, hashlib
from openai import OpenAI

# ─────────────────────────── Helpers ───────────────────────────
def read_pdf(file):
    file.seek(0)
    data = file.read()
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def normalize(val, vmin, vmax, invert=False):
    if vmax == vmin:
        return 1.0
    x = (val - vmin) / (vmax - vmin)
    return 1 - x if invert else x

def file_sha1(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except Exception:
        return "unknown"

# ───────────────────────── App Header ──────────────────────────
st.set_page_config(page_title="Document Buddy (Weighted)", layout="centered")
st.title("Document Buddy — Weighted Model Comparison")

# show which file is running (path + short hash) to avoid “wrong file” confusion
running_path = os.path.abspath(__file__) if "__file__" in globals() else "(interactive)"
st.caption(f"Running file: `{running_path}` | sha1: `{file_sha1(running_path)}`")

st.write("Upload a PDF/TXT, ask a question, and compare models via Quality, Speed, and Cost (weighted).")

# Sidebar: API key + Weights
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("---")
    st.subheader("Weights")
    w_quality = st.slider("Quality weight", 0.0, 1.0, 0.50, 0.05)
    w_speed   = st.slider("Speed weight",   0.0, 1.0, 0.30, 0.05)
    w_cost    = st.slider("Cost weight",    0.0, 1.0, 0.20, 0.05)
    # normalize if user makes them not sum to 1
    s = w_quality + w_speed + w_cost
    if s == 0:
        w_quality, w_speed, w_cost = 0.5, 0.3, 0.2
    else:
        w_quality, w_speed, w_cost = (w_quality/s, w_speed/s, w_cost/s)
    st.caption(f"Normalized: quality={w_quality:.2f}, speed={w_speed:.2f}, cost={w_cost:.2f}")

if not openai_api_key:
    st.info("Enter your OpenAI API key in the sidebar to continue.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ────────────────────── Upload & Question ──────────────────────
uploaded_file = st.file_uploader("Upload document (.txt or .pdf)", type=["txt", "pdf"])
if not uploaded_file:
    st.session_state.pop("document_text", None)

question = st.text_area("Your question", placeholder="E.g., Summarize this syllabus.", disabled=not uploaded_file)

# ────────────────────── Process & Compare ──────────────────────
if uploaded_file and question:
    if "document_text" not in st.session_state:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext == "txt":
            uploaded_file.seek(0)
            st.session_state.document_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif ext == "pdf":
            st.session_state.document_text = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type."); st.stop()

    doc = st.session_state.document_text
    prompt = f"DOCUMENT:\n{doc}\n\nQUESTION: {question}"

    models = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5", "gpt-5-nano"]
    pricing = {
        "gpt-3.5-turbo": {"in": 0.50, "out": 1.50},
        "gpt-4.1":       {"in": 5.00, "out": 15.00},
        "gpt-5":         {"in": 1.25, "out": 10.00},
        "gpt-5-nano":    {"in": 0.05, "out": 0.40},
    }

    results = {}
    for m in models:
        with st.spinner(f"Querying {m}…"):
            t0 = time.perf_counter()
            resp = client.responses.create(
                model=m,
                instructions=(
                    "You are a document-reading assistant. Answer ONLY from the document; "
                    "if not present, reply: Not found in document."
                ),
                input=prompt,
            )
            latency = time.perf_counter() - t0

        # usage fields exist on Responses API
        in_tok  = getattr(resp.usage, "input_tokens", None)
        out_tok = getattr(resp.usage, "output_tokens", None)
        cost = None
        if in_tok is not None and out_tok is not None and m in pricing:
            cost = (in_tok/1e6)*pricing[m]["in"] + (out_tok/1e6)*pricing[m]["out"]

        results[m] = {
            "answer": resp.output_text,
            "latency": latency,
            "tokens_in": in_tok,
            "tokens_out": out_tok,
            "cost": cost if cost is not None else 0.0,
        }

    # Manual quality (1–4) → adjust per your actual observations
    quality_raw = {
        "gpt-3.5-turbo": 1,
        "gpt-4.1":       3,
        "gpt-5":         4,
        "gpt-5-nano":    2,
    }

    latencies = [results[m]["latency"] for m in models]
    costs     = [results[m]["cost"]    for m in models]

    for m in models:
        r = results[m]
        r["score_quality"] = (quality_raw[m] - 1) / 3.0  # 1..4 → 0..1
        r["score_speed"]   = normalize(r["latency"], min(latencies), max(latencies), invert=True)
        r["score_cost"]    = normalize(r["cost"],     min(costs),     max(costs),     invert=True)
        r["composite"]     = (
            r["score_quality"] * w_quality +
            r["score_speed"]   * w_speed   +
            r["score_cost"]    * w_cost
        )
        # Also compute each criterion's contribution to composite for explanation
        r["contrib_quality"] = r["score_quality"] * w_quality
        r["contrib_speed"]   = r["score_speed"]   * w_speed
        r["contrib_cost"]    = r["score_cost"]    * w_cost

    # — Answers in tabs
    tabs = st.tabs(models)
    for i, m in enumerate(models):
        with tabs[i]:
            st.subheader(m)
            st.markdown("Answer")
            st.write(results[m]["answer"])
            st.caption(
                f"Latency: {results[m]['latency']:.2f}s • "
                f"In/Out tokens: {results[m]['tokens_in'] or '—'}/{results[m]['tokens_out'] or '—'} • "
                f"Estimated cost: ${results[m]['cost']:.4f}"
            )

    # — Weighted table
    st.header("Weighted Criteria & Ranking")
    table_rows = []
    for m in models:
        r = results[m]
        table_rows.append({
            "Model": m,
            "Quality (0–1)": f"{r['score_quality']:.2f}",
            "Speed (0–1)":   f"{r['score_speed']:.2f}",
            "Cost (0–1)":    f"{r['score_cost']:.2f}",
            "Composite":     f"{r['composite']:.2f}",
        })
    st.table(table_rows)

    # Determine rankings
    best = max(models, key=lambda m: results[m]["composite"])
    ranked = sorted(models, key=lambda m: results[m]["composite"], reverse=True)
    st.success(f"Best Overall (by your weights): {best}")

    # — Dynamic worded summary
    st.subheader("Summary")
    best_r = results[best]
    # Identify which criterion contributed most for the winner
    contribs = {
        "quality": best_r["contrib_quality"],
        "speed":   best_r["contrib_speed"],
        "cost":    best_r["contrib_cost"],
    }
    main_driver = max(contribs, key=contribs.get)

    # Who wins each raw criterion (for contrast)
    best_quality = max(models, key=lambda m: results[m]["score_quality"])
    best_speed   = max(models, key=lambda m: results[m]["score_speed"])
    best_cost    = max(models, key=lambda m: results[m]["score_cost"])

    summary_lines = []
    summary_lines.append(
        f"With your weights (quality {w_quality:.0%}, speed {w_speed:.0%}, cost {w_cost:.0%}), "
        f"{best} ranks first with a composite score of {best_r['composite']:.2f}."
    )
    summary_lines.append(
        f"The largest contribution to the winning score comes from {main_driver} "
        f"({best_r[f'contrib_{main_driver}']:.2f} of the total)."
    )

    # If the winner is not the best in any single criterion, call it a compromise
    is_compromise = (best != best_quality) and (best != best_speed) and (best != best_cost)
    if is_compromise:
        summary_lines.append(
            f"{best} is not the top model on any single criterion "
            f"(best quality: {best_quality}, best speed: {best_speed}, best cost: {best_cost}), "
            "but it offers the strongest overall balance given your weights."
        )

    # Runner-up comparison
    if len(ranked) > 1:
        runner = ranked[1]
        diff = results[best]["composite"] - results[runner]["composite"]
        summary_lines.append(
            f"The runner-up is {runner} with a composite of {results[runner]['composite']:.2f}, "
            f"{diff:.2f} points behind."
        )

    # If gpt-5-nano excels in cost or speed, mention it
    if results["gpt-5-nano"]["score_cost"] > 0.8 or results["gpt-5-nano"]["score_speed"] > 0.8:
        summary_lines.append(
            "Note: gpt-5-nano remains the most efficient choice on speed and/or cost, "
            "but it tends to underperform on raw quality."
        )

    # If gpt-5 is winner because of balance, add one-liner rationale
    if best == "gpt-5" and is_compromise:
        summary_lines.append(
            "This makes gpt-5 a good compromise: high enough quality to be reliable, "
            "while avoiding the worst trade-offs on latency and cost."
        )

    st.write(" ".join(summary_lines))
    st.caption("Adjust the weights in the sidebar to see how the ranking and rationale change.")

# ───────────────────── Run tips if nothing loaded ─────────────────────
if not uploaded_file:
    st.info("Upload a .pdf or .txt to begin.")
