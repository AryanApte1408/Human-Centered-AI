import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

def read_pdf(file):
    file.seek(0)
    data = file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    text_chunks = []
    for page in doc:
        text_chunks.append(page.get_text())
    return "\n".join(text_chunks)

# ——————————————————————————————————————————————————————————
# App header
# ——————————————————————————————————————————————————————————
st.title("Document Buddy")
st.write(
    "Upload a PDF or TXT and ask a question about it. "
    "Provide your OpenAI API key below (you can get one "
    "[here](https://platform.openai.com/account/api-keys))."
)

# ——————————————————————————————————————————————————————————
# API key input
# ——————————————————————————————————————————————————————————
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("🔑 Enter your OpenAI API key to continue.")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# ——————————————————————————————————————————————————————————
# File upload (.txt or .pdf) and session clearing
# ——————————————————————————————————————————————————————————
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .pdf)", type=["txt", "pdf"]
)

if not uploaded_file:
    # clear any stored document when user removes the file
    for key in ("document_text",):
        st.session_state.pop(key, None)

# ——————————————————————————————————————————————————————————
# Question input
# ——————————————————————————————————————————————————————————
question = st.text_area(
    "Ask a question about the document",
    placeholder="E.g. Give me a short summary.",
    disabled=not uploaded_file
)

# ——————————————————————————————————————————————————————————
# Process and query models
# ——————————————————————————————————————————————————————————
if uploaded_file and question:
    # parse document (cache in session to avoid re-reading on every rerun)
    if "document_text" not in st.session_state:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext == "txt":
            uploaded_file.seek(0)
            st.session_state.document_text = uploaded_file.read().decode("utf-8", errors="ignore")
        elif ext == "pdf":
            st.session_state.document_text = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

    doc = st.session_state.document_text
    prompt = f"DOCUMENT:\n{doc}\n\nQUESTION: {question}"

    # Models to compare (Responses API)
    models = ["gpt-3.5-turbo", "gpt-4.1", "gpt-5", "gpt-5-nano"]
    responses = {}

    for model_id in models:
        with st.spinner(f"Querying {model_id}…"):
            resp = client.responses.create(
                model=model_id,
                instructions=(
                    "You are a document-reading assistant. Answer only using the content of the document. "
                    "If the answer is not present, reply 'Not found in document.'"
                ),
                input=prompt,
            )
        # extract answer
        answer = resp.output_text
        responses[model_id] = answer

    # display in tabs
    tabs = st.tabs(models)
    for idx, model_id in enumerate(models):
        with tabs[idx]:
            st.subheader(f"{model_id}")
            st.markdown("**Prompt:**")
            st.write(prompt)
            st.markdown("---")
            st.markdown("**Answer:**")
            st.write(responses[model_id])

    # analysis section
    st.header("Analysis")
    st.write(
        "- **Quality:** Generally, `gpt-4.1` and `gpt-5` deliver the most accurate, well-reasoned answers.\n"
        "- **Cost & Speed:** `gpt-5-nano` is the fastest and cheapest, but may miss nuance. "
        "`gpt-5` often strikes the best balance between clarity, latency, and token cost.\n"
        "Summarize your pick for **best overall** and **best cost-effective** based on these results."
    )
