"""
app.py - Colorful, interactive Case Q&A.   Run:  streamlit run app.py
Uses a FREE AI by default (Google Gemini). See README for Ollama or Anthropic.
"""
import html
import os
import re

import chromadb
import requests
import streamlit as st

DB_DIR = "chroma_db"
COLLECTION = "cases"
TOP_K = 6
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
ANTHROPIC_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a legal research assistant. You answer questions about ONE court case \
using only the document excerpts provided to you.

Rules:
- Use only the excerpts. Never use outside knowledge about the case.
- After each fact, cite its source in the form [file, page N].
- If the excerpts do not contain the answer, say: "This is not found in the uploaded documents for this case."
- Be clear and concise. Use simple language.
- You are not a lawyer; do not give legal advice."""

GRADIENTS = [
    "linear-gradient(135deg,#6366f1,#8b5cf6)", "linear-gradient(135deg,#ec4899,#f97316)",
    "linear-gradient(135deg,#06b6d4,#3b82f6)", "linear-gradient(135deg,#10b981,#06b6d4)",
    "linear-gradient(135deg,#f59e0b,#ef4444)", "linear-gradient(135deg,#8b5cf6,#ec4899)",
]
ICONS = {"judgment": "⚖️", "order": "📜", "proof": "🔎", "other": "📄"}
SUGGESTIONS = ["Who are the parties?", "What was the final order?", "What evidence was produced?",
               "Any deadlines or payments ordered?", "Why did the court decide this way?"]

CSS = """<style>
.hero{background:linear-gradient(135deg,#4f46e5,#9333ea 55%,#ec4899);padding:1.4rem 1.8rem;border-radius:18px;color:#fff;margin-bottom:1rem;box-shadow:0 8px 24px rgba(79,70,229,.3)}
.hero h1{color:#fff;margin:0;font-size:1.9rem;padding:0}
.hero p{margin:.3rem 0 0;opacity:.92}
.badge{padding:2px 10px;border-radius:999px;font-size:.72rem;font-weight:600;color:#fff;margin-right:6px}
.b-judgment{background:#2563eb}.b-order{background:#f59e0b}.b-proof{background:#10b981}.b-other{background:#6b7280}
.case-card{border-radius:14px;padding:14px 16px;color:#fff;margin-bottom:6px;min-height:90px}
.case-card b{font-size:.95rem}.case-card small{opacity:.92}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#eef2ff,#fce7f3)}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:#1e1b4b !important}
[data-testid="stMetric"]{background:rgba(99,102,241,.1);border-left:5px solid #6366f1;padding:10px 14px;border-radius:12px}
.stButton>button{border-radius:999px;border:1px solid #a5b4fc;transition:all .15s}
.stButton>button:hover{background:linear-gradient(135deg,#6366f1,#ec4899);color:#fff;border-color:transparent;transform:translateY(-1px)}
[data-testid="stChatMessage"]{border-radius:14px;padding:.6rem .9rem;margin-bottom:.5rem;background:rgba(99,102,241,.07)}
</style>"""

st.set_page_config(page_title="Case Q&A", page_icon="⚖️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ---------- AI ----------
def call_llm(system, messages, max_tokens=1000):
    try:
        if PROVIDER == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                              "parts": [{"text": m["content"]}]} for m in messages],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
            r = requests.post(url, json=body, timeout=120,
                              headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            r.raise_for_status()
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        if PROVIDER == "ollama":
            body = {"model": OLLAMA_MODEL, "stream": False,
                    "messages": [{"role": "system", "content": system}] + messages}
            r = requests.post("http://localhost:11434/api/chat", json=body, timeout=600)
            r.raise_for_status()
            return r.json()["message"]["content"]
        if PROVIDER == "anthropic":
            import anthropic
            resp = anthropic.Anthropic().messages.create(
                model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=system, messages=messages)
            return resp.content[0].text
        return f"Unknown LLM_PROVIDER '{PROVIDER}'. Use gemini, ollama or anthropic."
    except requests.HTTPError as e:
        return f"The AI service returned an error ({e.response.status_code}): {e.response.text[:300]}"
    except requests.ConnectionError:
        return "Could not connect to the AI service. Check your internet (or that Ollama is running)."
    except Exception as e:
        return f"Something went wrong while calling the AI: {e}"


# ---------- data ----------
@st.cache_resource
def get_collection():
    return chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)


@st.cache_data
def case_stats():
    data = get_collection().get(include=["metadatas"])
    stats = {}
    for m in data["metadatas"]:
        s = stats.setdefault(m["case"], {"docs": set(), "chunks": 0})
        s["docs"].add((m["file"], m["doc_type"]))
        s["chunks"] += 1
    return {k: {"docs": sorted(v["docs"]), "chunks": v["chunks"]} for k, v in sorted(stats.items())}


def retrieve(case, question):
    res = get_collection().query(query_texts=[question], n_results=TOP_K, where={"case": case})
    return list(zip(res["documents"][0], res["metadatas"][0]))


def format_context(chunks):
    return "\n\n---\n\n".join(
        f"[{m['file']}, page {m['page']}] ({m['doc_type']})\n{t}" for t, m in chunks)


def ask_llm(case, question, history):
    chunks = retrieve(case, question)
    if not chunks:
        return "This is not found in the uploaded documents for this case.", []
    msg = f"Case: {case}\n\nExcerpts:\n{format_context(chunks)}\n\nQuestion: {question}"
    return call_llm(SYSTEM_PROMPT, history[-6:] + [{"role": "user", "content": msg}], 1000), chunks


def summarize_case(case):
    data = get_collection().get(where={"case": case}, limit=30, include=["documents", "metadatas"])
    chunks = list(zip(data["documents"], data["metadatas"]))
    prompt = (f"Case: {case}\n\nExcerpts:\n{format_context(chunks)}\n\n"
              "Write a structured summary with these headings: Parties, Facts in brief, Issues, "
              "Court's reasoning, Final order. If something is missing, write 'Not found'. "
              "Cite sources as [file, page N].")
    return call_llm(SYSTEM_PROMPT, [{"role": "user", "content": prompt}], 1500)


# ---------- helpers ----------
def pretty(name):
    return name.replace("_", " ")


def badge(dtype):
    dtype = dtype if dtype in ICONS else "other"
    return f'<span class="badge b-{dtype}">{ICONS[dtype]} {dtype}</span>'


def select_case(name):
    st.session_state.case_select = name


def set_pending(q):
    st.session_state.pending = q


def render_sources(sources):
    with st.expander(f"📚 Sources used ({len(sources)})"):
        for text, meta in sources:
            st.markdown(f"{badge(meta['doc_type'])} **{html.escape(meta['file'])}** · page {meta['page']} · {html.escape(str(meta.get('section', '')))}",
                        unsafe_allow_html=True)
            st.caption(text[:400] + ("..." if len(text) > 400 else ""))


def verify_citations(answer, chunks):
    """HARNESS check: every [file, page N] the AI cites must exist in the retrieved chunks."""
    cites = re.findall(r"([\w\-. ()]+\.(?:pdf|txt)),\s*page\s*(\d+)", answer, flags=re.I)
    if not cites:
        return "ℹ️ No citations in this answer."
    valid = {(m["file"], str(m["page"])) for _, m in chunks}
    bad = [f"{f.strip()} p.{p}" for f, p in cites
           if not any(f.strip().endswith(vf) and p == vp for vf, vp in valid)]
    if bad:
        return "⚠️ Unverified citation(s): " + ", ".join(bad)
    return f"✅ All {len(cites)} citation(s) verified against the retrieved documents."


# ---------- checks ----------
if PROVIDER == "gemini" and not os.environ.get("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is not set. Get a free key at aistudio.google.com, set it in your "
             "terminal, and restart the app.")
    st.stop()
if PROVIDER == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set.")
    st.stop()
try:
    stats = case_stats()
except Exception:
    st.error("No database found. Put your cases in 'cases/' and run: python ingest.py")
    st.stop()
if not stats:
    st.warning("The database is empty. Add case folders and run: python ingest.py")
    st.stop()
cases = list(stats)

# ---------- sidebar ----------
st.sidebar.markdown("## ⚖️ Case Finder")
search = st.sidebar.text_input("🔍 Search case name")
matches = [c for c in cases if search.lower() in c.lower()] if search else cases
if not matches:
    st.sidebar.warning("No case matches that search.")
    st.stop()
if st.session_state.get("case_select") not in matches:
    st.session_state.case_select = matches[0]
case = st.sidebar.selectbox("📂 Select case", matches, key="case_select", format_func=pretty)
st.sidebar.caption(f"{len(cases)} cases loaded")
st.sidebar.markdown("**Documents in this case**")
for fname, dtype in stats[case]["docs"]:
    st.sidebar.markdown(f"{badge(dtype)} {html.escape(fname)}", unsafe_allow_html=True)

if st.session_state.get("current_case") != case:
    st.session_state.current_case = case
    st.session_state.messages = []
    st.session_state.summary = None

# ---------- header ----------
st.markdown(f'<div class="hero"><h1>⚖️ {html.escape(pretty(case))}</h1>'
            '<p>Ask anything. Answers come only from this case\'s documents, with sources.</p></div>',
            unsafe_allow_html=True)
n_q = sum(1 for m in st.session_state.messages if m["role"] == "user")
c1, c2, c3 = st.columns(3)
c1.metric("📁 Documents", len(stats[case]["docs"]))
c2.metric("🧩 Text chunks", stats[case]["chunks"])
c3.metric("💬 Questions asked", n_q)

with st.expander("🗂️ Browse all cases"):
    cols = st.columns(3)
    for i, name in enumerate(cases):
        with cols[i % 3]:
            st.markdown(f'<div class="case-card" style="background:{GRADIENTS[i % len(GRADIENTS)]}">'
                        f'<b>{html.escape(pretty(name))}</b><br><small>📁 {len(stats[name]["docs"])} documents'
                        f'</small></div>', unsafe_allow_html=True)
            st.button("Open case", key=f"open_{i}", on_click=select_case, args=(name,))

# ---------- handle a question (typed or from a suggestion chip) ----------
typed = st.chat_input("Type your question about this case...")
question = typed or st.session_state.pop("pending", None)
if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.spinner("🔎 Searching the case documents..."):
        answer, sources = ask_llm(case, question, history)
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources,
                                         "check": verify_citations(answer, sources)})

# ---------- tabs ----------
tab_ask, tab_sum, tab_docs = st.tabs(["💬 Ask questions", "✨ Case summary", "📁 Documents"])

with tab_ask:
    st.markdown("**Try a quick question:**")
    chip_cols = st.columns(len(SUGGESTIONS))
    for i, q in enumerate(SUGGESTIONS):
        chip_cols[i].button(q, key=f"chip_{i}", on_click=set_pending, args=(q,), use_container_width=True)
    if not st.session_state.messages:
        st.info("👆 Click a quick question or type your own in the box at the bottom.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "⚖️"):
            st.markdown(m["content"])
            if m.get("check"):
                st.caption(m["check"])
            if m.get("sources"):
                render_sources(m["sources"])

with tab_sum:
    if st.button("✨ Generate summary", type="primary"):
        with st.spinner("Reading the case documents..."):
            st.session_state.summary = summarize_case(case)
    if st.session_state.summary:
        st.markdown(st.session_state.summary)
        st.download_button("⬇️ Download summary", st.session_state.summary,
                           file_name=f"{case}_summary.txt")
    else:
        st.caption("Click the button to get a structured summary of this case.")

with tab_docs:
    for fname, dtype in stats[case]["docs"]:
        st.markdown(f"{badge(dtype)} **{html.escape(fname)}**", unsafe_allow_html=True)

st.caption("Research aid only, not legal advice. Sample cases are fictional. Verify against the original documents.")
