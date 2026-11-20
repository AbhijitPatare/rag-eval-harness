import html
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.generate import answer
from app.retrieval import search

PER_SESSION = 8
GLOBAL_PER_HOUR = 60
GITHUB = "https://github.com/AbhijitPatare/rag-eval-harness"
EXAMPLES = {
    "Audit requirements": "How often must a deemed university get its books audited?",
    "Inspecting foreign campuses": "Is the UGC allowed to inspect a foreign university campus in India?",
    "PhD duration": "What is the maximum duration of a PhD programme?",
    "Out-of-scope question": "What is the annual tuition fee at MIT-ADT University?",
}

st.set_page_config(page_title="UGC Regulations Q&A", layout="centered")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp{background:#F6F7F9}
.stApp,.stMarkdown,.stTextInput input,.stButton button,.stExpander summary{font-family:Inter,system-ui,sans-serif}
#MainMenu,footer{visibility:hidden}
header[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1080px;padding-top:0;padding-bottom:3rem}
.top{display:flex;align-items:center;gap:12px;padding:16px 0;border-bottom:1px solid #E4E7EC;margin-bottom:32px}
.top .name{font-weight:700;font-size:17px;color:#101828}
.pill{font-size:12px;font-weight:600;background:#EEF3FF;color:#1F3FA8;padding:4px 10px;border-radius:999px}
.sp{flex:1}
.top a,.evh a{font-size:14px;font-weight:500;color:#1F3FA8;text-decoration:none}
h1.hero{margin:0;padding:0;font-size:36px;letter-spacing:-0.02em;color:#101828}
p.sub{margin:8px 0 24px;font-size:16px;line-height:1.5;color:#475467;max-width:680px}
.qlbl{font-size:13px;font-weight:600;color:#475467;margin-bottom:6px}
.stTextInput input{height:48px;border-radius:10px;font-size:16px;background:#fff}
[data-testid="stFormSubmitButton"] button{height:48px;border-radius:10px;font-weight:600}
.stButton button{border-radius:999px;font-size:13px;min-height:0;padding:4px 12px;background:#fff;border:1px solid #D0D5DD;color:#344054}
.stButton button:hover{border-color:#1F4FD8;color:#1F3FA8}
.card{background:#fff;border:1px solid #E4E7EC;border-radius:14px;padding:24px;margin-top:24px}
.row{display:flex;align-items:center;gap:10px}
.lbl{font-size:12px;font-weight:600;letter-spacing:.06em;color:#475467}
.badge{font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px}
.ok{background:#E6F4EE;color:#0B5D43}
.warn{background:#FEF3E2;color:#7A4B00}
.t{font-size:13px;color:#475467}
.ans{margin:14px 0 0;font-size:18px;line-height:1.6;color:#101828}
.cite{font-size:13px;font-weight:600;background:#EEF3FF;color:#1F3FA8;padding:2px 7px;border-radius:6px}
h2.h2{font-size:15px;margin:28px 0 12px;padding:0;color:#101828}
.src{background:#fff;border:1px solid #E4E7EC;border-radius:12px;padding:16px;height:100%;box-sizing:border-box}
.src.cited{border-color:#1F4FD8}
.num{font-size:13px;font-weight:600;background:#F2F4F7;color:#475467;padding:2px 7px;border-radius:6px}
.num.on{background:#EEF3FF;color:#1F3FA8}
.tag{font-size:12px;font-weight:600;color:#475467}
.src.cited .tag{color:#1F3FA8}
.doc{font-size:13px;font-weight:600;margin-top:8px;color:#101828;word-break:break-all}
.pg{font-size:12px;color:#475467;margin-top:2px}
.ex{font-size:13px;line-height:1.5;color:#475467;margin-top:8px;max-height:84px;overflow:hidden}
.evh{display:flex;align-items:center}
.evh h2{margin:0;font-size:15px;color:#101828}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:16px}
.stat b{display:block;font-size:26px;color:#101828}
.stat span{font-size:13px;line-height:1.4;color:#475467}
.foot{margin-top:24px;font-size:12px;line-height:1.5;color:#475467}
@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

HEADER = (
    '<div class="top"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#1F4FD8" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3h8l4 4v14H6z"/>'
    '<path d="M14 3v4h4M9 12h6M9 16h6"/></svg><span class="name">UGC Regulations Q&amp;A</span>'
    '<span class="pill">Evaluated RAG</span><span class="sp"></span>'
    f'<a href="{GITHUB}#evaluation-design">Methodology</a><a href="{GITHUB}" style="margin-left:16px">GitHub</a></div>'
    '<h1 class="hero">Ask the UGC regulations</h1>'
    '<p class="sub">Answers come only from 9 UGC documents and cite their sources. '
    "If the documents don't say, it tells you instead of guessing.</p>"
)

EVAL = (
    '<div class="card"><div class="evh"><h2>How well does this work?</h2><span class="sp"></span>'
    f'<a href="{GITHUB}#results">Read the evaluation</a></div><div class="stats">'
    '<div class="stat"><b>0.84</b><span>hit@5 on 31 held-out questions (26/31)</span></div>'
    '<div class="stat"><b>27/31</b><span>answers judged correct on the test set</span></div>'
    '<div class="stat"><b>8/8</b><span>out-of-scope questions correctly refused</span></div>'
    '<div class="stat"><b>Hybrid + rerank</b><span>pgvector, full-text, cross-encoder; Gemini answers</span></div>'
    '</div></div>'
    '<p class="foot">Research demo. Answers can be wrong or incomplete: check the cited passages and the original documents.</p>'
)


@st.cache_resource
def global_log():
    return []


def allowed() -> bool:
    now = time.time()
    log = global_log()
    log[:] = [t for t in log if now - t < 3600]
    if len(log) >= GLOBAL_PER_HOUR:
        return False
    log.append(now)
    return True


def retrieve(q: str):
    last = None
    for i in range(3):  # a hosted DB may be waking from idle
        try:
            return search(q, "recursive", "hybrid", True, 8)
        except Exception as e:
            last = e
            time.sleep(3 * (i + 1))
    raise last


@st.cache_resource(show_spinner="Loading models (first start takes a minute)...")
def warm():
    try:
        retrieve("warm up")
    except Exception:
        pass
    return True


def esc(s: str) -> str:
    return html.escape(" ".join(s.split()))


def with_cites(text: str) -> str:
    t = html.escape(text).replace("\n", "<br>")
    return re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', t)


def src_card(i: int, c: dict, cited: bool) -> str:
    return (
        f'<div class="src{" cited" if cited else ""}"><div class="row">'
        f'<span class="num{" on" if cited else ""}">{i}</span><span class="tag">{"Cited" if cited else "Retrieved"}</span></div>'
        f'<div class="doc">{esc(c["doc_id"])}</div><div class="pg">Page {c["page"]}</div>'
        f'<div class="ex">{esc(c["text"])[:260]}...</div></div>'
    )


def pick(text: str):
    st.session_state["q"] = text
    st.session_state["run_q"] = text


def run(q: str):
    st.session_state["n"] = st.session_state.get("n", 0) + 1
    if st.session_state["n"] > PER_SESSION:
        st.session_state["res"] = {"error": "Session limit reached. Reload the page to continue."}
        return
    if not allowed():
        st.session_state["res"] = {"error": "The demo is rate-limited right now. Try again later."}
        return
    try:
        t0 = time.time()
        with st.spinner("Searching and answering..."):
            chunks = retrieve(q)
            res = answer(q, chunks)
        res.update(chunks=chunks, ms=int((time.time() - t0) * 1000))
        st.session_state["res"] = res
    except Exception as e:
        st.session_state["res"] = {
            "error": f"Something went wrong ({type(e).__name__}). The model or database may be busy. Try again in a minute."
        }


def render(res: dict):
    chunks = res["chunks"]
    cited = []
    if res["found"]:
        for n in (int(x) for x in re.findall(r"\[(\d+)\]", res["answer"])):
            if 1 <= n <= len(chunks) and n not in cited:
                cited.append(n)
        ns = len({(chunks[n - 1]["doc_id"], chunks[n - 1]["page"]) for n in cited}) or len(res["sources"])
        badge = f'<span class="badge ok">Grounded in {ns} source{"s" if ns != 1 else ""}</span>'
        body = with_cites(res["answer"])
    else:
        badge = '<span class="badge warn">Not in the documents</span>'
        body = "The documents don't contain an answer to that question. Rather than guess, the system declines."
    st.markdown(
        f'<div class="card"><div class="row"><span class="lbl">ANSWER</span>{badge}<span class="sp"></span>'
        f'<span class="t">{res["ms"] / 1000:.1f} s</span></div><p class="ans">{body}</p></div>',
        unsafe_allow_html=True,
    )
    order = cited + [i for i in range(1, len(chunks) + 1) if i not in cited]
    st.markdown('<h2 class="h2">Retrieved passages</h2>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, i in zip(cols, order[:3]):
        col.markdown(src_card(i, chunks[i - 1], i in cited), unsafe_allow_html=True)
    with st.expander(f"All {len(chunks)} retrieved passages"):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**[{i}] {c['doc_id']}, page {c['page']}**")
            st.text(c["text"][:700])


warm()
st.markdown(HEADER, unsafe_allow_html=True)

st.session_state.setdefault("q", "")
with st.container(border=True):
    st.markdown('<div class="qlbl">Your question</div>', unsafe_allow_html=True)
    with st.form("ask", border=False):
        c1, c2 = st.columns([6, 1])
        q = c1.text_input("Your question", key="q", label_visibility="collapsed",
                          placeholder="Ask a question about the documents")
        go = c2.form_submit_button("Ask", type="primary", use_container_width=True)
    for col, (label, text) in zip(st.columns(4), EXAMPLES.items()):
        col.button(label, key=f"ex_{label}", on_click=pick, args=(text,), use_container_width=True)

pending = st.session_state.pop("run_q", None)
if go and q.strip():
    pending = q
if pending:
    run(pending)

res = st.session_state.get("res")
if res:
    if "error" in res:
        st.error(res["error"])
    else:
        render(res)

st.markdown(EVAL, unsafe_allow_html=True)