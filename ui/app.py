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

st.set_page_config(page_title="UGC Regulations Q&A", layout="wide")


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


warm()

st.title("UGC Regulations Q&A")
st.caption(
    "Answers come only from 9 UGC regulation documents, with citations. Retrieval: hybrid search + cross-encoder rerank. "
    f"Evaluation write-up and code: [GitHub]({GITHUB})."
)

examples = [
    "How often must a deemed university get its books audited?",
    "Is the UGC allowed to inspect a foreign university campus in India?",
    "What is the maximum duration of a PhD programme?",
    "What is the annual tuition fee at MIT-ADT University?",
]
cols = st.columns(len(examples))
for c, ex in zip(cols, examples):
    if c.button(ex, use_container_width=True):
        st.session_state["q"] = ex

q = st.text_input("Ask a question about the documents", key="q")
if st.button("Ask", type="primary") and q.strip():
    st.session_state["n"] = st.session_state.get("n", 0) + 1
    if st.session_state["n"] > PER_SESSION:
        st.warning("Session limit reached. Reload the page to continue.")
    elif not allowed():
        st.warning("The demo is rate-limited right now. Try again later.")
    else:
        try:
            t0 = time.time()
            with st.spinner("Searching and answering..."):
                chunks = retrieve(q)
                res = answer(q, chunks)
            ms = int((time.time() - t0) * 1000)
            if res["found"]:
                st.markdown(res["answer"])
                st.caption("Sources: " + "; ".join(f"{s['doc_id']} p{s['page']}" for s in res["sources"]) + f" | {ms} ms")
            else:
                st.info("The documents don't contain an answer to that question (NOT_FOUND).")
                st.caption(f"{ms} ms")
            with st.expander("Retrieved passages"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**[{i}] {c['doc_id']}, page {c['page']}**")
                    st.text(c["text"][:700])
        except Exception as e:
            st.error(f"Something went wrong ({type(e).__name__}). The model or database may be busy. Try again in a minute.")

st.caption("Research demo. Answers may be wrong or incomplete: check the cited passages and the original documents.")
