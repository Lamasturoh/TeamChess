
"""
Lagos Transit Assistant: Streamlit front end for the RAG pipeline.

Run with:   streamlit run app.py
Needs:      OPENAI_API_KEY (Groq) and EMBEDDING_API_KEY in your .env file.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------
# CONFIG (same values as your notebook)
# ---------------------------------------------------------------
CHAT_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_BASE_URL = "https://api.groq.com/openai/v1"
CHAT_MODEL = "openai/gpt-oss-20b"

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_BASE_URL = "https://qwen-embed.publicaai.com/v1"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

CHROMA_PATH = "chroma_store"
CHROMA_COLLECTION = "lagos_transit"
SKLEARN_PATH = "lagos_transit_store/lagos_transit_index.json"

PROMPT = ChatPromptTemplate.from_template(
    """You are an expert AI Transit Consultant and Route Assistant for Lagos State, Nigeria.
You will be provided with context containing factual commuter survey data, route costs, travel durations, peak traffic times, and transit challenges across all 20 Local Government Areas (LGAs) in Lagos.

Context:
{context}

User Question:
{question}

Instructions:
1. Provide an accurate, clear, and actionable response based strictly on the provided transit context.
2. If the user asks about fares, travel times, or route recommendations, extract precise numbers (in Nigerian Naira ₦) and estimates from the context.
3. Keep the tone professional, concise, and helpful to a Lagos commuter or transit planner.
4. If the context does not contain enough information to fully answer the question, state what is known from the context and gently advise the user to refine their query.

Answer:"""
)

SUGGESTIONS = [
    ("Agege to Ikeja", "What is the average fare and travel time from Agege to Ikeja during the early morning rush?"),
    ("Worst rush-hour routes", "Which routes have the longest travel times during the evening rush hour?"),
    ("Cheapest ways to travel", "What are the cheapest transport options for commuters in Lagos and what do they cost?"),
    ("Biggest commuter problems", "What are the most common challenges commuters report on Lagos routes?"),
]

# ---------------------------------------------------------------
# PAGE SETUP + STYLE
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Lagos Transit Assistant",
    page_icon="🚌",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #10222E;
    --danfo: #F6B80B;
    --lagoon: #1D5C7A;
    --paper: #F2F5F7;
    --muted: #5B6B76;
    --line: #DCE3E8;
}

html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--paper); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 760px; }

/* ---------- Hero ---------- */
.danfo-stripe {
    height: 26px; border-radius: 6px; margin-bottom: 1.6rem;
    background: linear-gradient(to bottom,
        var(--danfo) 0 36%, var(--ink) 36% 64%, var(--danfo) 64% 100%);
}
.hero h1 {
    font-family: 'Bricolage Grotesque', sans-serif;
    font-weight: 800; font-size: 2.7rem; line-height: 1.05;
    letter-spacing: -0.02em; color: var(--ink); margin: 0 0 .7rem 0;
}
.hero p { color: var(--muted); font-size: 1.05rem; max-width: 34em; margin: 0 0 1.6rem 0; }

/* Compact header once chat has started */
.mini-head { display: flex; align-items: center; gap: .8rem; margin-bottom: 1.2rem; }
.mini-head .bar { width: 10px; height: 34px; border-radius: 3px;
    background: linear-gradient(to right, var(--danfo) 0 36%, var(--ink) 36% 64%, var(--danfo) 64% 100%); }
.mini-head span { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700;
    font-size: 1.35rem; color: var(--ink); }

/* ---------- Buttons (suggestions) ---------- */
.stApp .stButton > button {
    width: 100%; text-align: left; justify-content: flex-start;
    background: #fff; color: var(--ink); border: 1.5px solid var(--line);
    border-radius: 12px; padding: .85rem 1rem; font-weight: 600;
    transition: border-color .15s, box-shadow .15s;
}
.stApp .stButton > button:hover { border-color: var(--ink); color: var(--ink); box-shadow: 0 2px 0 var(--danfo); }
.stApp .stButton > button:focus-visible { outline: 3px solid var(--danfo); outline-offset: 2px; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] {
    background: #fff; border: 1px solid var(--line); border-radius: 14px;
    padding: 1rem 1.1rem; margin-bottom: .8rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--ink); border-color: var(--ink);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) * { color: #fff !important; }

[data-testid="stChatInput"] { border-radius: 14px; border: 1.5px solid var(--line); background: #fff; }
[data-testid="stChatInput"]:focus-within { border-color: var(--ink); box-shadow: 0 2px 0 var(--danfo); }

/* Sources */
.src { border-left: 3px solid var(--danfo); padding: .15rem 0 .15rem .8rem; margin: .5rem 0;
    color: var(--muted); font-size: .9rem; }
.src b { color: var(--ink); }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--ink); }
[data-testid="stSidebar"] * { color: #E8EEF2; }
[data-testid="stSidebar"] h3 {
    font-family: 'Bricolage Grotesque', sans-serif; color: #fff !important; font-weight: 700;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.06); color: #fff; border: 1px solid rgba(255,255,255,.18);
}
[data-testid="stSidebar"] .stButton > button:hover { border-color: var(--danfo); box-shadow: none; color: #fff; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.14); }
.side-note { font-size: .85rem; color: #9FB1BD !important; line-height: 1.45; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
@media (max-width: 640px) { .hero h1 { font-size: 2rem; } }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# RESOURCES
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Lagos transit data...")
def load_vector_store():
    """Open the saved vector store. Tries Chroma first, then the SKLearn JSON index."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )

    if Path(CHROMA_PATH).exists():
        db = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        if db._collection.count() > 0:
            return db

    if Path(SKLEARN_PATH).exists():
        return SKLearnVectorStore(
            embedding=embeddings, persist_path=SKLEARN_PATH, serializer="json"
        )

    raise FileNotFoundError(
        "No saved index found. Build it first (Chroma folder 'chroma_store' "
        "or 'lagos_transit_store/lagos_transit_index.json')."
    )


def build_chain(temperature: float):
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=CHAT_API_KEY,
        base_url=CHAT_BASE_URL,
        temperature=temperature,
        streaming=True,
    )
    return PROMPT | llm | StrOutputParser()


def retrieve(db, question: str, k: int):
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "fetch_k": max(10, k * 3)}
    )
    return retriever.invoke(question)


def render_sources(docs):
    with st.expander(f"Sources ({len(docs)} passages used)"):
        for i, d in enumerate(docs, 1):
            snippet = d.page_content.strip().replace("\n", " ")
            if len(snippet) > 320:
                snippet = snippet[:320].rstrip() + "..."
            meta = ", ".join(f"{k}: {v}" for k, v in (d.metadata or {}).items())
            label = f"Passage {i}" + (f" ({meta})" if meta else "")
            st.markdown(
                f'<div class="src"><b>{label}</b><br>{snippet}</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------
# STATE
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None


def ask(question: str):
    st.session_state.pending = question


def clear_chat():
    st.session_state.messages = []
    st.session_state.pending = None


# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("### Lagos Transit Assistant")
    st.markdown(
        '<p class="side-note">Answers come from commuter survey data covering '
        "fares, travel times and peak hours across Lagos State.</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Try asking**")
    for i, (label, q) in enumerate(SUGGESTIONS):
        st.button(label, key=f"side_{i}", on_click=ask, args=(q,))

    st.divider()
    st.markdown("**Settings**")
    k = st.slider("Passages to retrieve", 2, 10, 4)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.1, 0.05)
    show_sources = st.toggle("Show sources", value=True)

    st.divider()
    st.button("Clear conversation", on_click=clear_chat, disabled=not st.session_state.messages)

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if not st.session_state.messages and not st.session_state.pending:
    st.markdown(
        """
<div class="hero">
  <div class="danfo-stripe"></div>
  <h1>Ask about any route in Lagos</h1>
  <p>Get fares in naira, travel times and rush-hour patterns from commuter surveys across all 20 LGAs.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (label, q) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            st.button(label, key=f"hero_{i}", on_click=ask, args=(q,))
else:
    st.markdown(
        '<div class="mini-head"><div class="bar"></div><span>Lagos Transit Assistant</span></div>',
        unsafe_allow_html=True,
    )

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🚌" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            render_sources(msg["sources"])

# New question (typed or from a suggestion button)
typed = st.chat_input("Ask about a route, fare or travel time in Lagos")
question = typed or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🚌"):
        try:
            db = load_vector_store()
            with st.spinner("Checking the survey data..."):
                docs = retrieve(db, question, k)
            context = "\n\n".join(d.page_content for d in docs)

            chain = build_chain(temperature)
            answer = st.write_stream(
                chain.stream({"context": context, "question": question})
            )
            if show_sources:
                render_sources(docs)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": docs}
            )
        except FileNotFoundError as e:
            st.error(f"{e}")
        except Exception as e:  # network, auth, rate limit, etc.
            st.error(
                "Could not get an answer. Check that OPENAI_API_KEY and "
                f"EMBEDDING_API_KEY are set in your .env file.\n\nDetails: {e}"
            )
