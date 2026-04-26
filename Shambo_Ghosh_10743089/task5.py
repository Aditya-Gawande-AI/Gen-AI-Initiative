import os
import re
import requests
import pdfplumber
from typing import TypedDict, Optional, Literal, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from google import genai

# --------------------
# ENV & GEMINI SETUP
# --------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
PDF_PATH = os.getenv("RAG_PDF_PATH")

client = genai.Client(api_key=GEMINI_API_KEY)


def gemini(prompt: str) -> str:
    resp = client.models.generate_content(model=MODEL, contents=[prompt])
    return (resp.text or "").strip()

# --------------------
# TRACE (OPTION A)
# --------------------
def log_step(state: "State", msg: str) -> None:
    """Append to state trace and print live."""
    state.setdefault("trace", []).append(msg)
    print(f"[TRACE] {msg}", flush=True)

# --------------------
# LOCAL PDF RAG
# --------------------
def load_pdf_text(path: str) -> str:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text = re.sub(r"\s+", " ", text)
                pages.append(text)

    return " ".join(pages)

def chunk_text(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks

def retrieve_chunks(query: str, chunks: List[str], k: int = 3) -> List[str]:
    q_terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
    scored = []
    for ch in chunks:
        score = sum(ch.lower().count(t) for t in q_terms)
        scored.append((score, ch))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for s, c in scored if s > 0][:k]

# Load and chunk the PDF once at startup
PDF_TEXT = load_pdf_text(PDF_PATH)
KB_CHUNKS = chunk_text(PDF_TEXT)
print(f"✅ PDF loaded | Chunks created: {len(KB_CHUNKS)}")

# --------------------
# SERPER WEB SEARCH
# --------------------
def serper_search(query: str) -> str:
    if not SERPER_API_KEY:
        return "Missing SERPER_API_KEY"

    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json().get("organic", [])[:3]

    return "\n".join(
        f"- {r.get('title')}: {r.get('snippet')}" for r in results
    ) or "No results found."

# --------------------
# LANGGRAPH STATE
# --------------------
class State(TypedDict, total=False):
    query: str
    next_step: Literal["llm", "rag", "web"]
    attempts: int
    max_attempts: int
    draft: Optional[str]
    validation_passed: bool
    feedback: Optional[str]
    force_fail_once: bool
    final: Optional[str]
    trace: List[str]   # ✅ Option A: trace list

# --------------------
# NODES
# --------------------
def supervisor(state: State) -> State:
    state["attempts"] += 1
    log_step(state, f"supervisor called | attempt={state['attempts']}")

    if state["attempts"] > state["max_attempts"]:
        log_step(state, "max_attempts exceeded -> forcing llm and passing validation")
        state["validation_passed"] = True
        state["next_step"] = "llm"
        return state

    decision = gemini(
        f"""Decide best node:
- llm (general)
- rag (SAP PDF)
- web (latest info)

Query: {state["query"]}
Feedback: {state.get("feedback","")}

Answer with only: llm | rag | web"""
    ).lower()

    chosen = decision if decision in ["llm", "rag", "web"] else "llm"
    state["next_step"] = chosen
    log_step(state, f"supervisor decided -> {chosen}")
    return state

def llm_node(state: State) -> State:
    log_step(state, "llm node called")
    state["draft"] = gemini(f"Answer clearly:\n{state['query']}")
    log_step(state, f"llm node produced draft_len={len(state['draft'] or '')}")
    return state

def rag_node(state: State) -> State:
    log_step(state, "rag node called")
    context = retrieve_chunks(state["query"], KB_CHUNKS)

    if not context:
        log_step(state, "rag retrieval found 0 chunks")
        state["draft"] = "No relevant information in the PDF."
        return state

    log_step(state, f"rag retrieval found {len(context)} chunks")
    state["draft"] = gemini(
        f"""Answer using this SAP document context:

{context}

Question: {state['query']}
End with: Sources: SAP Operations Guide"""
    )
    log_step(state, f"rag node produced draft_len={len(state['draft'] or '')}")
    return state

def web_node(state: State) -> State:
    log_step(state, "web node called")
    results = serper_search(state["query"])
    log_step(state, "web search finished (serper)")
    state["draft"] = gemini(
        f"""Answer using web results only:

{results}

Question: {state['query']}
Add Sources."""
    )
    log_step(state, f"web node produced draft_len={len(state['draft'] or '')}")
    return state

def validate(state: State) -> State:
    log_step(state, "validate node called")

    if state.get("force_fail_once"):
        state["force_fail_once"] = False
        state["validation_passed"] = False
        state["feedback"] = "Forced demo failure"
        log_step(state, "validate forced fail once -> returning to supervisor")
        return state

    if not state.get("draft") or len(state["draft"]) < 120:
        state["validation_passed"] = False
        state["feedback"] = "Answer too short or empty"
        log_step(state, "validate failed -> Answer too short/empty -> returning to supervisor")
        return state

    state["validation_passed"] = True
    log_step(state, "validate passed -> going to final")
    return state

def final_node(state: State) -> State:
    log_step(state, "final node called")
    state["final"] = state["draft"]
    return state

# --------------------
# BUILD GRAPH
# --------------------
graph = StateGraph(State)

graph.add_node("supervisor", supervisor)
graph.add_node("llm", llm_node)
graph.add_node("rag", rag_node)
graph.add_node("web", web_node)
graph.add_node("validate", validate)
graph.add_node("final", final_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    lambda s: s["next_step"],
    {"llm": "llm", "rag": "rag", "web": "web"},
)

graph.add_edge("llm", "validate")
graph.add_edge("rag", "validate")
graph.add_edge("web", "validate")

graph.add_conditional_edges(
    "validate",
    lambda s: "final" if s["validation_passed"] else "supervisor",
    {"final": "final", "supervisor": "supervisor"},
)

graph.add_edge("final", END)

app = graph.compile()

# --------------------
# RUN
# --------------------
if __name__ == "__main__":
    q = input("Enter question: ").strip()

    result = app.invoke({
        "query": q,
        "attempts": 0,
        "max_attempts": 3,
        "force_fail_once": False,
        "trace": [],  # ✅ initialize trace
    })

    print("\n✅ FINAL ANSWER\n")
    print(result.get("final", ""))




