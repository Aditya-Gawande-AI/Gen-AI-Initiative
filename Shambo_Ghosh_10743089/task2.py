import os, requests
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph

from google import genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")

client = genai.Client(api_key=GEMINI_API_KEY)

class AgentState(TypedDict):
    company: str
    company_info: Optional[str]
    stock_info: Optional[str]
    report: Optional[str]

def gemini(prompt: str) -> str:
    r = client.models.generate_content(model=MODEL, contents=[prompt])
    return (r.text or "").strip()

def agent1_company(state: AgentState):
    state["company_info"] = gemini(
        f"Give a concise business overview of {state['company']}: industry, products/services, HQ, global presence."
    )
    return state

def agent2_stock(state: AgentState):
    if not SERPER_API_KEY:
        raise SystemExit("Missing SERPER_API_KEY in .env")

    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": f"{state['company']} stock price", "num": 5},
        timeout=10,
    )
    resp.raise_for_status()
    org = resp.json().get("organic", [])[:3]
    state["stock_info"] = "\n".join([o.get("snippet", "") for o in org if o.get("snippet")]) or "No stock snippets found."
    return state

def agent3_report(state: AgentState):
    state["report"] = gemini(
        "Create a clear business-friendly report (non-investment advice) using:\n"
        f"COMPANY INFO:\n{state.get('company_info','')}\n\n"
        f"STOCK INFO:\n{state.get('stock_info','')}\n\n"
        "Include: (1) overview summary, (2) current stock sentiment, (3) overall insight."
    )
    return state

graph = StateGraph(AgentState)
graph.add_node("company", agent1_company)
graph.add_node("stock", agent2_stock)
graph.add_node("report", agent3_report)
graph.set_entry_point("company")
graph.add_edge("company", "stock")
graph.add_edge("stock", "report")
app = graph.compile()

if __name__ == "__main__":
    company = input("Enter company name: ").strip()
    out = app.invoke({"company": company, "company_info": None, "stock_info": None, "report": None})
    print("\n--- FINAL REPORT ---\n", out["report"])
