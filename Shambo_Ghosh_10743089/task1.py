import os, requests, urllib3
from typing import TypedDict, Optional, List
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from langgraph.graph import StateGraph
from google import genai

# --- setup ---
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SAP_USERNAME = os.getenv("SAP_USERNAME")
SAP_PASSWORD = os.getenv("SAP_PASSWORD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
SAP_BASE_URL = os.getenv("SAP_BASE_URL")

client = genai.Client(api_key=GEMINI_API_KEY)

def g(prompt: str) -> str:
    r = client.models.generate_content(model=MODEL, contents=[prompt])
    return (r.text or "").strip()

# --- state ---
class GraphState(TypedDict):
    research: Optional[str]
    orders: Optional[List[dict]]
    explanation: Optional[str]

# --- agent 1: research (gemini) ---
def agent1_research(s: GraphState):
    s["research"] = g(
        "You are an SAP API expert. Explain how to retrieve top sales orders using "
        "API_SALES_ORDER_SRV/A_SalesOrder from SAP API Business Hub. "
        "Include OData options like $top, $select, and $orderby. Keep it clear and practical."
    )
    return s

# --- agent 2: fetch top 10 (sap api) ---
def agent2_fetch(s: GraphState):
    url = f"{SAP_BASE_URL}?$top=10"
    r = requests.get(
        url,
        auth=HTTPBasicAuth(SAP_USERNAME, SAP_PASSWORD),
        headers={"Accept": "application/json"},
        verify=False,
        timeout=10,
    )
    r.raise_for_status()
    s["orders"] = r.json().get("d", {}).get("results", []) or []
    return s

# --- agent 3: explain 1st order (gemini) ---
def agent3_explain(s: GraphState):
    if not s.get("orders"):
        s["explanation"] = "No sales orders returned from SAP."
        return s

    o = s["orders"][0]
    summary = {
        "SalesOrder": o.get("SalesOrder"),
        "SalesOrderType": o.get("SalesOrderType"),
        "SalesOrganization": o.get("SalesOrganization"),
        "DistributionChannel": o.get("DistributionChannel"),
        "OrganizationDivision": o.get("OrganizationDivision"),
        "SalesDistrict": o.get("SalesDistrict"),
        "SoldToParty": o.get("SoldToParty"),
    }

    s["explanation"] = g(
        "You are an SAP Sales and Distribution functional consultant. "
        "Explain this sales order in business-friendly terms (what each field means, why it matters):\n"
        f"{summary}"
    )
    return s

# --- graph ---
graph = StateGraph(GraphState)
graph.add_node("research", agent1_research)
graph.add_node("fetch", agent2_fetch)
graph.add_node("explain", agent3_explain)

graph.set_entry_point("research")
graph.add_edge("research", "fetch")
graph.add_edge("fetch", "explain")

app = graph.compile()

if __name__ == "__main__":
    out = app.invoke({"research": None, "orders": None, "explanation": None})

    print("\n--- AGENT 1 (RESEARCH) ---\n", out["research"])
    print("\n--- AGENT 2 (TOP 10 ORDERS) ---")
    for i, o in enumerate(out["orders"] or [], 1):
        print(f"{i}. SalesOrder={o.get('SalesOrder')}  Type={o.get('SalesOrderType')}  "
              f"SO={o.get('SalesOrganization')}  DC={o.get('DistributionChannel')}  "
              f"Div={o.get('OrganizationDivision')}  SoldTo={o.get('SoldToParty')}")

    print("\n--- AGENT 3 (EXPLANATION OF 1st ORDER) ---\n", out["explanation"])
