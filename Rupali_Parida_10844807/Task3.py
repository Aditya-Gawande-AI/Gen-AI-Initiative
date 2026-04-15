import os, json, mimetypes, urllib3, requests
from dotenv import load_dotenv
from typing import Dict, Any, List
from typing_extensions import TypedDict, NotRequired

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# ---- Config ----
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SENDER = os.getenv("SENDER_MAIL")
QUERY = f"from:{SENDER} has:attachment after:2026/03/31"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))

SAP_URL = os.getenv("SAP_SO_URL")
SAP_USER = os.getenv("SAP_USERNAME")
SAP_PWD  = os.getenv("SAP_PASSWORD")

llm = init_llm("gpt-4o", max_tokens=1024)

# ---- Graph State ----
class GraphState(TypedDict, total=False):
    attachments: Dict[str, List[Dict[str, Any]]]
    agent2_response: NotRequired[List[Dict[str, Any]]]
    sales_order_number: NotRequired[List[str]]
    sap_raw_response: NotRequired[List[Dict[str, Any]]]

# ---- Helpers ----
def gmail_service():
    creds = Credentials.from_authorized_user_file("token.json", SCOPES) if os.path.exists("token.json") else None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            creds = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES).run_local_server(port=0)
        open("token.json", "w").write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def iter_parts(payload):
    for p in (payload or {}).get("parts", []) or []:
        yield from iter_parts(p)
    if payload and not payload.get("parts"):
        yield payload

def b64url_to_b64(s: str) -> str:
    s = s.replace("-", "+").replace("_", "/")
    return s + ("=" * (-len(s) % 4))

def data_url(filename: str, b64: str) -> str:
    mime = mimetypes.guess_type(filename)[0]
    return f"data:{mime};base64,{b64}"

# ---- Agent 1: Gmail attachments ----
def agent1(state: GraphState) -> dict:
    svc = gmail_service()
    msgs = svc.users().messages().list(userId="me", q=QUERY, maxResults=MAX_RESULTS).execute().get("messages", [])
    out: Dict[str, List[Dict[str, Any]]] = {}

    for m in msgs:
        msg_id = m["id"]
        msg = svc.users().messages().get(userId="me", id=msg_id, format="full").execute()
        payload = msg.get("payload", {})
        headers = {h["name"].lower(): h["value"] for h in payload.get("headers", []) if "name" in h and "value" in h}

        for part in iter_parts(payload):
            fn = part.get("filename")
            if not fn:
                continue
            body = part.get("body", {})
            raw = body.get("data")
            if not raw and body.get("attachmentId"):
                raw = svc.users().messages().attachments().get(
                    userId="me", messageId=msg_id, id=body["attachmentId"]
                ).execute().get("data")
            if not raw:
                continue

            b64 = b64url_to_b64(raw)
            out.setdefault(msg_id, []).append({
                "filename": fn,
                "data_url": data_url(fn, b64),
                "subject": headers.get("subject", "(no subject)"),
                "from": headers.get("from", "(unknown from)"),
            })

    print(f"[Agent1] Found {len(msgs)} emails. Attachments captured for {len(out)} message(s).")
    return {"attachments": out}

# ---- Agent 2: Send image to LLM -> barcode/serial ----
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a barcode decoding specialist. Your ONLY task is to read barcodes from images and "
     "determine the numeric value they encode. Return ONLY the value or NO_BARCODE."),
    ("user", [
        {"type": "text",
         "text": "Focus only on any barcode present. If unreadable, return exactly: NO_BARCODE."},
        {"type": "image_url", "image_url": {"url": "{image_data_url}"}},
    ])
])
CHAIN = PROMPT | llm | StrOutputParser()

def agent2(state: GraphState) -> dict:
    results = []
    for msg_id, items in (state.get("attachments") or {}).items():
        for it in items:
            resp = CHAIN.invoke({"image_data_url": it["data_url"]}).strip()
            print(f"[Agent2] {it['filename']} => {resp}")
            results.append({"message_id": msg_id, "filename": it["filename"], "barcode_result": resp})
    return {"agent2_response": results}

# ---- Agent 3: Create Sales Order in SAP for each decoded serial ----
def agent3(state: GraphState) -> dict:
    items = state.get("agent2_response") or []
    if not items:
        return {"sap_raw_response": [], "sales_order_number": []}

    if not (SAP_URL and SAP_USER and SAP_PWD):
        return {"sap_raw_response": [{"status": "FAILED", "reason": "Missing SAP env vars"}], "sales_order_number": []}

    s = requests.Session()
    s.auth = (SAP_USER, SAP_PWD)
    s.verify = False

    processed, so_nums = [], []

    for r in items:
        serial = (r.get("barcode_result") or "").strip()
        if not serial or serial == "NO_BARCODE":
            continue

        fetch = s.get(SAP_URL, headers={"Accept": "application/json", "X-CSRF-Token": "Fetch"})
        token = fetch.headers.get("X-CSRF-Token")
        if not token:
            processed.append({"serial_number": serial, "status": "FAILED", "reason": "CSRF fetch failed"})
            continue

        payload = {
            "SalesOrderType": "OR",
            "SalesOrganization": "1710",
            "DistributionChannel": "10",
            "OrganizationDivision": "00",
            "SoldToParty": "101",
            "PurchaseOrderByCustomer": serial,
            "CustomerPaymentTerms": "0001",
            "to_Item": [{"Material": "ORDER_BOM", "RequestedQuantity": "2"}],
        }

        post = s.post(SAP_URL, headers={"Accept":"application/json","Content-Type":"application/json","X-CSRF-Token":token},
                      data=json.dumps(payload))

        if post.status_code not in (200, 201):
            processed.append({"serial_number": serial, "status": "FAILED", "http_status": post.status_code})
            continue

        try:
            so = post.json().get("d", {}).get("SalesOrder", "")
        except Exception:
            so = ""

        processed.append({"serial_number": serial, "status": "SUCCESS", "sales_order_number": so})
        if so:
            so_nums.append(so)

    print(f"[Agent3] Processed {len(processed)} serial(s). Success: {len(so_nums)}")
    return {"sap_raw_response": processed, "sales_order_number": so_nums}

def final(state: GraphState) -> dict:
    print("SO Numbers:", state.get("sales_order_number", []))
    return {}

# ---- Build Graph ----
g = StateGraph(GraphState)
g.add_node("Fetch mails", agent1)
g.add_node("Get serial no", agent2)
g.add_node("get sales order number", agent3)
g.add_node("final", final)

g.add_edge(START, "Fetch mails")
g.add_edge("Fetch mails", "Get serial no")
g.add_edge("Get serial no", "get sales order number")
g.add_edge("get sales order number", "final")
g.add_edge("final", END)

app = g.compile()
app.invoke({})