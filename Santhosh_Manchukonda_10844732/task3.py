
import os
import base64
import re
import json
import pickle
import requests
import urllib3
from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired

from langgraph.graph import StateGraph, START, END

# Gmail
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# SAP AI Core GenAI Hub (LLM)
from gen_ai_hub.proxy.langchain.init_models import init_llm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

# -------------------- LLM (SAP AI Core via GenAI Hub) --------------------
llm = init_llm("gpt-4o", temperature=0.3, max_tokens=500)

# -------------------- State --------------------
class GraphState(TypedDict, total=False):
    image_path: NotRequired[str]
    serial_number: NotRequired[str]
    sap_response: NotRequired[str]

# -------------------- Gmail Auth --------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def gmail_service():
    creds = None

    if os.path.exists("token.pkl"):
        with open("token.pkl", "rb") as f:
            creds = pickle.load(f)

    if not creds:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.pkl", "wb") as f:
            pickle.dump(creds, f)

    return build("gmail", "v1", credentials=creds)

# -------------------- Download Attachment (image) --------------------
def download_first_image_attachment(service, msg_id, parts):
    for part in parts:
        filename = part.get("filename", "")
        body = part.get("body", {})
        att_id = body.get("attachmentId")

        if filename and att_id and filename.lower().endswith((".jpg", ".jpeg", ".png")):
            att = service.users().messages().attachments().get(
                userId="me", messageId=msg_id, id=att_id
            ).execute()

            file_bytes = base64.urlsafe_b64decode(att["data"])
            os.makedirs("downloads", exist_ok=True)
            path = os.path.join("downloads", filename)

            with open(path, "wb") as f:
                f.write(file_bytes)

            return path

        # recursive for nested parts
        if "parts" in part:
            p = download_first_image_attachment(service, msg_id, part["parts"])
            if p:
                return p

    return None

# -------------------- Agent 1: Read Gmail + Get Image --------------------
def agent1(state: GraphState):
    print("\nAgent 1: Reading Gmail and downloading image...")

    service = gmail_service()
    results = service.users().messages().list(userId="me", maxResults=10).execute()
    messages = results.get("messages", [])

    for meta in messages:
        msg_id = meta["id"]
        msg = service.users().messages().get(userId="me", id=msg_id).execute()
        payload = msg.get("payload", {})

        parts = payload.get("parts", [])
        if parts:
            image_path = download_first_image_attachment(service, msg_id, parts)
            if image_path:
                print(f"✅ Agent1: Downloaded image -> {image_path}")
                return {"image_path": image_path}

    raise RuntimeError("No image attachment found in last 10 emails.")

# -------------------- Agent 2: LLM extracts serial from IMAGE --------------------
def agent2(state: GraphState):
    print("\nAgent 2: Extracting serial number using SAP AI Core (GenAI Hub)...")

    image_path = state["image_path"]
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = llm.invoke([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Extract ONLY the serial number from this image. Return only the serial number."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
        ]
    }])

    text_out = getattr(resp, "content", str(resp)).strip()

    # Extract first likely serial token
    m = re.search(r"(SN[- ]?\d{4,}|\b\d{6,}\b|\b[A-Z0-9]{6,}\b)", text_out, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"Serial number not found in model output: {text_out}")

    serial = m.group(1)
    print(f"✅ Agent2: Extracted Serial Number -> {serial}")
    return {"serial_number": serial}

# -------------------- Agent 3: Create Sales Order in SAP --------------------
def agent3(state: GraphState):
    print("\nAgent 3: Creating SAP Sales Order...")

    url = os.getenv("SAP_POST_URL")
    username = os.getenv("SAP_USERNAME")
    password = os.getenv("SAP_PASSWORD")

    serial = state["serial_number"]
    session = requests.Session()

    # Fetch CSRF token (required for POST)
    token_resp = session.get(
        url,
        auth=(username, password),
        headers={"X-CSRF-Token": "Fetch"},
        verify=False
    )
    csrf_token = token_resp.headers.get("X-CSRF-Token") or token_resp.headers.get("x-csrf-token")

    payload = {
        "SalesOrderType": "OR",
        "SalesOrganization": "1710",
        "DistributionChannel": "10",
        "OrganizationDivision": "00",
        "SoldToParty": "101",
        "PurchaseOrderByCustomer": serial,
        "CustomerPaymentTerms": "0001",
        "to_Item": [{"Material": "ORDER_BOM", "RequestedQuantity": "2"}]
    }  
    resp = session.post(
        url,
        auth=(username, password),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-CSRF-Token": csrf_token
        },
        json=payload,
        verify=False
    )

    print("✅ Agent3: SAP Response received")
    return {"sap_response": resp.text}

# -------------------- Final Print --------------------
def final_print(state: GraphState):
    print("\n✅ FINAL RESULT:")
    print(" - Image Path:", state.get("image_path"))
    print(" - Serial Number:", state.get("serial_number"))

    try:
        data = json.loads(state.get("sap_response", "{}"))
        order = data.get("d", {}).get("SalesOrder")
        if order:
            print(" - Sales Order:", order)
        else:
            print(" - SAP Response:", state.get("sap_response"))
    except Exception:
        print(" - SAP Response:", state.get("sap_response"))

    return {}

# -------------------- Graph --------------------
builder = StateGraph(GraphState)
builder.add_node("agent1", agent1)
builder.add_node("agent2", agent2)
builder.add_node("agent3", agent3)
builder.add_node("final", final_print)

builder.add_edge(START, "agent1")
builder.add_edge("agent1", "agent2")
builder.add_edge("agent2", "agent3")
builder.add_edge("agent3", "final")
builder.add_edge("final", END)

app = builder.compile()

if __name__ == "__main__":
    app.invoke({})
