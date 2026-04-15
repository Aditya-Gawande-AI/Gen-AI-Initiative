import os
import re
import json
import base64
import pickle
import requests
import urllib3
from typing_extensions import TypedDict, NotRequired

from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langgraph.graph import StateGraph, START, END

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------ ENV --------------------------
load_dotenv()

# ------------------------ LLM --------------------------
llm = init_llm("gpt-4o", temperature=0.3, max_tokens=1500)

# ------------------------ CONSTANTS --------------------------
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_PATH = "token.pkl"
CRED_PATH = "credentials.json"
TARGET_SUBJECT = "Product Serial Number Image for System Processing"

SERIAL_REGEX = re.compile(r"\b[A-Z0-9-]{6,}\b", re.IGNORECASE)

SAP_ENV_VARS = ("SAP_POST_URL", "SAP_USERNAME", "SAP_PASSWORD")

SAP_DEFAULT_PAYLOAD = {
    "SalesOrderType": "OR",
    "SalesOrganization": "1710",
    "DistributionChannel": "10",
    "OrganizationDivision": "00",
    "SoldToParty": "101",
    "CustomerPaymentTerms": "0001",
    "to_Item": [{"Material": "ORDER_BOM", "RequestedQuantity": "2"}],
}

# ------------------------ STATE --------------------------
class GraphState(TypedDict, total=False):
    image_path: NotRequired[str]
    serial_number: NotRequired[str]
    sap_response: NotRequired[str]

# ------------------------ SMALL HELPERS --------------------------
def _load_pickle(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def _save_pickle(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _get_header_value(headers, name: str) -> str:
    for h in headers or []:
        if h.get("name") == name:
            return h.get("value", "") or ""
    return ""

def _decode_attachment_data(data: str) -> bytes:
    return base64.urlsafe_b64decode(data.encode("utf-8"))

def _find_first_attachment(service, msg_id: str, parts):
    """
    Returns the saved file_path of the first found attachment with filename,
    scanning recursively (same logic as original, just cleaner).
    """
    for part in parts or []:
        filename = part.get("filename")
        body = part.get("body", {})

        # Direct attachment
        if filename and "attachmentId" in body:
            attachment_id = body["attachmentId"]
            attachment = (
                service.users()
                .messages()
                .attachments()
                .get(userId="me", messageId=msg_id, id=attachment_id)
                .execute()
            )

            file_data = _decode_attachment_data(attachment["data"])
            file_path = os.path.join(os.getcwd(), filename)

            with open(file_path, "wb") as f:
                f.write(file_data)

            print(f" - Attachment saved: {filename}")
            return file_path

        # Recursive multipart
        nested_parts = part.get("parts")
        if nested_parts:
            result = _find_first_attachment(service, msg_id, nested_parts)
            if result:
                return result

    return None

# ------------------------ GMAIL AUTH --------------------------
def authenticate_gmail():
    creds = _load_pickle(TOKEN_PATH)

    if not creds or not getattr(creds, "valid", False):
        if creds and getattr(creds, "expired", False) and getattr(creds, "refresh_token", None):
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CRED_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        _save_pickle(TOKEN_PATH, creds)

    service = build("gmail", "v1", credentials=creds)
    return service

# ------------------------ AGENT 1 --------------------------
def node_agent1_read_email(state: GraphState):
    print("\nAgent 1: Reading email and attachment.....")

    service = authenticate_gmail()
    results = service.users().messages().list(userId="me", maxResults=5).execute()
    messages = results.get("messages", [])

    for msg_meta in messages:
        msg_id = msg_meta["id"]
        msg = service.users().messages().get(userId="me", id=msg_id).execute()

        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        subject = _get_header_value(headers, "Subject")

        if TARGET_SUBJECT not in subject:
            continue

        print(" - Target Email Found")

        parts = payload.get("parts")
        if parts:
            image_path = _find_first_attachment(service, msg_id, parts)
            if image_path:
                print(" - Image fetched:", image_path)
                return {"image_path": image_path}

    raise Exception("No Matching Email with attachment found")

# ------------------------ AGENT 2 --------------------------
def node_agent2_extract_serial(state: GraphState):
    print("\nAgent 2: Extracting serial number.......")

    image_path = state["image_path"]

    with open(image_path, "rb") as img:
        image_bytes = img.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = llm.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract only the serial number from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        ]
    )

    text = getattr(response, "content", str(response))
    match = SERIAL_REGEX.search(text)

    if not match:
        raise Exception(f"Could not extract serial number. Model output: {text}")

    serial_number = match.group(0).strip().upper()
    print(" - Extracted Serial Number:", serial_number)
    return {"serial_number": serial_number}

# ------------------------ AGENT 3 --------------------------
def node_agent3_create_order(state: GraphState):
    print("\nAgent 3: Creating SAP Sales Order.........")

    serial = state["serial_number"]
    url = os.getenv("SAP_POST_URL")
    username = os.getenv("SAP_USERNAME")
    password = os.getenv("SAP_PASSWORD")

    if not url or not username or not password:
        raise Exception("Missing SAP_POST_URL / SAP_USERNAME / SAP_PASSWORD in .env")

    try:
        session = requests.Session()

        token_response = session.get(
            url,
            auth=(username, password),
            headers={"X-CSRF-Token": "Fetch", "Accept": "application/json"},
            verify=False,
        )

        csrf_token = token_response.headers.get("X-CSRF-Token") or token_response.headers.get("x-csrf-token")
        print(" - CSRF Token fetched:", csrf_token)

        payload = dict(SAP_DEFAULT_PAYLOAD)
        payload["PurchaseOrderByCustomer"] = serial

        response = session.post(
            url,
            auth=(username, password),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-CSRF-Token": csrf_token or "",
            },
            json=payload,
            verify=False,
        )

        result = response.text
    except Exception as e:
        result = str(e)

    print(" - SAP Response:\n", result)
    return {"sap_response": result}

# ------------------------ FINAL PRINT --------------------------
def node_print(state: GraphState):
    print("\nFINAL RESULT:")

    try:
        data = json.loads(state["sap_response"])
        sales_order = data.get("d", {}).get("SalesOrder")

        print(" - Sales Order Created Successfully!")
        print(" - Order ID:", sales_order)
        print(" - Serial Number:", state.get("serial_number"))
    except Exception:
        print(state.get("sap_response"))

    return {}

# ------------------------ GRAPH --------------------------
builder = StateGraph(GraphState)

builder.add_node("agent1", node_agent1_read_email)
builder.add_node("agent2", node_agent2_extract_serial)
builder.add_node("agent3", node_agent3_create_order)
builder.add_node("print", node_print)

builder.add_edge(START, "agent1")
builder.add_edge("agent1", "agent2")
builder.add_edge("agent2", "agent3")
builder.add_edge("agent3", "print")
builder.add_edge("print", END)

app = builder.compile()

# ------------------------ RUN --------------------------
app.invoke({})