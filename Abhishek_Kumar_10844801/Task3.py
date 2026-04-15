from dotenv import load_dotenv
import os
import mimetypes
import urllib3
import requests
import json
from typing import Dict, List, Any
from typing_extensions import TypedDict, NotRequired

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = init_llm("gpt-4o", max_tokens=1024)


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SENDER = os.getenv("SENDER_MAIL")
print(SENDER)
QUERY = f"from:{SENDER} has:attachment"
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "50"))


# Graph
class GraphState(TypedDict, total=False):
    attachments: Dict[str, List[Dict[str, Any]]]
    agent1_response: NotRequired[str]
    agent2_response: NotRequired[str]
    serial_number: NotRequired[str]
    agent3_response: NotRequired[str]
    sales_order_number: NotRequired[str]
    sap_raw_response: NotRequired[Dict[str, Any]]


def gmail_service():
    creds = None

    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def iter_parts(payload):
    if not payload:
        return
    parts = payload.get("parts")
    if parts:
        for p in parts:
            yield from iter_parts(p)
    else:
        yield payload


# base64url to base64 format conversion
def gmail_base64url_to_base64(gmail_b64url: str) -> str:

    b64 = gmail_b64url.replace("-", "+").replace("_", "/")
    padding = "=" * (-len(b64) % 4)
    return b64 + padding


def guess_mime_type(filename: str, fallback="application/octet-stream") -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or fallback


def to_data_url(base64_str: str, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64_str}"


# Agent1 - gets the attachment
def agent1_gmail_fetch(state: GraphState) -> dict:
    service = gmail_service()

    response = (
        service.users()
        .messages()
        .list(userId="me", q=QUERY, maxResults=MAX_RESULTS)
        .execute()
    )

    messages = response.get("messages", [])
    print(f"\n[Agent 1] Found {len(messages)} mails with attachment.")

    attachments: Dict[str, List[Dict[str, Any]]] = {}

    for m in messages:
        msg_id = m["id"]
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        payload = msg.get("payload", {})
        headers = {
            h["name"].lower(): h["value"]
            for h in payload.get("headers", [])
            if "name" in h and "value" in h
        }

        subject = headers.get("subject", "(no subject)")
        from_header = headers.get("from", "(unknown from)")

        for part in iter_parts(payload):
            filename = part.get("filename")
            body = part.get("body", {})

            if not filename:
                continue

            gmail_base64url = body.get("data")

            if not gmail_base64url:
                att_id = body.get("attachmentId")
                if att_id:
                    att = (
                        service.users()
                        .messages()
                        .attachments()
                        .get(userId="me", messageId=msg_id, id=att_id)
                        .execute()
                    )
                    gmail_base64url = att.get("data")

            if not gmail_base64url:
                continue

            b64 = gmail_base64url_to_base64(gmail_base64url)
            mime_type = guess_mime_type(filename)
            data_url = to_data_url(b64, mime_type)

            attachments.setdefault(msg_id, []).append(
                {
                    "filename": filename,
                    "mime_type": mime_type,
                    "base64": b64,
                    "data_url": data_url,
                    "subject": subject,
                    "from": from_header,
                }
            )

            print(f"[Agent 1] Captured {filename} ({mime_type})")

    return {
        "attachments": attachments,
        "agent1_response": "Gmail attachments fetched and converted to base64 data URLs",
    }


# Agent2-Sends image to LLM and recieves barcode

def agent2_consume_images(state: GraphState) -> dict:
    attachments = state.get("attachments", {})
    print("\n[Agent 2] Receiving attachments")

    results = []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a barcode decoding specialist. "
                "Your ONLY task is to read barcodes from images and determine the numeric value they encode. "
                "Do not describe the image. Do not explain your reasoning. "
                "Read the gaps between black and white and calculate the number for the barcode.",
            ),
            (
                "user",
                [
                    {
                        "type": "text",
                        "text": "Convert the image provided into greayscale and 100% contrast."
                        "REfine with higher resolution treatment."
                        "Carefully examine the image and focus only on any barcode present.\n"
                        "- Identify the barcode region.\n"
                        "- Read the number that the barcode encodes.\n"
                        "- If human‑readable numeric/alphabetical digits appear below the bars, use them to verify.\n"
                        "- Ignore all other text or graphics.\n\n"
                        "If a barcode number is clearly readable, return it.\n"
                        "If no barcode is visible or the number cannot be read with confidence, return exactly:\n"
                        "NO_BARCODE",
                    },
                    {"type": "image_url", "image_url": {"url": "{image_data_url}"}},
                ],
            ),
        ]
    )

    extracted_serial = ""

    chain = prompt | llm | StrOutputParser()

    for msg_id, items in attachments.items():
        for item in items:
            print(
                f"""
Message ID : {msg_id}
From       : {item.get('from')}
Subject    : {item.get('subject')}
Filename   : {item.get('filename')}
Data URL   : {item.get('data_url', '')[:80]}...
"""
            )

            response = chain.invoke({"image_data_url": item["data_url"]})

            print("\n[Agent 2] Barcode:")
            print(response)

            results.append(
                {
                    "message_id": msg_id,
                    "filename": item.get("filename"),
                    "barcode_result": response,
                }
            )

            if not extracted_serial and response and response != "NO_BARCODE":
                extracted_serial = response

    return {"agent2_response": results, "serial_number": extracted_serial}



def agent3_create_sales_order(state: GraphState) -> dict:
    agent2_results = state.get("agent2_response", [])

    if not agent2_results:
        return {"agent3_response": "No barcode data found from Agent 2"}

    url = os.getenv("SAP_SO_URL")
    user = os.getenv("SAP_USERNAME")
    pwd  = os.getenv("SAP_PASSWORD")

    if not url or not user or not pwd:
        return {"agent3_response": "SAP credentials or URL missing in .env"}

    session = requests.Session()
    session.auth = (user, pwd)
    session.verify = False

    created_orders = []

    for r in agent2_results:
        serial_number = (r.get("barcode_result") or "").strip()

        if not serial_number or serial_number == "NO_BARCODE":
            continue

        print(f"\n[Agent 3] Creating Sales Order for serial: {serial_number}")

        fetch_resp = session.get(
            url,
            headers={
                "Accept": "application/json",
                "X-CSRF-Token": "Fetch"
            }
        )

        csrf_token = fetch_resp.headers.get("X-CSRF-Token")
        if not csrf_token:
            created_orders.append({
                "serial_number": serial_number,
                "status": "FAILED",
                "reason": "CSRF token fetch failed"
            })
            continue

        payload = {
            "SalesOrderType": "OR",
            "SalesOrganization": "1710",
            "DistributionChannel": "10",
            "OrganizationDivision": "00",
            "SoldToParty": "101",
            "PurchaseOrderByCustomer": serial_number,
            "CustomerPaymentTerms": "0001",
            "to_Item": [
                {
                    "Material": "ORDER_BOM",
                    "RequestedQuantity": "2"
                }
            ]
        }

   
        post_resp = session.post(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-CSRF-Token": csrf_token
            },
            data=json.dumps(payload)
        )

        print("[Agent 3] HTTP Status:", post_resp.status_code)
        # print("[Agent 3] Response:", post_resp.text)

        if post_resp.status_code not in (200, 201):
            created_orders.append({
                "serial_number": serial_number,
                "status": "FAILED",
                "http_status": post_resp.status_code
            })
            continue

        try:
            resp_json = post_resp.json()
            so_number = resp_json.get("d", {}).get("SalesOrder", "")
        except Exception:
            so_number = ""

        created_orders.append({
            "serial_number": serial_number,
            "status": "SUCCESS",
            "sales_order_number": so_number
        })

    return {
        "agent3_response": f"{len(created_orders)} Sales Orders processed",
        "sap_raw_response": created_orders,
        "sales_order_number": [
            o.get("sales_order_number")
            for o in created_orders
            if o.get("status") == "SUCCESS"
        ]
    }



def node_print_final(state: GraphState) -> dict:
    print("Agents Status:")
    print("Agent 1:", state.get("agent1_response"))
    print("Agent 2:", state.get("agent2_response"))
    print("Agent 3:", state.get("agent3_response"))
    print("SO No. :", state.get("sales_order_number"))
    return {}


builder = StateGraph(GraphState)

builder.add_node("agent1_gmail_fetch", agent1_gmail_fetch)
builder.add_node("agent2_consume_images", agent2_consume_images)
builder.add_node("agent3_create_sales_order", agent3_create_sales_order)


builder.add_edge(START, "agent1_gmail_fetch")
builder.add_edge("agent1_gmail_fetch", "agent2_consume_images")
builder.add_edge("agent2_consume_images", "agent3_create_sales_order")
builder.add_edge("agent3_create_sales_order", "print_final")
builder.add_node("print_final", node_print_final)
builder.add_edge("print_final", END)

app = builder.compile()


app.invoke({})
