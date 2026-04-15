import os, requests, base64, pickle, re, json, urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired

# LangGraph + LLM
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langgraph.graph import StateGraph, START, END

# Gmail API
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ============================================================
# Environment Setup
# ============================================================
load_dotenv()

# ============================================================
# LLM Configuration
# ============================================================
llm = init_llm("gpt-4o", temperature=0.3, max_tokens=1500)

# ============================================================
# Graph State Definition
# ============================================================
class GraphState(TypedDict, total=False):
    image_path: NotRequired[str]
    serial_number: NotRequired[str]
    sap_response: NotRequired[str]

# ============================================================
# Gmail Authentication
# ============================================================
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None

    if os.path.exists('token.pkl'):
        with open('token.pkl', 'rb') as token_file:
            creds = pickle.load(token_file)

    if not creds:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = pickle.load("token")
        with open('token.pkl', 'wb') as token_file:
            pickle.dump(creds, token_file)

    gmail_service = build('gmail', 'v1', credentials=creds)
    return gmail_service

# ============================================================
# Attachment Downloader (Recursive)
# ============================================================
def get_attachments(service, msg_id, parts):
    for part in parts:
        attachment_filename = part.get('filename')
        part_body = part.get('body', {})

        if attachment_filename:
            if 'attachmentId' in part_body:
                attachment_id = part_body['attachmentId']
                attachment_obj = service.users().messages().attachments().get(
                    userId='me',
                    messageId=msg_id,
                    id=attachment_id
                ).execute()

                attachment_bytes = base64.urlsafe_b64decode(attachment_obj['data'])
                saved_file_path = os.path.join(os.getcwd(), attachment_filename)

                with open(saved_file_path, 'wb') as file_handle:
                    file_handle.write(attachment_bytes)

                print(f" Attachment saved: {attachment_filename}")
                return saved_file_path

        # Nested / multipart support (recursive)
        if 'parts' in part:
            nested_result = get_attachments(service, msg_id, part['parts'])
            if nested_result:
                return nested_result

    return None

# ============================================================
# AGENT 1: Read Email + Fetch Attachment
# ============================================================
def node_agent1_read_email(state: GraphState):
    print("\n" + "=" * 60)
    print(" AGENT 1 | Reading email and looking for attachment...")
    print("=" * 60)

    gmail_service = authenticate_gmail()
    list_response = gmail_service.users().messages().list(
        userId='me',
        maxResults=5
    ).execute()

    message_list = list_response.get('messages', [])

    for msg_meta in message_list:
        message_id = msg_meta['id']

        message_obj = gmail_service.users().messages().get(
            userId='me',
            id=message_id
        ).execute()

        payload = message_obj['payload']
        headers = payload['headers']

        email_subject = ""
        for header in headers:
            if header['name'] == 'Subject':
                email_subject = header['value']

        if "Product Serial Number" not in email_subject:
            continue

        print(" Target email found (Subject matched).")

        if 'parts' in payload:
            attachment_path = get_attachments(gmail_service, message_id, payload['parts'])

            if attachment_path:
                print(f" Image fetched at: {attachment_path}")
                return {"image_path": attachment_path}

    raise Exception("No Matching Email with attachment found")

# ============================================================
# AGENT 2: Extract Serial Number from Image
# ============================================================
def node_agent2_extract_serial(state: GraphState):
    print("\n" + "=" * 60)
    print(" AGENT 2 | Extracting serial number from image...")
    print("=" * 60)

    image_path = state["image_path"]

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    llm_response = llm.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract only the serial number from this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ])

    extracted_serial = re.findall(r'[A-Z0-9]+', str(llm_response))[0]

    print(f" Extracted Serial Number: {extracted_serial}")

    return {"serial_number": extracted_serial}

# ============================================================
# AGENT 3: Create SAP Sales Order
# ============================================================
def node_agent3_create_order(state: GraphState):
    print("\n" + "=" * 60)
    print(" AGENT 3 | Creating SAP Sales Order...")
    print("=" * 60)

    serial_number = state["serial_number"]
    sap_post_url = os.getenv("SAP_POST_URL")
    sap_username = os.getenv("SAP_USERNAME")
    sap_password = os.getenv("SAP_PASSWORD")

    try:
        http_session = requests.Session()

        token_response = http_session.get(
            sap_post_url,
            auth=(sap_username, sap_password),
            headers={"X-CSRF-Token": "Fetch"},
            verify=False
        )

        csrf_token = token_response.headers.get("X-CSRF-Token")
        print(f" CSRF Token fetched: {csrf_token}")

        sales_order_payload = {
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

        post_response = http_session.post(
            sap_post_url,
            auth=(sap_username, sap_password),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-CSRF-Token": csrf_token
            },
            json=sales_order_payload,
            verify=False,
        )

        sap_result_text = post_response.text
    except Exception as exc:
        sap_result_text = str(exc)

    print("\n SAP Response:")
    print("-" * 60)
    # print(sap_result_text)
    # print("-" * 60)

    return {"sap_response": sap_result_text}

# ============================================================
# FINAL OUTPUT NODE
# ============================================================
def node_print(state: GraphState):
    print("\n" + "=" * 60)
    print(" FINAL RESULT")
    print("=" * 60)

    try:
        response_json = json.loads(state["sap_response"])
        sales_order_id = response_json.get("d", {}).get("SalesOrder")

        print(" Sales Order Created Successfully!")
        print(f" Order ID      : {sales_order_id}")
        print(f" Serial Number : {state.get('serial_number')}")

    except:
        print(" Unable to parse JSON. Raw response below:")
        print(state["sap_response"])

    return {}

# ============================================================
# Graph Wiring
# ============================================================
builder_node = StateGraph(GraphState)

builder_node.add_node("agent1", node_agent1_read_email)
builder_node.add_node("agent2", node_agent2_extract_serial)
builder_node.add_node("agent3", node_agent3_create_order)
builder_node.add_node("print", node_print)

builder_node.add_edge(START, "agent1")
builder_node.add_edge("agent1", "agent2")
builder_node.add_edge("agent2", "agent3")
builder_node.add_edge("agent3", "print")
builder_node.add_edge("print", END)

app = builder_node.compile()

# ============================================================
# Run
# ============================================================
app.invoke({})
