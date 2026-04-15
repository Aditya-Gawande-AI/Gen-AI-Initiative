import os
import re
import json
import base64
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Gmail libraries
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# SETTINGS
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_QUERY = "newer_than:30d"
MAX_MESSAGES = 10


# STEP 1: CONNECT TO GMAIL (login and create gmail service)
def connect_gmail(credentials_path="credentials.json", token_path="token.json"):
    creds = None

    # If token.json already exists, reuse it
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)

        # Save login for next time
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    # Create Gmail API service object
    service = build("gmail", "v1", credentials=creds)
    return service


# STEP 2: Get the first image from recent emails
def get_first_image_from_gmail(service):
    # Get list of recent emails
    res = service.users().messages().list(
        userId="me", q=GMAIL_QUERY, maxResults=MAX_MESSAGES
    ).execute()

    messages = res.get("messages", [])
    if not messages:
        raise RuntimeError("No emails found in last 30 days.")

    # Loop emails one by one
    for m in messages:
        msg_id = m["id"]

        # Get full email details (including attachments info)
        msg = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        payload = msg.get("payload", {}) or {}

        # Email can have multiple nested parts (attachments/text)
        parts_to_check = [payload]

        while parts_to_check:
            part = parts_to_check.pop()
            parts_to_check.extend(part.get("parts", []) or [])

            mime_type = (part.get("mimeType") or "").lower()

            # Check the image type
            if not mime_type.startswith("image/"):
                continue

            body = part.get("body", {}) or {}
            attachment_id = body.get("attachmentId")
            data = None

            # If image is an attachment -> download using attachmentId
            if attachment_id:
                att = service.users().messages().attachments().get(
                    userId="me", messageId=msg_id, id=attachment_id
                ).execute()
                data = att.get("data")
            else:
                # Sometimes image is inline
                data = body.get("data")

            if data:
                print("Found an image in Gmail.")
                image_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
                return mime_type, image_bytes

    raise RuntimeError("No image found in last 30 days emails.")


# STEP 3: Use LLM to extract Serial Number
def extract_serial_number(image_bytes, mime_type, model_name="gpt-4o"):
    # SAP GenAI Hub OpenAI proxy
    from gen_ai_hub.proxy.native.openai import chat  # type: ignore

    # Convert image bytes into data URL format
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"

    # Message to AI
    messages = [
        {"role": "system", "content": "Extract ONLY the SERIAL NUMBER from the image. Return only the serial number."},
        {"role": "user", "content": [
            {"type": "text", "text": "Return serial number only."},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ]

    # Call the model
    resp = chat.completions.create(
        model_name=model_name,
        messages=messages,
        temperature=0
    )

    # Read LLM text output
    text = resp.choices[0].message.content.strip()

    # Try to detect serial using regex pattern
    match = re.search(r"[A-Z0-9][A-Z0-9\-_\/]{4,}", text, flags=re.IGNORECASE)

    serial = match.group(0) if match else text
    return serial


# STEP 4: Create Sales Order in SAP (CSRF + session)
def create_sales_order_in_sap(serial):
    base_url = os.getenv("SAP_BASE_URL", "").rstrip("/")
    username = os.getenv("SAP_USERNAME", "")
    password = os.getenv("SAP_PASSWORD", "")
    csrf_token = os.getenv("SAP_CSRF_TOKEN", "")

    if not (base_url and username and password and csrf_token):
        raise RuntimeError("Missing SAP_BASE_URL / SAP_USERNAME / SAP_PASSWORD / SAP_CSRF_TOKEN in .env")

    url = f"{base_url}/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder"
    auth = HTTPBasicAuth(username, password)

    # Sales order payload
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

    # Use Session to keep cookies + token
    s = requests.Session()
    s.verify = False

    # Step A: Create session cookies
    s.get(url, headers={"Accept": "application/json"}, auth=auth)

    # Helper: POST with token
    def post_so(token):
        return s.post(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-CSRF-Token": token,
            },
            auth=auth,
            json=payload,
        )

    # Step B: Try POST with provided CSRF token
    r = post_so(csrf_token)

    # If token expired -> 403 -> fetch new token and retry
    if r.status_code == 403:
        print("CSRF token expired. Fetching new token and retrying...")

        t = s.get(url, headers={"X-CSRF-Token": "Fetch", "Accept": "application/json"}, auth=auth)
        t.raise_for_status()

        new_token = t.headers.get("X-CSRF-Token") or t.headers.get("x-csrf-token")
        if not new_token:
            raise RuntimeError("CSRF fetch worked, but token not found in headers.")

        r = post_so(new_token)

    r.raise_for_status()
    return r.json()


# STEP 5: Extract sales order number from response
def get_sales_order_number(resp_json):
    d = resp_json.get("d", {})
    if isinstance(d, dict):
        for key in ("SalesOrder", "VBELN", "Vbeln", "SalesOrderID", "SalesOrderId"):
            if d.get(key):
                return str(d[key])
    return None


# MAIN PROGRAM (runs everything in order)
def main():
    load_dotenv(override=True)

    print("\n--------Agent-1------------\n")

    # Connect Gmail
    gmail = connect_gmail()
    print("Gmail connected.")

    # Get first image
    mime, img_bytes = get_first_image_from_gmail(gmail)
    print(f"Image downloaded. Type: {mime}")

    # Extract serial no
    print("\n-----------Agent-2------------\n")
    serial = extract_serial_number(img_bytes, mime)
    print(f"SERIAL NUMBER: {serial}")

    # Create sales order in SAP
    print("\n------------Agent-3-----------\n")
    print("\nCreating Sales Order in SAP...")
    resp = create_sales_order_in_sap(serial)

    # Print Sales Order number
    so_num = get_sales_order_number(resp)
    print(f"SALES ORDER: {so_num if so_num else '(Not found in response)'}")

    # Print full SAP response
    print("\nFull SAP Response JSON:")
    print(json.dumps(resp, indent=2))


if __name__ == "__main__":
    main()