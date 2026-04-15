import os
import re
import requests
import urllib3
from dotenv import load_dotenv
from simplegmail import Gmail
from simplegmail.query import construct_query
os.environ.pop("GOOGLE_API_KEY", None)
from google import genai
from google.genai import types
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph


# -------------------- STATE --------------------
class AgentState(TypedDict):
    image_paths: List[str]
    serial_number: Optional[str]
    sap_response: Optional[dict]


# -------------------- NODE 1: GMAIL --------------------
def gmail_agent(state: AgentState):
    gmail = Gmail()
    messages = gmail.get_messages(query=construct_query(sender="Shambo.Ghosh@ltm.com"))

    SAVE_DIR = "images_task_3"
    os.makedirs(SAVE_DIR, exist_ok=True)

    image_paths = []

    for message in messages:
        for att in message.attachments:
            if att.filetype and att.filetype.startswith("image/"):
                if (att.filename or "").lower().startswith("outlook-"):
                    continue
                out_path = os.path.join(SAVE_DIR, att.filename)
                att.save(out_path, overwrite=True)
                image_paths.append(out_path)
                print("Saved:", out_path)

    state["image_paths"] = image_paths
    return state


# -------------------- NODE 2: GEMINI VISION --------------------
def gemini_vision_agent(state: AgentState):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for path in state["image_paths"]:
        uploaded = client.files.upload(file=path)

        resp = client.models.generate_content(
                model="models/gemini-flash-latest",
                contents=["Extract serial number and write a line that Serial number of product is :", uploaded]
                )


        print(resp.text)

        m = re.search(r"(\d{6,})", resp.text)
        if m:
            state["serial_number"] = m.group(1)
            return state

    raise SystemExit("Serial number not found from images.")


# -------------------- NODE 3: SAP CREATE SALES ORDER --------------------
def sap_agent(state: AgentState):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    SAP_USERNAME = os.getenv("SAP_USERNAME")
    SAP_PASSWORD = os.getenv("SAP_PASSWORD")
    SAP_BASE_URL = os.getenv("SAP_BASE_URL")

    url = f"{SAP_BASE_URL}"

    session = requests.Session()
    session.auth = (SAP_USERNAME, SAP_PASSWORD)
    session.verify = False

    token_resp = session.head(url, headers={"X-CSRF-Token": "Fetch"})
    csrf_token = token_resp.headers.get("x-csrf-token") or token_resp.headers.get("X-CSRF-Token")

    if not csrf_token:
        raise SystemExit("Could not fetch CSRF token.")

    payload = {
        "SalesOrderType": "OR",
        "SalesOrganization": "1710",
        "DistributionChannel": "10",
        "OrganizationDivision": "00",
        "SoldToParty": "101",
        "PurchaseOrderByCustomer": state["serial_number"],
        "CustomerPaymentTerms": "0001",
        "to_Item": [{"Material": "ORDER_BOM", "RequestedQuantity": "2"}],
    }
    resp = session.post(
        url,
        json=payload,
        headers={
            "X-CSRF-Token": csrf_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    

    #print("SAP Status:", resp.status_code)
    #print(resp.json())

    if resp.status_code not in (200, 201):
        raise SystemExit(resp.text)

    state["sap_response"] = resp.json()
    return state


# -------------------- NODE 4: GEMINI FORMATTER --------------------
def gemini_formatter_agent(state: AgentState):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    resp = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents=[
            "From this SAP API response, return in few lines only the parameters that have some value and not blank:\n"
            "SalesOrder: <value>\n"
            "PurchaseOrderByCustomer: <value>\n\n"
            "Every parameter should be in new line"
            f"{state['sap_response']}"
        ],
    )

    print("\nFINAL OUTPUT:")
    print(resp.text)
    return state


# -------------------- GRAPH --------------------
load_dotenv()

graph = StateGraph(AgentState)

graph.add_node("gmail", gmail_agent)
graph.add_node("vision", gemini_vision_agent)
graph.add_node("sap", sap_agent)
graph.add_node("final", gemini_formatter_agent)

graph.set_entry_point("gmail")
graph.add_edge("gmail", "vision")
graph.add_edge("vision", "sap")
graph.add_edge("sap", "final")

app = graph.compile()

initial_state: AgentState = {
    "image_paths": [],
    "serial_number": None,
    "sap_response": None,
}

app.invoke(initial_state)