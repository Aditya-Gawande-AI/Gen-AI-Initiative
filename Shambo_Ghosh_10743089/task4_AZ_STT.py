import os, re, json, requests, urllib3
from typing import TypedDict, Optional, Dict, Any
from dotenv import load_dotenv
from google import genai
from requests.auth import HTTPBasicAuth
from langgraph.graph import StateGraph, END
load_dotenv()
import azure.cognitiveservices.speech as speechsdk
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SPEECH_KEY = os.getenv("SPEECH_KEY", "")
SPEECH_ENDPOINT = os.getenv("SPEECH_ENDPOINT", "")
SPEECH_LANG = os.getenv("SPEECH_LANG", "en-US")
SAP_URL = os.getenv("SAP_BASE_URL", "").strip()
SAP_USER = os.getenv("SAP_USERNAME", "").strip()
SAP_PASS = os.getenv("SAP_PASSWORD", "").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "").strip()

class State(TypedDict):
    text: Optional[str]
    should_post: Optional[bool]
    payload: Optional[Dict[str, Any]]
    sap_result: Optional[Dict[str, Any]]
    gemini_result: Optional[Dict[str, Any]]
    error: Optional[str]

def stt_node(s: State) -> State:
    s["error"] = None
    if not SPEECH_KEY or not SPEECH_ENDPOINT:
        s["text"] = input("Type: ").strip()
        return s
    cfg = speechsdk.SpeechConfig(subscription=SPEECH_KEY, endpoint=SPEECH_ENDPOINT)
    cfg.speech_recognition_language = SPEECH_LANG
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=speechsdk.audio.AudioConfig(use_default_microphone=True))
    print("Say: create sales order type OR sales org 1710 distribution channel 10")
    res = rec.recognize_once()
    s["text"] = (res.text or "").strip()
    return s

def decide_node(s: State) -> State:
    t = (s.get("text") or "").lower()
    s["should_post"] = ("create" in t and "order" in t)
    return s

def build_payload_node(s: State) -> State:
    t = (s.get("text") or "").lower()
    so_type, sales_org, dist_ch = "OR", "1710", "10"
    m = re.search(r"(?:sales\s*)?order\s*type\s*([a-z0-9]{2,4})|\btype\s*([a-z0-9]{2,4})\b", t)
    if m:
        so_type = (m.group(1) or m.group(2)).upper()
    m = re.search(r"\bsales\s*org(?:anization)?\s*([0-9]{3,6})\b", t)
    if m:
        sales_org = m.group(1)
    m = re.search(r"\b(?:distribution|dist)\s*channel\s*([0-9]{1,2})\b|\bchannel\s*([0-9]{1,2})\b", t)
    if m:
        dist_ch = (m.group(1) or m.group(2)).zfill(2)
    m = re.search(r"\b([a-z]{2,4})\s+([0-9]{3,6})\s+([0-9]{1,2})\b", t)
    if m:
        so_type, sales_org, dist_ch = m.group(1).upper(), m.group(2), m.group(3).zfill(2)
    s["payload"] = {
        "SalesOrderType": so_type,
        "SalesOrganization": sales_org,
        "DistributionChannel": dist_ch,
        "OrganizationDivision": "00",
        "SoldToParty": "101",
        "PurchaseOrderByCustomer": "123456",
        "CustomerPaymentTerms": "0001",
        "to_Item": [{"Material": "ORDER_BOM", "RequestedQuantity": "2"}]
    }
    return s

def sap_post_node(s: State) -> State:
    s["error"] = None
    s["sap_result"] = None
    if not s.get("should_post"):
        s["sap_result"] = {"status": None, "body": "No create order intent"}
        return s
    if not SAP_URL or not SAP_USER or not SAP_PASS:
        s["error"] = "Missing SAP env vars"
        return s
    session = requests.Session()
    session.auth = HTTPBasicAuth(SAP_USER, SAP_PASS)
    session.verify = False
    r = session.get(SAP_URL, headers={"X-CSRF-Token": "Fetch", "Accept": "application/json"}, timeout=30)
    token = r.headers.get("X-CSRF-Token") or r.headers.get("x-csrf-token")
    if not token:
        s["error"] = "CSRF token not returned by SAP"
        return s
    r = session.post(
        SAP_URL,
        headers={"X-CSRF-Token": token, "Content-Type": "application/json", "Accept": "application/json"},
        json=s["payload"],
        timeout=30,
    )
    try:
        body = r.json() if r.text else {}
    except Exception:
        body = {"raw": r.text}
    s["sap_result"] = {"status": r.status_code, "body": body}
    return s

def gemini_node(s: State) -> State:
    s["gemini_result"] = None
    if not GEMINI_API_KEY or not GEMINI_MODEL:
        s["gemini_result"] = {"success": False, "message": "Missing GEMINI_API_KEY or GEMINI_MODEL", "details": "Set both env vars to enable Gemini consolidation"}
        return s
    sap_result = s.get("sap_result") or {}
    prompt = (
        "You are an SAP SD expert.\n"
        "Analyze the SAP HTTP response below and tell clearly:\n"
        "1. Whether the sales order was created or not\n"
        "2. If created, mention the sales order number (if present)\n"
        "3. If failed, explain the reason in simple terms\n"
        "4. Give a short final status message suitable for end users\n"
        "Do not guess. Use only what is present in the response.\n\n"
        "SAP Response:\n"
        f"{json.dumps(sap_result, indent=2)}\n\n"
        "Return the response in simple lines not in JSON Format:\n"
        
    )
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    s["gemini_result"] = (resp.text or "").replace("\\n", "\n").strip()
    return s

g = StateGraph(State)
g.add_node("stt", stt_node)
g.add_node("decide", decide_node)
g.add_node("build", build_payload_node)
g.add_node("post", sap_post_node)
g.add_node("gemini", gemini_node)

g.set_entry_point("stt")
g.add_edge("stt", "decide")
g.add_edge("decide", "build")
g.add_edge("build", "post")
g.add_edge("post", "gemini")
g.add_edge("gemini", END)

app = g.compile()

if __name__ == "__main__":
    out = app.invoke({"text": None, "should_post": None, "payload": None, "sap_result": None, "gemini_result": None, "error": None})
    print("\n--- TEXT ---\n", out.get("text"))
    print("\n--- POST? ---\n", out.get("should_post"))
    print("\n--- PAYLOAD ---\n", json.dumps(out.get("payload"), indent=2))
    print("\n--- SAP RESULT ---\n", json.dumps(out.get("sap_result"), indent=2))
    print("\n--- GEMINI RESULT ---\n", out.get("gemini_result"))
    if out.get("error"):
        print("\n--- ERROR ---\n", out["error"])


