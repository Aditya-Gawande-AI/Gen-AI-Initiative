import os
import requests
import base64
import pickle
import re
import json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
from dotenv import load_dotenv
from typing_extensions import TypedDict, NotRequired
 
# LangGraph + LLM
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langgraph.graph import StateGraph, START, END
 
# Gmail API
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
 
#------------------------ ENV --------------------------
load_dotenv()
 
#------------------------ LLM --------------------------
llm = init_llm("gpt-4o", temperature=0.3, max_tokens=1500)

#------------------------ STATE --------------------------
class GraphState(TypedDict, total=False):
    image_path: NotRequired[str]
    serial_number: NotRequired[str]
    sap_response: NotRequired[str]
 
#------------------------ GMAIL AUTH --------------------------
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
 
def authenticate_gamil():
    creds = None
 
    if os.path.exists('token.pkl'):
        with open('token.pkl','rb') as token:
            creds = pickle.load(token)
    if not creds:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json',SCOPES)
        creds = pickle.load(token)
        with open('token.pkl','wb') as token:
            pickle.dump(creds,token)
   
    service = build('gmail','v1',credentials=creds)
    return service
 
#------------------------ GET ATTACHMENT --------------------------
def get_attachments(service, msg_id, parts):
    for part in parts:
        filename = part.get('filename')
        body = part.get('body', {})
 
        if filename:
            if 'attachmentId' in body:
                attachment_id = body['attachmentId']
                attachment = service.users().messages().attachments().get(
                    userId='me',
                    messageId=msg_id,
                    id=attachment_id
                ).execute()
 
                file_data = base64.urlsafe_b64decode(attachment['data'])
                file_path = os.path.join(os.getcwd(), filename)
 
                with open(file_path,'wb') as f:
                    f.write(file_data)
               
                print(f" - Attachment saved: {filename}")
                return file_path
       
        # Recursive check
        if 'parts' in part:
            result = get_attachments(service, msg_id, part['parts'])
            if result:
                return result
   
    return None
 
#------------------------ AGENT 1 --------------------------
def node_agent1_read_email(state: GraphState):
    print("\nAgent 1: Reading email and attachment.....")
 
    service = authenticate_gamil()
    results = service.users().messages().list(
        userId='me',
        maxResults=5
    ).execute()
 
    messages = results.get('messages',[])
 
    for msg_meta in messages:
        msg_id = msg_meta['id']
 
        msg = service.users().messages().get(
            userId='me',
            id=msg_id
        ).execute()

        payload = msg['payload']
        headers = payload['headers']
 
        subject = ""
 
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
           
        if "Product Serial Number" not in subject:
            continue
 
        print(" - Target Email Found")
 
        if'parts' in payload:
            image_path = get_attachments(service, msg_id, payload['parts'])
 
            if image_path:
                print(" - Image fetched:", image_path)
                return {"image_path": image_path}
           
    raise Exception("No Matching Email with attachment found")
 
#----------------------- AGENT 2 --------------------------
def node_agent2_extract_serial(state: GraphState):
    print("\nAgent 2: Extracting serial number.......")
 
    image_path = state["image_path"]
 
    with open(image_path, "rb") as img:
        image_bytes = img.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
   
    response = llm.invoke([
        {
            "role":"user",
            "content":[
                {"type":"text", "text":"Extract only the serial number from this image."},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ])
 
    serial_number = re.findall(r'[A-Z0-9]+',str(response))[0]
 
    print(" - Extracted Serial Number:",serial_number)
 
    return {"serial_number":serial_number}
 
#----------------------- AGENT 3 --------------------------
def node_agent3_create_order(state: GraphState):
    print("\nAgent 3: Creating SAP Sales Order.........")
 
    serial = state["serial_number"]
    url = os.getenv("SAP_POST_URL")
    username = os.getenv("SAP_USERNAME")
    password = os.getenv("SAP_PASSWORD")
 
    try:
        session = requests.Session()
 
        token_response = session.get(
            url,
            auth=(username, password),
            headers={"X-CSRF-Token":"Fetch"},
            verify=False
        )
 
        csrf_token = token_response.headers.get("X-CSRF-Token")
 
        print(" - CSRF Token fetched:",csrf_token)
 
        payload = {
            "SalesOrderType":"OR",
            "SalesOrganization":"1710",
            "DistributionChannel":"10",
            "OrganizationDivision":"00",
            "SoldToParty":"101",
            "PurchaseOrderByCustomer":serial,
            "CustomerPaymentTerms":"0001",
            "to_Item":[
                {
                    "Material":"ORDER_BOM",
                    "RequestedQuantity":"2"
                }
            ]
        }
 
        response = session.post(
            url,
            auth=(username, password),
            headers={
                "Content-Type":"application/json",
                "Accept":"application/json",
                "X-CSRF-Token":csrf_token
            },
            json=payload,
            verify=False,
        )
        
        result = response.text
    except Exception as e:
        result = str(e)
   
    print(" - SAP Response:\n",result)
    return {"sap_response":result}
 
#----------------------- FINAL PRINT --------------------------
def node_print(state: GraphState):
    print("\nFINAL RESULT:")
 
    try:
        data = json.loads(state["sap_response"])
        sales_order = data.get("d",{}).get("SalesOrder")
 
        print(" - Sales Order Created Successfully!")
        print(" - Order ID:",sales_order)
        print(" - Serial Number:",state.get("serial_number"))
   
    except:
        print(state["sap_response"])
 
    return {}

#----------------------- GRAPH --------------------------
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
 
#----------------------- RUN --------------------------
app.invoke({})
 




 