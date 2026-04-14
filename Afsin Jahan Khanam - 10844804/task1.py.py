
from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
#langChain / LLM
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
#Optional MongoDB
try:
    from pymongo import MongoClient
    mongo_available = True
except:
    mongo_available = False
 
#Load environment variables
load_dotenv()
 
#initialize LLM
model = init_llm("gpt-4o")
 
template = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant."),
    ("user","{input}")
    ])
 
chain = template | model | StrOutputParser()
 
# MongoDB Setup (Optional)
collection = None
 
if mongo_available:
    try:
        client = MongoClient("mongodb://localhost:27017/",serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client["sap_ai"]
        collection = db["results"]
        print("MongoDB Connected")
    except:
        print("MongoDB not running(Skipping DB Storage)")
        collection = None
 
# Agent 1 -> LLM(SAP API Info)
print("\n Running Agent 1....")
 
query = """How to retrieve top sales orders using API_SALES_ORDER_SRV/A_SalesOrder service in SAP API Business Hub? Explain step-by-step."""
 
agent1_response = chain.invoke({"input":query})
 
print("\nAgent 1 Output:\n",agent1_response)
 
if collection is not None:
    collection.insert_one({"agent": "agent1", "data": agent1_response})
 
# AGENT 2 -> SAP API Call
print("\n Running Agent 2...")
 
url = "xxx"
username = "xxx"
password = "xxx"
 
try:
    response = requests.get(
        url,
        auth=HTTPBasicAuth(username, password),
        headers={"Accept":"application/json"},
        verify=False
        )
   
    # print("Status Code:",response.status_code)
    # print("Raw Response:\n",response.text[:500])
 
    data = response.json()
 
    print("\nAgent 2 Output:\n", data)
 
    if collection is not None:
        collection.insert_one({"agent":"agent2", "data":data})
 
except Exception as e:
    print("API Error:",e)
    data = None
 
# AGENT 3 -> Explain Sales Order
print("\n Running Agent 3...")
 
if data:
    try:
        sales_orders = data["d"]["results"]
        first_order = sales_orders[0]
 
        explanation_query = f"""Explain this SAP sales order in simple terms: {first_order}"""
 
        agent3_response = chain.invoke({"input":explanation_query})
       
        print("\nAgent 3 Output:\n",agent3_response)
 
        if collection is not None:
            collection.insert_one({"agent":"agent3", "data":agent3_response})
   
    except Exception as e:
        print("Agent 3 Error:",e)
 
else:
    print("No data acailable for Agent 3")
 
print("\n Task Completed")
