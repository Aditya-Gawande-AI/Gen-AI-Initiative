from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
import urllib3
 
# Disable SSL warnings (DEV system)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
# LangChain / SAP GenAI Hub
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
# Optional MongoDB
try:
    from pymongo import MongoClient
    mongo_available = True
except ImportError:
    mongo_available = False
 
# Load environment variables
load_dotenv()
 
# -------------------- LLM SETUP --------------------
llm = init_llm("gpt-4o")
 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful SAP assistant."),
    ("user", "{input}")
])
 
chain = prompt | llm | StrOutputParser()
 
# -------------------- MONGODB SETUP --------------------
collection = None
 
if mongo_available:
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client["sap_ai"]
        collection = db["sales_order_agents"]
        print(" MongoDB Connected")
    except Exception:
        print("⚠ MongoDB not running. Output will be printed only.")
        collection = None
 
# =====================================================
# AGENT 1 – LLM knowledge from SAP API Business Hub
# =====================================================
print("\n Running Agent 1...")
 
agent1_query = """
How to retrieve top sales orders using API_SALES_ORDER_SRV/A_SalesOrder
service in SAP API Business Hub? Explain step-by-step.
"""
 
agent1_response = chain.invoke({"input": agent1_query})
 
print("\nAgent 1 Output:\n", agent1_response)
 
if collection:
    collection.insert_one({
        "agent": "agent1",
        "description": "How to view top sales orders using SAP API",
        "data": agent1_response
    })
 
# =====================================================
# AGENT 2 – SAP API Call
# =====================================================
print("\n  Running Agent 2...")
 
url = "https://172.19.151.9:44302/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10"
 
try:
    response = requests.get(
        url,
        auth=HTTPBasicAuth("Developer", "dev@S09"),
        headers={"Accept": "application/json"},
        verify=False
    )
 
    response.raise_for_status()
    sales_data = response.json()
 
    print("\nAgent 2 Output:\n", sales_data)
 
    if collection:
        collection.insert_one({
            "agent": "agent2",
            "description": "Top 10 Sales Orders from SAP",
            "data": sales_data
        })
 
except Exception as e:
    print(" SAP API Error:", e)
    sales_data = None
 
# =====================================================
# AGENT 3 – LLM Explanation of Single Sales Order
# =====================================================
print("\n Running Agent 3...")
 
if sales_data:
    try:
        sales_orders = sales_data["d"]["results"]
        sample_order = sales_orders[0]
 
        agent3_query = f"""
        Explain the following SAP Sales Order in simple business terms:
        {sample_order}
        """
 
        agent3_response = chain.invoke({"input": agent3_query})
 
        print("\nAgent 3 Output:\n", agent3_response)
 
        if collection:
            collection.insert_one({
                "agent": "agent3",
                "description": "Explanation of Sales Order",
                "data": agent3_response
            })
 
    except Exception as e:
        print(" Agent 3 Error:", e)
else:
    print(" No sales order data available for Agent 3")
 
print("\n Task Completed Successfully")