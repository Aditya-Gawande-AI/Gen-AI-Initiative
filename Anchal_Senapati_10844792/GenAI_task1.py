from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
# LOAD ENV
load_dotenv()
 
# INITIALIZE LLM
model = init_llm("gpt-4o")
 
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
 
chain = template | model | StrOutputParser()
 
 
# AGENT 1 (LLM INFO)
def agent_1():
    print("\nRunning Agent 1...")
 
    query = """How to retrieve top sales orders using API_SALES_ORDER_SRV/A_SalesOrder service on SAP API Business Hub?"""
 
    result = chain.invoke({"input": query})
 
    print("\nAgent 1 Output:\n", result)
 
    return result


# AGENT 2 (SAP API CALL)
def agent_2():
    print("\nRunning Agent 2...")
 
    url = os.getenv("CALL_API")
    username = os.getenv("USER")
    password = os.getenv("PASSWORD")
 
    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"},
            verify=False
        )
 
        print("Status Code:", response.status_code)
 
        if response.status_code != 200:
            print("API Error:", response.text)
            return None
        
        else:
            data = response.json()
            orders = data.get("d",{}).get("results",[])[:10]
 
        print("\nAgent 2 Output:\n", orders)
        return orders
 
    except Exception as e:
        print("API Exception:", e)
        print("Using mock data due to connection issue...")

        orders = [
    {
        "SalesOrder": "500001",
        "CustomerName": "ABC Corporation",
        "NetAmount": "10000",
        "Currency": "INR",
        "CreationDate": "2024-03-01"
    },
    {
        "SalesOrder": "500002",
        "CustomerName": "XYZ Pvt Ltd",
        "NetAmount": "15000",
        "Currency": "INR",
        "CreationDate": "2024-03-02"
    },
    {
        "SalesOrder": "500003",
        "CustomerName": "MNO Industries",
        "NetAmount": "75000",
        "Currency": "INR",
        "CreationDate": "2024-03-03"
    },
    {
        "SalesOrder": "500004",
        "CustomerName": "PQR Enterprises",
        "NetAmount": "35000",
        "Currency": "INR",
        "CreationDate": "2024-03-04"
    },
    {
        "SalesOrder": "500005",
        "CustomerName": "DEF Solutions",
        "NetAmount": "80000",
        "Currency": "INR",
        "CreationDate": "2024-03-05"
    }
]
 
        return orders
 

# AGENT 3 (LLM EXPLANATION)
def agent_3(order):
    print("\nRunning Agent 3...")
 
    query = f"Explain this SAP sales order in simple terms: {order}"
 
    result = chain.invoke({"input": query})
 
    print("\nAgent 3 Output:\n", result)
 
    return result

# MAIN FLOW
if __name__ == "__main__":
 
    print("\nStarting Task...\n")
 
    agent1_result = agent_1()
 
    orders = agent_2()
 
    if orders:
        agent3_result = agent_3(orders[0])
    else:
        print("No data available for Agent 3")
 
    print("\nTask Completed ✅")