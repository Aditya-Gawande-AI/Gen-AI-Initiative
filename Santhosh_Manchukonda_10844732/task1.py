import os
import requests
import json
import urllib3
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.messages import HumanMessage

load_dotenv()

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION (SAP AI Hub) ---
llm = init_llm('gpt-4o', max_tokens=1000)

def print_step(title):
    print(f"\n{'='*20} {title} {'='*20}")

# --- AGENT 1: THE RESEARCHER ---
def run_agent_1():
    print_step("------------AGENT 1----------------")
    prompt = "Explain briefly how to view top sales orders using API_SALES_ORDER_SRV/A_SalesOrder on SAP API Business Hub."
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"RESEARCH SUMMARY:\n{response.content}")
    return response.content

# --- AGENT 2: THE DATA FETCHER ---
def run_agent_2():
    print_step("-----------AGENT 2------------------")
    
    url = "https://172.19.151.9:44302/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10"

    
    print(f"DEBUG: Connecting to {url}")
    
    try:
        response = requests.get(
            url, 
            auth=("Developer", "dev@S09"),
            verify=False,
            timeout=15,
            headers={"Accept": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            orders = data.get('d', {}).get('results', [])
            print(f"SUCCESS: Retrieved {len(orders)} orders.")
            
            if orders:
                print("\nSAMPLE DATA (Order #1):")
                print(json.dumps(orders[0], indent=2))
            return orders
        else:
            print(f"FAILED: SAP Status {response.status_code}")
            return None
    except Exception as e:
        print(f"CONNECTION ERROR: {e}")
        return None

# --- AGENT 3: THE ANALYST ---
def run_agent_3(orders):
    print_step("---------------AGENT 3-------------------")
    
    # Analyze the first order from the list
    sample_order = orders[0]
    
    prompt = f"Explain the details of this SAP Sales Order in simple business terms: {json.dumps(sample_order)}"
    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"BUSINESS EXPLANATION:\n{response.content}")

# --- EXECUTION ---
if __name__ == "__main__":
    run_agent_1()
    data = run_agent_2()
    if data:
        run_agent_3(data)
