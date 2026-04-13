import os
import requests
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from gen_ai_hub.proxy.langchain.init_models import init_llm


# Loading the env variables
load_dotenv()

SAP_BASE_URL = os.getenv("SAP_BASE_URL")
SAP_USERNAME = os.getenv("SAP_USERNAME")
SAP_PASSWORD = os.getenv("SAP_PASSWORD")

# Init LLM (SAP Generative AI Hub)
llm = init_llm("gpt-4o" )

# AGENT 1

print("\n---------------AGENT‑1---------------\n")

agent1_prompt = """
Explain how to view TOP SALES ORDERS using
SAP OData service API_SALES_ORDER_SRV
Entity A_SalesOrder.

Include:
- Endpoint
- $top
- $orderby
- Important fields
- Example request
"""

agent1_response = llm.invoke([
    HumanMessage(content=agent1_prompt)
])

print(agent1_response.content)

# AGENT 2 – Call SAP Oata API to get Top 10 Sales Orders

print("\n----------------AGENT‑2------------------\n")

url = f"{SAP_BASE_URL}/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10"

headers = {
    "Accept": "application/json"
}

response = requests.get(
    url,
    auth=(SAP_USERNAME, SAP_PASSWORD),
    headers=headers,
    verify=False
)

response.raise_for_status()
data = response.json()

sales_orders = data["d"]["results"]

print(f"Top {len(sales_orders)} Sales Orders fetched:\n")
for so in sales_orders:
    print(
        f"SalesOrder: {so.get('SalesOrder')}, "
        f"NetAmount: {so.get('TotalNetAmount')}, "
        f"Currency: {so.get('TransactionCurrency')}"
    )

# AGENT 3 – Asking LLM to explain ONE sales order

print("\n--------------AGENT‑3----------------\n")

one_sales_order = sales_orders[0]

agent3_prompt = f"""
Explain the following SAP Sales Order in simple terms:

Sales Order Number: {one_sales_order.get('SalesOrder')}
Net Amount: {one_sales_order.get('TotalNetAmount')}
Currency: {one_sales_order.get('TransactionCurrency')}
Sold-To Party: {one_sales_order.get('SoldToParty')}

Explain:
- What this sales order represents
- Business meaning of amount and currency
- Make the answer short and crisp
"""

agent3_response = llm.invoke([
    HumanMessage(content=agent3_prompt)
])

print(agent3_response.content)
