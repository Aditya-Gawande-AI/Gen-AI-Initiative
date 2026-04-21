from dotenv import load_dotenv
import os
import json
import requests
import urllib3
 
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
 
load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
 
llm = init_llm("gpt-4o")
 
 
agent1_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an SAP technical assistant."),
    ("user",
     "Explain how to retrieve top sales orders using "
     "API_SALES_ORDER_SRV/A_SalesOrder from SAP API Business Hub."
     "Do not bold any text.")
])
 
agent1_chain = agent1_prompt | llm | StrOutputParser()
agent1_response = agent1_chain.invoke({})
 
print("\nHow to view Top Sales Orders Data:")
print(agent1_response)
 
SAP_URL = (
    "https://172.19.151.9:44302/sap/opu/odata/sap/"
    "API_SALES_ORDER_SRV/A_SalesOrder?$top=10&$format=json"
)
 
response = requests.get(
    SAP_URL,
    auth=(os.getenv("SAP_USERNAME"), os.getenv("SAP_PASSWORD")),
    headers={"Accept": "application/json"},
    verify=False
)
 
response.raise_for_status()
sales_orders = response.json()
orders_list = sales_orders["d"]["results"]
 
print("\nAvailable Sales Orders:")
for idx, order in enumerate(orders_list, start=1):
    print(f"\nOrder {idx}")
    print(f"  Sales Order   : {order.get('SalesOrder')}")
    print(f"  Sold-To Party : {order.get('SoldToParty')}")
    print(f"  Net Amount    : {order.get('TotalNetAmount')} {order.get('TransactionCurrency')}")
 
while True:
    try:
        choice = int(input("\nEnter a number between 1 and 10 to explain that sales order: "))
        if 1 <= choice <= 10:
            break
        else:
            print("Please enter a number between 1 and 10.")
    except ValueError:
        print("Please enter a valid numeric value.")
 
selected_order = orders_list[choice - 1]
 
print("\nSelected Sales Order:")
print(f"Sales Order Number: {selected_order.get('SalesOrder')}")
 
 
agent3_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a business analyst. Explain sales orders clearly "
     "for non-technical stakeholders."),
    ("user",
     "Explain the following SAP sales order in bullets but story like, "
     "focusing on customer, value, and business meaning. "
     "Everything should be without bold.\n{sales_order}")
])
 
agent3_chain = agent3_prompt | llm | StrOutputParser()
 
agent3_response = agent3_chain.invoke({
    "sales_order": json.dumps(selected_order, indent=2)
})
 
 
