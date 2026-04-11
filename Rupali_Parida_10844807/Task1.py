from dotenv import load_dotenv
import os
import json
import random
import requests
import urllib3

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

language_model = init_llm("gpt-4o", temperature=0.7, max_tokens=512)
output_parser = StrOutputParser()

technical_explanation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an SAP technical consultant."),
        (
            "user",
            "Explain how to retrieve top sales orders using "
            "API_SALES_ORDER_SRV/A_SalesOrder from SAP API Business Hub. "
            "Do not use bold text.",
        ),
    ]
)

technical_chain = technical_explanation_prompt | language_model | output_parser
technical_explanation = technical_chain.invoke({})

print("\nHow to retrieve top sales orders:")
print(technical_explanation)

sap_sales_order_url = (
    "https://172.19.151.9:44302/sap/opu/odata/sap/"
    "API_SALES_ORDER_SRV/A_SalesOrder?$top=10&$format=json"
)

sap_response = requests.get(
    sap_sales_order_url,
    auth=(os.getenv("SAP_USERNAME"), os.getenv("SAP_PASSWORD")),
    headers={"Accept": "application/json"},
    verify=False,
)

sap_response.raise_for_status()
sap_data = sap_response.json()
sales_order_records = sap_data["d"]["results"]

print("\nAll Retrieved Sales Orders:")
for count, sales_order in enumerate(sales_order_records, start=1):
    print(f"\nSales Order {count}")
    print(f"  Sales Order     : {sales_order.get('SalesOrder')}")
    print(f"  Sales Org       : {sales_order.get('SalesOrganization')}")
    print(f"  Sold-To Party   : {sales_order.get('SoldToParty')}")
    print(
        f"  Net Amount      : {sales_order.get('TotalNetAmount')} {sales_order.get('TransactionCurrency')}"
    )
    print(f"  Created On      : {sales_order.get('CreationDate')}")

focused_sales_order = random.choice(sales_order_records)

print("\nSelected Sales Order:")
print(f"Sales Order Number: {focused_sales_order.get('SalesOrder')}")

business_explanation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior business analyst explaining SAP data "
            "to non-technical stakeholders. Limit answer to 1600 characters.",
        ),
        (
            "user",
            "Explain the following SAP sales order in bullet points using "
            "a narrative style focused on customer, value, and business meaning. "
            "Avoid any formatting like bold.\n{order_details}",
        ),
    ]
)

business_chain = business_explanation_prompt | language_model | output_parser

business_explanation = business_chain.invoke(
    {"order_details": json.dumps(focused_sales_order, indent=2)}
)

print("\nBusiness Explanation of Selected Sales Order:")
print(business_explanation)
