import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import urllib3
import os
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

llm = init_llm("gpt-4o")

SAP_USERNAME = os.getenv("SAP_USERNAME")
SAP_PASSWORD = os.getenv("SAP_PASSWORD")

def agent1_research():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SAP API expert."),
        ("user",
         "Explain in detail how to retrieve top sales orders using "
         "API_SALES_ORDER_SRV/A_SalesOrder from SAP API Business Hub. "
         "Include OData options like $top, $select, and $orderby.")
    ])

    chain = prompt | llm | StrOutputParser()
    output = chain.invoke({})

    print("\nAGENT 1 OUTPUT\n")
    print(output)

def agent2_fetch_sales_orders():
    url = (
        "https://172.19.151.9:44302/"
        "sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10"
    )

    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(SAP_USERNAME,SAP_PASSWORD),
            headers={"Accept": "application/json"},
            verify=False,
            timeout=10
        )

        response.raise_for_status()
        orders = response.json().get("d", {}).get("results", [])

        print("\nTOP 10 SALES ORDERS\n")

        for idx, order in enumerate(orders, start=1):
            print(
                f"{idx}. Sales Order : {order.get('SalesOrder')}, "
                f"Order Type : {order.get('SalesOrderType')}, "
                f"Sales Organization : {order.get('SalesOrganization')}, "
                f"Distribution Ch. : {order.get('DistributionChannel')}, "
                f"Division : {order.get('OrganizationDivision')}, "
                f"Sales District : {order.get('SalesDistrict')}, "
                f"Sold-To Party : {order.get('SoldToParty')}"
            )

        return orders

    except Exception as e:
        print("SAP Connection Error:", e)
        return []

def agent3_explain_sales_order(order):
    order_summary = (
        f"Sales Order : {order.get('SalesOrder')}, "
        f"Order Type : {order.get('SalesOrderType')}, "
        f"Sales Organization : {order.get('SalesOrganization')}, "
        f"Distribution Channel : {order.get('DistributionChannel')}, "
        f"Division : {order.get('OrganizationDivision')}, "
        f"Sales District : {order.get('SalesDistrict')}, "
        f"Sold-To Party : {order.get('SoldToParty')}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SAP Sales and Distribution functional consultant."),
        ("user", "Explain the following sales order in business-friendly terms:\n{order}")
    ])

    chain = prompt | llm | StrOutputParser()
    explanation = chain.invoke({"order": order_summary})

    print("\nAGENT 3 OUTPUT\n")
    print(explanation)

if __name__ == "__main__":
    agent1_research()
    sales_orders = agent2_fetch_sales_orders()
    if sales_orders:
        agent3_explain_sales_order(sales_orders[0])
