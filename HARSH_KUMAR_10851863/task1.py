import os
import json
import warnings
from typing import Any, Dict, List, Optional

import requests

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Gen AI Hub
from gen_ai_hub.proxy.langchain.init_models import init_llm



# ----------------------------
# Agent 1: LLM guidance prompt
# ----------------------------
def agent1(llm) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are an SAP API Business Hub assistant. Be accurate, practical, and concise."),
        ("user",
         "Explain how to view TOP sales orders using the OData service "
         "'API_SALES_ORDER_SRV' and entity set 'A_SalesOrder' on SAP API Business Hub.\n\n"
         "Include:\n"
         "1) Typical endpoint pattern\n"
         "2) How to use $top and $orderby (give examples)\n"
         "3) Example GET request with query options\n"
         "4) Where to find this in SAP API Business Hub docs\n"
         "5) Common troubleshooting tips (auth basics, GET doesn’t need CSRF)\n\n"
         "Answer in a step-by-step format."
)
    ])
    chain = template | llm | StrOutputParser()
    return chain.invoke({})


# ----------------------------
# Agent 2: SAP OData call
# ----------------------------
def fetch_top_sales_orders(top: int = 10) -> List[Dict[str, Any]]:
    """
    Calls:
      /sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10
    """

    base_url = os.getenv("SAP_URL")
    user = os.getenv("SAP_USER", "Developer")
    pwd = os.getenv("SAP_PASSWORD", "dev@S09")

    url = (
        base_url
    )

    headers = {"Accept": "application/json"}

    verify_ssl = os.getenv("SAP_VERIFY_SSL", "false").lower() in ("1", "true", "yes")
    if not verify_ssl:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    resp = requests.get(
        url,
        headers=headers,
        auth=(user, pwd),
        timeout=60,
        verify=verify_ssl,
    )
    resp.raise_for_status()
    data = resp.json()

    # OData v2 often: {"d": {"results": [...]}}
    # OData v4 often: {"value": [...]}
    if isinstance(data, dict):
        if "d" in data and isinstance(data["d"], dict) and "results" in data["d"]:
            return data["d"]["results"]
        if "value" in data and isinstance(data["value"], list):
            return data["value"]

    return []


def pick_one_sales_order(sales_orders: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not sales_orders:
        return None
    for so in sales_orders:
        if isinstance(so, dict) and any(k.lower() == "salesorder" for k in so.keys()):
            return so
    return sales_orders[0]

# ----------------------------
# Agent 3: LLM explains one order
# ----------------------------
def agent3(llm, one_order: Dict[str, Any]) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system",
         "You are an SAP SD expert. Explain the sales order clearly for a beginner."),
        ("user",
         "Given this Sales Order JSON from API_SALES_ORDER_SRV/A_SalesOrder, explain:\n"
         "1) What this order represents\n"
         "2) Key fields and their meaning\n"
         "3) Any red flags or missing data you notice\n"
         "4) What follow-up calls could be made (items, partners, pricing) if needed\n\n"
         "Sales Order JSON:\n{sales_order_json}")
    ])
    chain = template | llm | StrOutputParser()
    return chain.invoke({"sales_order_json": json.dumps(one_order, indent=2)})


def main():
    load_dotenv()

    print("\n==============================")
    print(" Task 1 - 3 Agents (No MongoDB)")
    print("==============================\n")

    # LLM init
    model_name = os.getenv("GENAI_MODEL", "gpt-4o")
    llm = init_llm(model_name)

    # -------- Agent 1 --------
    print("----- Agent 1: LLM guidance for TOP Sales Orders -----\n")
    a1 = agent1(llm)
    print(a1)

    # -------- Agent 2 --------
    print("\n----- Agent 2: Fetch TOP 10 Sales Orders from SAP OData -----\n")
    try:
        orders = fetch_top_sales_orders(top=10)
    except Exception as e:
        print("Agent 2 failed while calling SAP API.")
        print("Error:", repr(e))
        print("\nTip: If this is an SSL error, keep SAP_VERIFY_SSL=false (default).")
        return

    print(f"Fetched {len(orders)} sales orders.\n")
    for i, so in enumerate(orders[:10], start=1):
        if not isinstance(so, dict):
            continue
        sales_order_no = so.get("SalesOrder") or so.get("salesorder") or so.get("SalesOrderID") or "N/A"
        sold_to = so.get("SoldToParty") or so.get("SoldTo") or "N/A"
        org = so.get("SalesOrganization") or "N/A"
        print(f"{i:02d}. SalesOrder={sales_order_no} | SoldToParty={sold_to} | SalesOrg={org}")

    # -------- Agent 3 --------
    print("\n----- Agent 3: Explain ONE sales order using LLM -----\n")
    one = pick_one_sales_order(orders)
    if not one:
        print("No sales orders received, so Agent 3 cannot run.")
        return

    a3 = agent3(llm, one)
    print(a3)


if __name__ == "__main__":
    main()