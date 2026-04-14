import os
import requests
import urllib3
from requests.auth import HTTPBasicAuth

# ✅ LangChain (OpenAI) - used in Agent 1 & Agent 3
from langchain_openai import ChatOpenAI

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------
# SAP CONFIG
# -----------------------
SAP_URL = os.getenv(
    "SAP_URL",
    "https://172.19.151.9:44302/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10"
)
SAP_USER = os.getenv("SAP_USER", "Developer")
SAP_PASS = os.getenv("SAP_PASS", "dev@S09")

# ✅ Set SKIP_LLM=1 to disable LLM safely (corporate network)
SKIP_LLM = os.getenv("SKIP_LLM", "0") == "1"


# -----------------------
# LangChain Safe LLM Call
# -----------------------
def llm_invoke_safe(prompt: str) -> str:
    """
    Uses LangChain ChatOpenAI to call the LLM.
    If key missing / network blocked, returns a failure message string.
    """
    if SKIP_LLM:
        return "(LLM skipped – SKIP_LLM=1)"

    if not os.getenv("OPENAI_API_KEY"):
        return "(OPENAI_API_KEY not set)"

    try:
        llm = ChatOpenAI(temperature=0)  
        result = llm.invoke(prompt)

        # ChatOpenAI returns an AIMessage -> take .content
        return getattr(result, "content", str(result))

    except Exception as e:
        return f"(LLM failed: {e})"


# -----------------------
# Agent 1: API Info (LangChain + fallback)
# -----------------------
def agent_1_output():
    print("▶ Agent 1: API Knowledge (LangChain)\n")

    prompt = (
        "In 5 short bullet points, explain how to view top 10 sales orders using "
        "SAP OData service API_SALES_ORDER_SRV and entity A_SalesOrder. "
        "Mention $top=10, $select, $filter, $orderby with short examples."
    )

    response = llm_invoke_safe(prompt)

    # ✅ fallback if LLM fails
    if response.startswith("(LLM failed") or "OPENAI_API_KEY" in response or "skipped" in response.lower():
        response = (
            "- Service: API_SALES_ORDER_SRV\n"
            "- Entity: A_SalesOrder\n"
            "- Top 10: /A_SalesOrder?$top=10\n"
            "- Select fields: $select=SalesOrder,SoldToParty,TotalNetAmount\n"
            "- Sort/filter: $orderby=CreationDate desc, $filter=SalesOrganization eq '1710'\n"
        )

    print(response)
    print()


# -----------------------
# Agent 2: Fetch Top 10 Sales Orders (short table)
# -----------------------
def agent_2_output():
    print("▶ Agent 2: Top 10 Sales Orders\n")

    response = requests.get(
        SAP_URL,
        auth=HTTPBasicAuth(SAP_USER, SAP_PASS),
        headers={"Accept": "application/json"},
        verify=False,
        timeout=30
    )

    if response.status_code != 200:
        print(f"❌ SAP Error: HTTP {response.status_code}")
        print(response.text[:300])  
        return []

    data = response.json()
    orders = data.get("d", {}).get("results", [])

    if not orders:
        print("❌ No sales orders found\n")
        return []

    print(f"{'SO No':<10} {'Sold-To':<12} {'Sales Org':<10} {'Net Amount':<12} {'Curr'}")
    print("-" * 60)

    for so in orders:
        print(
            f"{so.get('SalesOrder', ''):<10} "
            f"{so.get('SoldToParty', ''):<12} "
            f"{so.get('SalesOrganization', ''):<10} "
            f"{so.get('TotalNetAmount', ''):<12} "
            f"{so.get('TransactionCurrency', '')}"
        )

    print()
    return orders


# -----------------------
# Agent 3: Explain 1 Sales Order (LangChain + fallback, short)
# -----------------------
def agent_3_output(order):
    print("▶ Agent 3: Explain 1 Sales Order (LangChain)\n")

    # Keep prompt small to keep output short
    compact = {
        "SalesOrder": order.get("SalesOrder"),
        "SoldToParty": order.get("SoldToParty"),
        "SalesOrganization": order.get("SalesOrganization"),
        "TotalNetAmount": order.get("TotalNetAmount"),
        "TransactionCurrency": order.get("TransactionCurrency"),
        "OverallSDProcessStatus": order.get("OverallSDProcessStatus")
    }

    prompt = (
        "Explain this SAP sales order in 2-3 simple lines (business meaning). "
        "Use the provided fields only:\n"
        f"{compact}"
    )

    explanation = llm_invoke_safe(prompt)

    # ✅ fallback if LLM fails
    if explanation.startswith("(LLM failed") or "OPENAI_API_KEY" in explanation or "skipped" in explanation.lower():
        explanation = (
            f"Sales Order {compact.get('SalesOrder')} for customer {compact.get('SoldToParty')} "
            f"in Sales Org {compact.get('SalesOrganization')}. "
            f"Net Amount: {compact.get('TotalNetAmount')} {compact.get('TransactionCurrency')}."
        )

    print(explanation)
    print()


# -----------------------
# Main
# -----------------------
def main():
    print("\n==============================")
    print("   SAP Multi-Agent Output")
    print("==============================\n")

    # Agent 1 (LangChain)
    agent_1_output()

    # Agent 2 (SAP API)
    orders = agent_2_output()

    # Agent 3 (LangChain)
    if orders:
        agent_3_output(orders[0])
    else:
        print("❌ No sales orders available for explanation\n")

    print("✅ Completed\n")


if __name__ == "__main__":
    main()