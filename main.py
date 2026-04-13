import os
import json
import warnings
import requests
import urllib3
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from langchain_core.messages import HumanMessage

# GenAI Hub LLM
from gen_ai_hub.proxy.langchain.init_models import init_llm


# ------------------ BASIC SETUP ------------------
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load .env from current folder
load_dotenv()


# ------------------ LLM via GenAI Hub ------------------
def get_llm():
    # If credentials are wrong, this can fail at runtime with invalid_client
    return init_llm("gpt-4o")


# ------------------ Helper: clean printing ------------------
def line(title=""):
    if title:
        print("\n" + "=" * 22 + f" {title} " + "=" * 22 + "\n")
    else:
        print("\n" + "=" * 60 + "\n")


def safe(v, default=""):
    return default if v is None else v


def explain_status(code, mapping):
    if code in mapping:
        return f"{code} ({mapping[code]})"
    return safe(code, "N/A")


# ------------------ AGENT 1 ------------------
def agent1(llm):
    prompt = (
        "Explain how to retrieve top Sales Orders using SAP OData service "
        "API_SALES_ORDER_SRV (EntitySet: A_SalesOrder) via SAP API Business Hub. "
        "Include: where to find it on the hub, sample $top=10 query, what response looks like, "
        "and how to test it."
    )

    line("AGENT 1 OUTPUT (LLM)")
    resp = llm.invoke([HumanMessage(content=prompt)])
    print(resp.content)
    return resp.content


# ------------------ AGENT 2 ------------------
def agent2_fetch_sales_orders(debug=True):
    sap_base_url = os.getenv("SAP_BASE_URL", "https://172.19.151.9:44302").rstrip("/")
    sap_username = os.getenv("SAP_USERNAME", "Developer")
    sap_password = os.getenv("SAP_PASSWORD", "dev@S09")

    # Force JSON output using $format=json
    sap_url = (
        f"{sap_base_url}/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder"
        "?$top=10&$format=json"
    )

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    line("AGENT 2 OUTPUT (SAP API)")
    print("Calling:", sap_url)

    try:
        response = requests.get(
            sap_url,
            auth=HTTPBasicAuth(sap_username, sap_password),
            headers=headers,
            verify=False,
            timeout=30,
            allow_redirects=True
        )

        if debug:
            print("HTTP Status :", response.status_code)
            print("Content-Type:", response.headers.get("Content-Type"))
            print("Final URL   :", response.url)

        if response.status_code in (401, 403):
            print("\n Auth/Permission issue calling SAP API. Response snippet:\n")
            print(response.text[:600])
            return None

        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            print("\n SAP response is NOT JSON (maybe HTML/XML). First 600 chars:\n")
            print(response.text[:600])
            return None

        if isinstance(data, dict) and "error" in data:
            print("\n SAP returned an error JSON:\n")
            print(json.dumps(data["error"], indent=2))
            return None

        sales_orders = data.get("d", {}).get("results", [])

        print("\n Successfully retrieved top 10 Sales Orders from SAP.")
        print("Total Orders:", len(sales_orders))

        # Clean table-like preview for first 10
        if sales_orders:
            print("\n TOP SALES ORDERS (Preview)")
            print("-" * 80)
            print(f"{'SalesOrder':<12} {'SoldToParty':<12} {'NetAmount':<12} {'Currency':<8} {'Status':<12}")
            print("-" * 80)
            for so in sales_orders[:10]:
                print(
                    f"{safe(so.get('SalesOrder'),''): <12} "
                    f"{safe(so.get('SoldToParty'),''): <12} "
                    f"{safe(so.get('TotalNetAmount'),''): <12} "
                    f"{safe(so.get('TransactionCurrency'),''): <8} "
                    f"{safe(so.get('OverallSDProcessStatus'),''): <12}"
                )
            print("-" * 80)

        return data

    except requests.exceptions.RequestException as e:
        print("\n Failed to retrieve Sales Orders from SAP.")
        print("Error:", repr(e))
        return None


# ------------------ AGENT 3 ------------------
def agent3_explain_one_sales_order(llm, sales_orders_json):
    line("AGENT 3 OUTPUT (LLM)")

    if not sales_orders_json:
        print(" Agent 2 did not return data, so Agent 3 cannot run.")
        return None

    results = sales_orders_json.get("d", {}).get("results", [])
    one_sales_order = results[0] if results else None

    if not one_sales_order:
        print(" No Sales Order found in Agent 2 output.")
        return None

    prompt = f"""
Explain the following SAP Sales Order in very simple terms.
Also list the important fields and what they mean.

Sales Order JSON:
{json.dumps(one_sales_order, indent=2)}
"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    print(resp.content)
    return resp.content


# ------------------ CLEAN FALLBACK (when LLM is not available) ------------------
def fallback_clean_sales_order_explanation(sales_orders_json):
    line("AGENT 3 (FALLBACK - CLEAN OUTPUT)")

    results = (sales_orders_json or {}).get("d", {}).get("results", [])
    if not results:
        print("No sales order available to display.")
        return

    so = results[0]

    # Simple mappings (only what we know for sure)
    delivery_status_map = {"C": "Completed", "A": "Not Yet Started", "B": "Partially Processed"}
    process_status_map = {"C": "Completed", "A": "Not Yet Processed", "B": "Partially Processed"}
    block_status_map = {"C": "Not Blocked", "B": "Blocked"}

    print(" LLM not available due to GenAI Hub authentication issue (invalid_client/bad credentials).")
    print(" Showing Sales Order details.")

    print("\n SALES ORDER SUMMARY")
    print("-" * 60)
    print(f"Sales Order Number      : {safe(so.get('SalesOrder'),'N/A')}")
    print(f"Sales Order Type        : {safe(so.get('SalesOrderType'),'N/A')}")
    print(f"Sales Organization      : {safe(so.get('SalesOrganization'),'N/A')}")
    print(f"Distribution Channel    : {safe(so.get('DistributionChannel'),'N/A')}")
    print(f"Division                : {safe(so.get('OrganizationDivision'),'N/A')}")
    print(f"Sold-To Party (Customer): {safe(so.get('SoldToParty'),'N/A')}")
    print(f"Purchase Order (Cust)   : {safe(so.get('PurchaseOrderByCustomer'),'N/A')}")
    print(f"Total Net Amount        : {safe(so.get('TotalNetAmount'),'N/A')} {safe(so.get('TransactionCurrency'),'')}")
    print(f"Overall Process Status  : {explain_status(so.get('OverallSDProcessStatus'), process_status_map)}")
    print(f"Delivery Status         : {explain_status(so.get('OverallDeliveryStatus'), delivery_status_map)}")
    print(f"Block Status            : {explain_status(so.get('TotalBlockStatus'), block_status_map)}")
    print(f"Payment Terms           : {safe(so.get('CustomerPaymentTerms'),'N/A')}")
    print(f"Incoterms               : {safe(so.get('IncotermsClassification'),'N/A')} - {safe(so.get('IncotermsLocation1'),'')}")
    print("-" * 60)

    print("\n SIMPLE BUSINESS EXPLANATION")
    print("-" * 60)
    print(
        "This record is a Sales Order header from SAP (customer order).\n"
        "It shows who placed the order (Sold-To Party), the org details (Sales Org/Channel/Division),\n"
        "and the commercial values (Net Amount, Currency, Payment Terms).\n"
        "Status fields indicate whether the order and delivery are completed/blocked/partial."
    )
    print("-" * 60)


# ------------------ MAIN ------------------
def main():
    line("INIT")

    # Try to initialize LLM (Agent 1 & 3). If it fails, continue with Agent 2 + fallback output.
    llm = None
    llm_ok = False

    try:
        llm = get_llm()
        llm_ok = True
        print(" GenAI Hub LLM initialized successfully.")
    except Exception as e:
        print(" GenAI Hub LLM init FAILED (likely invalid_client / bad credentials).")
        print("Error:", repr(e))
        print("\n Proceeding with Agent 2 (SAP API) and clean fallback for Agent 3.\n")

    # Agent 1
    if llm_ok:
        try:
            agent1(llm)
        except Exception as e:
            print("\n Agent 1 failed.")
            print("Error:", repr(e))

    # Agent 2 (always)
    sales_orders = agent2_fetch_sales_orders(debug=True)

    # Agent 3
    if llm_ok:
        try:
            agent3_explain_one_sales_order(llm, sales_orders)
        except Exception as e:
            print("\n Agent 3 failed.")
            print("Error:", repr(e))
            # fallback anyway
            fallback_clean_sales_order_explanation(sales_orders)
    else:
        fallback_clean_sales_order_explanation(sales_orders)


if __name__ == "__main__":
    main()
