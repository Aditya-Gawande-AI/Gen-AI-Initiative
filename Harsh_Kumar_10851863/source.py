import os
import requests
import urllib3
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from pathlib import Path


 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# -----------------------
# SAP CONFIG
# -----------------------

SAP_URL = os.getenv("SAP_URL")
SAP_USER = os.getenv("SAP_USER")
SAP_PASSWORD = os.getenv("SAP_PASSWORD")

# -----------------------
# Agent 1: API Info\
# -----------------------
def agent_1_output():
    print("▶ Agent 1: API Knowledge\n")
    print("Service  : API_SALES_ORDER_SRV")
    print("Entity   : A_SalesOrder")
    print("Purpose  : Fetch Top 10 Sales Orders")
    print("Query    : $top=10, $select, $filter, $orderby\n")
 
 
# -----------------------
# Agent 2: Fetch Top 10 Sales Orders
# -----------------------
def agent_2_output():
    print("▶ Agent 2: Top 10 Sales Orders\n")
 
    response = requests.get(
        SAP_URL,
        auth=HTTPBasicAuth(SAP_USER, SAP_PASSWORD),
        headers={"Accept": "application/json"},
        verify=False,
        timeout=30
    )
 
    if response.status_code != 200:
        print(f"❌ SAP Error: HTTP {response.status_code}")
        return []
 
    data = response.json()
    orders = data.get("d", {}).get("results", [])
 
    print(f"{'SO No':<10} {'Sold-To':<12} {'Sales Org':<10} "
          f"{'Net Amount':<12} {'Currency'}")
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
# Agent 3: Explain One Sales Order
# -----------------------
def agent_3_output(order):
    print("▶ Agent 3: Sales Order Explanation\n")
 
    print(f"Sales Order No     : {order.get('SalesOrder')}")
    print(f"Sold-To Party      : {order.get('SoldToParty')}")
    print(f"Sales Organization : {order.get('SalesOrganization')}")
    print(f"Net Amount         : {order.get('TotalNetAmount')} "
          f"{order.get('TransactionCurrency')}")
    print("Meaning            : Sales order header fetched from SAP via OData\n")
 
 
# -----------------------
# Main
# -----------------------
def main():
    print("\n==============================")
    print("  SAP Multi-Agent Output")
    print("==============================\n")
 
    # Agent 1
    agent_1_output()
 
    # Agent 2
    orders = agent_2_output()
 
    # Agent 3
    if orders:
        agent_3_output(orders[0])
    else:
        print("❌ No sales orders available for explanation")

if __name__ == "__main__":
    main()
 