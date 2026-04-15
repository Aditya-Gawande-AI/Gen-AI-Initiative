

import os
import json
import urllib3
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# Disable SSL warnings for internal/self-signed cert environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_FILE = "agent2_output.json"

def main():
    # Load env values from config.env
    load_dotenv("config.env")

    base_url = os.getenv("SAP_BASE_URL", "").strip()
    username = os.getenv("SAP_USERNAME", "").strip()
    password = os.getenv("SAP_PASSWORD", "").strip()
    top_n = int(os.getenv("TOP_N", "10").strip())

    if not base_url or not username or not password:
        print("❌ Missing values in config.env")
        print("Make sure config.env contains SAP_BASE_URL, SAP_USERNAME, SAP_PASSWORD")
        return

    # Build final SAP OData URL
    url = f"{base_url}/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top={top_n}"

    print("\n▶ Agent 2: Top Sales Orders (SAP API Call)\n")
    print("Calling:", url)

    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept": "application/json"},
            verify=False,   # internal SAP endpoint / self-signed cert
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        print("\n❌ Network/API call failed.")
        print("Reason:", str(e))
        print("👉 Check VPN / network access to SAP host.")
        return

    # Check HTTP status
    if response.status_code != 200:
        print("\n❌ SAP API returned error")
        print("HTTP Status:", response.status_code)
        print("Response Text (first 500 chars):")
        print(response.text[:500])
        print("\n👉 Possible causes: wrong password, no authorization, VPN issue.")
        return

    # Parse JSON
    try:
        data = response.json()
    except Exception:
        print("\n❌ Response was not valid JSON.")
        print("Response Text (first 500 chars):")
        print(response.text[:500])
        return

    # Extract orders list (SAP OData format)
    try:
        orders = data["d"]["results"]
    except Exception:
        print("\n❌ Unexpected SAP response structure.")
        print(json.dumps(data, indent=2))
        return

    # Save raw orders to local json file for Agent 3
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(orders, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("\n⚠️ Could not save agent2_output.json:", str(e))

    # Print table (demo-friendly)
    print("\nAgent 2 Output: Top Sales Orders\n")
    print(f"{'No':<4}{'SalesOrder':<14}{'Sold-To':<12}{'Sales Org':<12}{'Net Amount':<14}{'Currency':<10}")
    print("-" * 66)

    for i, so in enumerate(orders, start=1):
        sales_order = str(so.get("SalesOrder", ""))
        sold_to = str(so.get("SoldToParty", ""))
        sales_org = str(so.get("SalesOrganization", ""))
        net_amt = str(so.get("TotalNetAmount", ""))
        currency = str(so.get("TransactionCurrency", so.get("Currency", "")))

        print(f"{i:<4}{sales_order:<14}{sold_to:<12}{sales_org:<12}{net_amt:<14}{currency:<10}")

    print(f"\n✅ Agent 2 completed. Raw output saved to {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
