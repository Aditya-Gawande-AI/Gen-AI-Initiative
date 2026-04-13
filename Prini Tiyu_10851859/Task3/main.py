import os
from dotenv import load_dotenv

from agents.agent1_gmail import fetch_latest_image_attachment
from agents.agent2_serial import extract_serial_from_image_bytes
from agents.agent3_sap_so import create_sales_order


def main():
    load_dotenv()

    print("\n" + "="*60)
    print(" AGENTIC AI TASK 3 - EXECUTION STARTED")
    print("="*60)

    # ===== Agent 1 =====
    print("\n Agent 1: Reading Gmail & fetching attachment...")
    query = os.getenv("GMAIL_QUERY", "in:anywhere has:attachment")
    filename, image_bytes, msg_id = fetch_latest_image_attachment(query)
    print(f" Attachment fetched successfully")
    print(f"    File Name   : {filename}")
    print(f"    Message ID : {msg_id}")

    # ===== Agent 2 =====
    print("\n Agent 2: Extracting Serial Number from image using LLM...")
    serial = extract_serial_from_image_bytes(image_bytes, filename=filename)

    if not serial.strip():
        raise RuntimeError(" Agent 2 failed to extract serial number.")

    print(f" Serial Number extracted")
    print(f"    Serial Number : {serial}")

    # ===== Agent 3 =====
    print("\n Agent 3: Creating Sales Order in SAP...")
    result = create_sales_order(serial)

    sales_order = result.get("d", {}).get("SalesOrder", "UNKNOWN")

    print(f" Sales Order created successfully in SAP")
    print(f"    Sales Order No             : {sales_order}")
    print(f"    PurchaseOrderByCustomer    : {serial}")

    # ===== Final Summary =====
    print("\n" + "="*60)
    print(" TASK COMPLETED SUCCESSFULLY ")
    print("="*60)
    print(f" Gmail Attachment   : {filename}")
    print(f" Extracted Serial   : {serial}")
    print(f" SAP Sales Order    : {sales_order}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
