"""
Agent 1:
- Call LLM (SAP Generative AI Hub via SDK) to explain how to view top sales orders
  using API_SALES_ORDER_SRV/A_SalesOrder on SAP API Business Hub.
- Print output to screen (no MongoDB)

Agent 2:
- Call SAP OData endpoint API_SALES_ORDER_SRV/A_SalesOrder?$top=10
- Save output to sales_orders.json

Agent 3:
- Read sales_orders.json
- Pick one sales order (default index 0)
- Call LLM to explain it in simple terms
- Print output to screen (no MongoDB)
"""

import os
import json
import warnings
import requests
from dotenv import load_dotenv, find_dotenv


# ----------------------------
# Shared Helpers (Env + LLM)
# ----------------------------
def load_env():
    """
    Loads .env reliably (even if you run from sap_agents folder).
    """
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        raise SystemExit(
            "❌ .env file not found.\n"
            "Place .env in your project root (Python_course/.env)."
        )
    load_dotenv(env_path, override=True)
    return env_path


def assert_aicore_credentials():
    required = [
        "AICORE_AUTH_URL",
        "AICORE_CLIENT_ID",
        "AICORE_CLIENT_SECRET",
        "AICORE_BASE_URL",
        "AICORE_RESOURCE_GROUP",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(
            "❌ SAP AI Core credentials missing.\n"
            f"Missing variables: {', '.join(missing)}\n\n"
            "Fix: Add these keys to .env (AICORE_*)."
        )


def extract_text(resp):
    """
    Works for both dict-like and object-like responses.
    """
    try:
        return resp.choices[0].message.content
    except Exception:
        pass
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        pass
    return str(resp)


def call_llm(messages, temperature=0.2, max_tokens=600):
    """
    Calls SAP GenAI Hub SDK (OpenAI-style wrapper) and returns raw response.
    Note: Import chat AFTER env is loaded.
    """
    from gen_ai_hub.proxy.native.openai import chat

    model_name = os.getenv("AICORE_LLM_MODEL", "gpt-4o")
    resp = chat.completions.create(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp


# ----------------------------
# Agent 1: LLM info about API
# ----------------------------
def run_agent_1():
    prompt = """
You are a SAP integration expert.

Explain how to view TOP sales orders using the
OData service API_SALES_ORDER_SRV / A_SalesOrder
as documented on SAP API Business Hub (api.sap.com).

Include:
1) What API_SALES_ORDER_SRV is
2) What A_SalesOrder represents
3) Example endpoint using $top=10
4) Brief notes on $select, $filter, $orderby
Keep it short and step-by-step.
""".strip()

    messages = [
        {"role": "system", "content": "You are a helpful SAP technical assistant."},
        {"role": "user", "content": prompt},
    ]

    resp = call_llm(messages=messages, temperature=0.2, max_tokens=600)

    print("\n" + "=" * 70)
    print("AGENT 1 OUTPUT (SAP GenAI Hub LLM)")
    print("=" * 70 + "\n")
    print(extract_text(resp))
    print("\n" + "-" * 70 + "\n")

    return resp


# ----------------------------
# Agent 2: Call OData and save
# ----------------------------
def run_agent_2(
    url="https://172.19.151.9:44302/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10",
    username="Developer",
    password="dev@S09",
    output_file="sales_orders.json",
):
    warnings.filterwarnings("ignore")

    response = requests.get(
        url,
        auth=(username, password),
        headers={"Accept": "application/json"},
        verify=False,
    )

    print("\n" + "=" * 70)
    print("AGENT 2 OUTPUT (OData API Call)")
    print("=" * 70)
    print("Status Code:", response.status_code)

    if response.status_code == 200:
        data = response.json()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Sales orders saved to {output_file}")
        print("\n" + "-" * 70 + "\n")
        return data
    else:
        print("❌ API call failed")
        print(response.text)
        print("\n" + "-" * 70 + "\n")
        return None


# ----------------------------
# Agent 3: Read JSON & explain
# ----------------------------
def read_sales_orders(file_path="sales_orders.json"):
    """
    Reads sales_orders.json produced by Agent 2.
    OData V2 usually returns: { "d": { "results": [ ... ] } }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("d", {}).get("results", [])
    if not results:
        raise ValueError("❌ No sales orders found inside d.results in sales_orders.json")
    return results


def run_agent_3(file_path="sales_orders.json", pick_index=0):
    sales_orders = read_sales_orders(file_path)

    if pick_index < 0 or pick_index >= len(sales_orders):
        raise ValueError(f"pick_index out of range. Valid: 0 to {len(sales_orders)-1}")

    one_so = sales_orders[pick_index]

    prompt = f"""
You are a SAP SD assistant.

Explain this Sales Order (header JSON) in simple terms.
Give your output in this structure:

1) Sales Order Number
2) Customer / Sold-to (if available)
3) Sales Organization / Distribution Channel / Division (if available)
4) Net value / Currency (if available)
5) Dates (document date, requested delivery date if available)
6) Overall status fields (if available)
7) What this order means in business language (2 lines)

Sales Order JSON:
{json.dumps(one_so, ensure_ascii=False)}
""".strip()

    messages = [
        {"role": "system", "content": "You explain SAP sales orders clearly for beginners."},
        {"role": "user", "content": prompt},
    ]

    resp = call_llm(messages=messages, temperature=0.2, max_tokens=900)

    print("\n" + "=" * 70)
    print(f"AGENT 3 OUTPUT – Sales Order Explanation (index: {pick_index})")
    print("=" * 70 + "\n")
    print(extract_text(resp))
    print("\n" + "-" * 70 + "\n")

    return resp


# ----------------------------
# Main Orchestrator: 1 → 2 → 3
# ----------------------------
def main(pick_index=0):
    env_path = load_env()
    print(f"✅ Loaded .env from: {env_path}")

    # Only needed for LLM calls (Agent 1 & 3)
    assert_aicore_credentials()

    # Run all agents in sequence
    run_agent_1()
    run_agent_2(output_file="sales_orders.json")
    run_agent_3(file_path="sales_orders.json", pick_index=pick_index)


if __name__ == "__main__":
    # Change pick_index if you want to explain a different sales order from top 10
    main(pick_index=0)