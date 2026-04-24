import os
import requests
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------- Agent 1 ----------
def agent1_company_info(company_name: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("user", """
Company: {company_name}

Provide:
1) Business overview (2-3 lines)
2) Industry
3) Headquarters (if known)
4) Key products/services (bullets)
""")
    ])
    llm = init_llm("gpt-4o", max_tokens=300)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"company_name": company_name})

# ---------- Agent 2 ----------
def fetch_marketstack_eod(symbol: str) -> dict:
    api_key = os.getenv("MARKETSTACK_API_KEY")
    if not api_key:
        return {"error": "MARKETSTACK_API_KEY missing in .env"}

    # MarketStack docs: access_key + symbols parameters [4](https://marketstack.com/documentation)[5](https://marketstack.com/documentation_v2)
    url = "https://api.marketstack.com/v1/eod/latest"
    params = {"access_key": api_key, "symbols": symbol}

    r = requests.get(url, params=params, timeout=30)
    data = r.json()

    try:
        row = data["data"][0]
        return {
            "symbol": row.get("symbol"),
            "date": row.get("date"),
            "close": row.get("close"),
            "exchange": row.get("exchange"),
            "name": row.get("name")
        }
    except Exception:
        return {"error": "Could not parse MarketStack response", "raw": data}

def agent2_stock_summary(stock_json: dict) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a finance assistant. Summarize clearly."),
        ("user", """
Here is stock data (JSON):
{stock_json}

Return a 2-3 line summary with symbol, date and close price. If error exists, explain error in 1 line.
""")
    ])
    llm = init_llm("gpt-4o", max_tokens=200)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"stock_json": stock_json})

# ---------- Agent 3 ----------
def agent3_final_report(company_info: str, stock_summary: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior analyst. Keep it short and clear."),
        ("user", """
Using the information below, write a concise report:

Company Info:
{company_info}

Stock Summary:
{stock_summary}

Include:
1) Company snapshot (2-3 lines)
2) Stock status (1-2 lines)
3) High-level interpretation (non-advisory, 2-3 lines)
""")
    ])
    llm = init_llm("gpt-4o", max_tokens=350)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"company_info": company_info, "stock_summary": stock_summary})

if __name__ == "__main__":
    load_dotenv()

    company = input("Enter company name: ").strip()
    symbol = input("Enter stock symbol : ").strip()

    out1 = agent1_company_info(company)
    print("\n--- Agent 1 Output (Company Info) ---\n", out1)

    stock_data = fetch_marketstack_eod(symbol)
    out2 = agent2_stock_summary(stock_data)
    print("\n--- Agent 2 Output (Stock Summary) ---\n", out2)

    out3 = agent3_final_report(out1, out2)
    print("\n--- Agent 3 Output (Final Report) ---\n", out3)