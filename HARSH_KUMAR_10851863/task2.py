from dotenv import load_dotenv
import os
import requests

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = init_llm("gpt-4o")
parser = StrOutputParser()

# -----------------------------
# Agent 1
# -----------------------------
def agent1_company_info(llm, company_name: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a business analyst. Be accurate, practical, and concise."),
        ("user",
         "Gather information about the company '{company_name}'.\n\n"
         "Include:\n"
         "1) Industry\n"
         "2) Headquarters\n"
         "3) Core business / products\n"
         "4) Recent highlights (keep it high-level)\n\n"
         "Answer in a structured format with headings."
        )
    ])
    return (template | llm | parser).invoke({"company_name": company_name})

# -----------------------------
# Agent 2 (MarketStack + LLM)
# -----------------------------
def agent2_stock_price(llm, company_name: str, symbol: str) -> str:
    api_key = os.getenv("MARKETSTACK_API_KEY")
    if not api_key:
        return "MARKETSTACK_API_KEY is not set in your .env file."

    url = "http://api.marketstack.com/v1/eod"
    params = {"access_key": api_key, "symbols": symbol, "limit": 1}
    resp = requests.get(url, params=params, timeout=10).json()

    if "error" in resp:
        return f"MarketStack Error: {resp['error'].get('message', resp['error'])}"

    if "data" not in resp or not resp["data"]:
        return "Stock market data not available."

    stock = resp["data"][0]
    raw_data = (
        f"Symbol: {stock.get('symbol')}\n"
        f"Exchange: {stock.get('exchange')}\n"
        f"Date: {stock.get('date')}\n"
        f"Open: {stock.get('open')}\n"
        f"Close: {stock.get('close')}\n"
        f"Volume: {stock.get('volume')}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial market analyst."),
        ("user",
         "Company: {company_name}\n\n"
         "Raw Stock Data:\n{raw_data}\n\n"
         "Explain the stock price clearly (exchange, close price, date). "
         "Mention this is end-of-day data if real-time is not available."
        )
    ])

    return (prompt | llm | parser).invoke({"company_name": company_name, "raw_data": raw_data})

# -----------------------------
# Agent 3
# -----------------------------
def agent3_final_report(llm, company_info: str, stock_info: str) -> str:
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a senior financial analyst."),
        ("user",
         "Company Information:\n{company_info}\n\n"
         "Stock Market Information:\n{stock_info}\n\n"
         "Generate a professional report that:\n"
         "1) Summarizes the company\n"
         "2) Interprets the stock price\n"
         "3) Gives a short insight/outlook"
        )
    ])
    return (template | llm | parser).invoke({"company_info": company_info, "stock_info": stock_info})

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    company_name = input("Enter company name: ")
    symbol = input("Enter stock symbol (e.g., AAPL): ")

    print("\n🔹 Agent 1: Fetching company information...")
    company_info = agent1_company_info(llm, company_name)
    print(company_info)

    print("\n🔹 Agent 2: Fetching stock market information...")
    stock_info = agent2_stock_price(llm, company_name, symbol)
    print(stock_info)

    print("\n🔹 Agent 3: Generating final report...\n")
    final_report = agent3_final_report(llm, company_info, stock_info)
    print(final_report)