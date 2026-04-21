import os
import requests
from dotenv import load_dotenv
 
from langchain_core.messages import HumanMessage
from gen_ai_hub.proxy.langchain.init_models import init_llm
 
load_dotenv()
 
llm = init_llm(
    model_name="gpt-4o",
    max_tokens=1000
)
 
 
def print_step(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")
 
def agent_1_company_info(company_name):
    print_step("AGENT 1 - COMPANY INFORMATION")
 
    prompt = f"""
    Provide a brief overview of the company "{company_name}".
    # Include:
    # - Industry
    # - What the company does
    """
 
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
 
    return response.content
 
 
def resolve_stock_symbol(company_name):
    """
    Finds stock ticker symbol automatically using Marketstack search.
    """
    url = "http://api.marketstack.com/v1/tickers"
    params = {
        "access_key": os.getenv("MARKETSTACK_API_KEY"),
        "search": company_name,
        "limit": 1
    }
 
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
 
        if "data" in data and data["data"]:
            symbol = data["data"][0]["symbol"]
            name = data["data"][0]["name"]
            return symbol, f"Resolved symbol '{symbol}' for company '{name}'"
 
    except Exception as e:
        return None, f"Error resolving stock symbol: {e}"
 
    return None, "Stock symbol could not be resolved"
 
 
def agent_2_stock_price(company_name):
    print_step("AGENT 2 - STOCK PRICE")
 
 
    symbol, resolution_msg = resolve_stock_symbol(company_name)
    print(resolution_msg)
 
 
    url = "http://api.marketstack.com/v1/eod"
    params = {
        "access_key": os.getenv("MARKETSTACK_API_KEY"),
        "symbols": symbol,
        "limit": 1
    }
 
    try:
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
 
        if "data" in data and data["data"]:
            price = data["data"][0]["close"]
            result = f"Stock Symbol: {symbol}, Latest Closing Price: {price}"
        else:
            result = f"Stock price not available for symbol {symbol}"
 
    except Exception as e:
        result = f"Error fetching stock price: {e}"
 
    print(result)
    return result
 
def agent_3_final_report(company_info, stock_info):
    print_step("AGENT 3 - FINAL REPORT")
 
    prompt = f"""
    You are a business analyst.
 
    make it short and clear
   
    Company Information:
    {company_info}
 
    Stock Information:
    {stock_info}
    """
 
    response = llm.invoke([HumanMessage(content=prompt)])
    print(response.content)
 
    return response.content
 
 
if __name__ == "__main__":
    print_step("MULTI-AGENT EXECUTION STARTED")
 
    company_name = input("Enter company name: ")
 
    company_info = agent_1_company_info(company_name)
    stock_info = agent_2_stock_price(company_name)
    agent_3_final_report(company_info, stock_info)
 
    print_step("EXECUTION COMPLETED")