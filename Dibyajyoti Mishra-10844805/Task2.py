from dotenv import load_dotenv
import os
import requests
 
# LangChain / LLM
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
 
# ------------------ LOAD CONFIG ------------------
load_dotenv()
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY")
 
# ------------------ LLM SETUP ------------------
llm = init_llm("gpt-4o")
 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial and business analyst."),
    ("user", "{input}")
])
 
chain = prompt | llm | StrOutputParser()
  
# ------------------ USER INPUT ------------------
company_name = input("Enter Company Name: ")
 
# =================================================
# AGENT 1 – COMPANY INFORMATION
# =================================================
print("\n Running Agent 1 – Company Overview")
 
agent1_query = f"""
Provide a concise overview of the company {company_name}.
Include:
- Industry
- Core business
- Headquarters
- Major products/services
"""
 
company_info = chain.invoke({"input": agent1_query})
print("\nAgent 1 Output:\n", company_info)
 
if collection:
    collection.insert_one({"agent": "agent1", "company": company_name, "data": company_info})
 
# =================================================
# AGENT 2 – STOCK PRICE FETCH
# =================================================
print("\n Running Agent 2 – Stock Price")
 
stock_price_data = None
 
try:
    search_url = "http://api.marketstack.com/v1/tickers"
    search_params = {
        "access_key": MARKETSTACK_API_KEY,
        "search": company_name,
        "limit": 1
    }
 
    ticker_response = requests.get(search_url, params=search_params, timeout=10).json()
    ticker_symbol = ticker_response["data"][0]["symbol"]
 
    eod_url = "http://api.marketstack.com/v1/eod/latest"
    eod_params = {
        "access_key": MARKETSTACK_API_KEY,
        "symbols": ticker_symbol
    }
 
    stock_response = requests.get(eod_url, params=eod_params, timeout=10).json()
    stock_price_data = stock_response["data"][0]
 
    # Formatted Output
    print(f"Company Symbol : {ticker_symbol}")
    print(f"Stock Price    : {stock_price_data['close']}")
    print(f"Exchange       : {stock_price_data['exchange']}")
 
except Exception as e:
    print(" MarketStack API error:", e)
    stock_price_data = None
 
# =================================================
# AGENT 3 – FINAL COMPREHENSION REPORT
# =================================================
print("\n Running Agent 3 – Business & Stock Analysis")
 
agent3_query = f"""
Using the information below, generate a clear business summary:
 
Company Information:
{company_info}
 
Stock Market Information:
{stock_price_data}
 
Explain:
- Company's market position
- Stock performance context
- Overall business outlook
"""
 
final_report = chain.invoke({"input": agent3_query})
print("\nAgent 3 Output:\n", final_report)
 
if collection:
    collection.insert_one({"agent": "agent3", "company": company_name, "data": final_report})
 
print("\n Task Completed Successfully")