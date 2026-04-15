from dotenv import load_dotenv
import os
import requests
 
# LLM (SAP GenAI Hub compatible)
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 

# ENV SETUP

load_dotenv()
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY")
 
model = init_llm("gpt-4o" , max_tokens=700)
 
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a expert business analyst."),
    ("user", "{input}")
])
 
chain = prompt | model | StrOutputParser()
 

# INPUT

company_name = input("\nEnter company name: ").strip()
 

# AGENT 1 – Company Information using LLM

print("\n================ AGENT 1 =================\n")
 
agent1_query = (
    f"Give a short description of company {company_name}. "
    "Include business domain, headquarters, and key offerings."
)
 
agent1_output = chain.invoke({"input": agent1_query})
print(agent1_output)
 

# AGENT 2 – Get Stock Market Price (MarketStack)

print("\n================ AGENT 2 =================\n")
 
stock_price_data = None
 
try:
    search_url = "http://api.marketstack.com/v1/tickers"
    search_params = {
        "access_key": MARKETSTACK_API_KEY,
        "search": company_name,
        "limit": 1
    }
 
    ticker_response = requests.get(search_url, params=search_params).json()
    ticker_symbol = ticker_response["data"][0]["symbol"]
 
    eod_url = "http://api.marketstack.com/v1/eod/latest"
    eod_params = {
        "access_key": MARKETSTACK_API_KEY,
        "symbols": ticker_symbol
    }
 
    stock_response = requests.get(eod_url, params=eod_params).json()
    stock_price_data = stock_response["data"][0]
 
    print(f"Company Symbol : {ticker_symbol}")
    print(f"Stock Price    : {stock_price_data['close']}")
    print(f"Exchange       : {stock_price_data['exchange']}")
 
except Exception as e:
    print("Stock API Error:", e)
 

# AGENT 3 – LLM Comprehension & Final Report

print("\n================ AGENT 3 =================\n")
 
agent3_query = f"""
Company Information:
{agent1_output}
 
Stock Market Data:
{stock_price_data}
 
Analyze the company based on business overview and stock price.
Provide a brief business insight and future outlook.
"""
 
agent3_output = chain.invoke({"input": agent3_query})
print(agent3_output)
 
print("\n PROCESS COMPLETED ")