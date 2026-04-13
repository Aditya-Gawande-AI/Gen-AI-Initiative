import os
import requests
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Initialize LLM using GenAI Hub
llm = init_llm(LLM_MODEL)

# --------------------------------------------------
# Agent 1: Company Information Agent
# --------------------------------------------------
def agent1_company_info(company_name: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business research expert."),
        (
            "user",
            "Provide a detailed but concise overview of the company {company}. "
            "Include industry, products/services, headquarters, and global presence."
        )
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"company": company_name})


# --------------------------------------------------
# Agent 2: Stock Market Price Agent (Google Serper)
# --------------------------------------------------
def agent2_stock_price(company_name: str) -> str:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "q": f"{company_name} stock price",
        "num": 5
    }

    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()

    results = response.json().get("organic", [])

    stock_snippets = []
    for r in results[:3]:
        stock_snippets.append(r.get("snippet", ""))

    return "\n".join(stock_snippets)


# --------------------------------------------------
# Agent 3: Comprehension & Reporting Agent
# --------------------------------------------------
def agent3_comprehend_and_report(company_info: str, stock_info: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior financial and business analyst."),
        (
            "user",
            """
            Based on the inputs below, generate a clear business-friendly report.

            COMPANY INFORMATION:
            {company_info}

            STOCK MARKET INFORMATION:
            {stock_info}

            The report should include:
            - Company overview summary
            - Current stock market sentiment
            - Overall business insight (non-investment advice)
            """
        )
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "company_info": company_info,
        "stock_info": stock_info
    })


# --------------------------------------------------
# Main Orchestrator
# --------------------------------------------------
if __name__ == "__main__":
    company_name = input("Enter company name: ")

    print("\n--- Agent 1: Company Information ---")
    company_info = agent1_company_info(company_name)
    print(company_info)

    print("\n--- Agent 2: Stock Market Data ---")
    stock_info = agent2_stock_price(company_name)
    print(stock_info)

    print("\n--- Agent 3: Final Business Report ---")
    final_report = agent3_comprehend_and_report(company_info, stock_info)
    print(final_report)