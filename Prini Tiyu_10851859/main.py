import os
from dotenv import load_dotenv

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

if not os.getenv("SERPER_API_KEY"):
    raise EnvironmentError("SERPER_API_KEY not found in .env file")

print("Environment variables loaded successfully")

print("Initializing LLM via SAP Generative AI Hub...")

llm = init_llm(
    "gpt-4o",
    temperature=0.3
)

print("LLM initialized\n")

# Agent 1
def agent_1_company_info(company_name: str) -> str:
    try:
        messages = [
            SystemMessage(content="You are a business analyst."),
            HumanMessage(
                content=f"Give an overview, industry, and core business of {company_name}."
            )
        ]
        
        response = llm.invoke(messages)
        return response.content


    except Exception as e:
        return f" Agent 1 failed: {e}"

# Agent 2: Stock Price Agent (Google Serper)
def agent_2_stock_price(company_name: str) -> str:
    try:
        search = GoogleSerperAPIWrapper(
            serper_api_key=os.getenv("SERPER_API_KEY")
        )
        query = f"{company_name} stock price today"
        result = search.run(query)
        return result

    except Exception as e:
        return f" Agent 2 failed: {e}"

# Agent 3: Final Report Agent
def agent_3_final_report(company_name: str, company_info: str, stock_info: str) -> str:
    try:
        messages = [
            SystemMessage(content="You are a financial analyst."),
            HumanMessage(
                content=f"""
Company Name: {company_name}

Company Information:
{company_info}

Stock Market Information:
{stock_info}

Generate a professional, concise summary combining business and stock insights.
"""
            )
        ]
        response = llm.invoke(messages)
        return response.content

    except Exception as e:
        return f" Agent 3 failed: {e}"

# MAIN EXECUTION
if __name__ == "__main__":
    print("    TASK-2 : AGENTIC AI     ")
   
    company_name = input("Enter company name: ").strip()

    print("\n Agent 1: Company Information ")
    company_info = agent_1_company_info(company_name)
    print(company_info)

    print("\n Agent 2: Stock Price ")
    stock_info = agent_2_stock_price(company_name)
    print(stock_info)

    print("\n Agent 3: Final Report ")
    final_report = agent_3_final_report(company_name, company_info, stock_info)
    print(final_report)

    print("\n Task-2 execution completed successfully")