import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper

from gen_ai_hub.proxy.langchain.init_models import init_llm


# Load environment variables

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# Initialize LLM

llm = init_llm("gpt-4o", max_tokens=500)


# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Keep the answer concise and factual."),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

# AGENT 1 – Company Overview

print("\n--------------- AGENT‑1 ---------------\n")

company_name = input("Enter the company name: ")

agent1_output = chain.invoke({
    "input": f"Briefly explain the company {company_name}"
})

print("\nCompany Overview:\n")
print(agent1_output)


# AGENT 2 – Stock Market Price

print("\n--------------- AGENT‑2 (STOCK PRICE) ---------------\n")

# Create Serper search tool
search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

raw_search_results = search.run(
    f"{company_name} stock price ticker exchange latest price"
)

stock_info = chain.invoke({
    "input": (
        f"From these search results, extract the latest stock price info for {company_name}.\n\n"
        f"{raw_search_results}\n\n"
        "IMPORTANT RULES:\n"
        "- If the company is Indian, return price in INR and exchange as NSE or BSE\n"
        "- If the company is US-based, return price in USD\n\n"
        "Return the result in this exact format:\n"
        "Company Name:\n"
        "Price:\n"
        "Currency:\n"
        "Exchange:\n"
    )
})

print("\nStock Price Information:\n")
print(stock_info)

# --------------------------------------------------
# AGENT 3 – Final Combined Report
# --------------------------------------------------
print("\n--------------- AGENT‑3 (FINAL REPORT) ---------------\n")

final_report = chain.invoke({
    "input": f"""
Company Name: {company_name}

Company Overview:
{agent1_output}

Stock Information:
{stock_info}

Create a final business report:
1) Short company summary
2) Stock price report
3) Details summary
"""
})

print("\nFinal Consolidated Report:\n")
print(final_report)
