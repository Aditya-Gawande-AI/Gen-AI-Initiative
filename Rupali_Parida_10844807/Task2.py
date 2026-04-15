from dotenv import load_dotenv
import os

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

llm = init_llm("gpt-4o", max_tokens=512)

company = input("Enter company name: ")

prompt1 = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a senior business analyst with strong knowledge of global companies,
        their operations, industries, products, and market presence.
        Your responses should be factual, structured, and easy to understand.
        """
    ),
    (
        "user",
        """
        Provide basic but meaningful information about the company below.
        Follow these rules strictly:
        - Give exactly 10 bullet points
        - Each bullet should be concise but informative
        - Cover areas such as industry, core business, products/services,
          geographical presence, customers, and overall business model

        Company name: {company_name}
        """
    ),
])

chain1 = prompt1 | llm | StrOutputParser()
agent1_response = chain1.invoke({"company_name": company})

print("\nCompany Overview:")
print(agent1_response)

# Agent-2
search = GoogleSerperAPIWrapper(
    serper_api_key=os.getenv("SERPER_API_KEY")
)

raw_stock_info = search.run(
    f"{company} stock price and recent market performance"
)

prompt2 = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an experienced stock market analyst with deep understanding of
        equity markets, investor sentiment, and short-term performance trends.
        You simplify complex financial information for general audiences.
        """
    ),
    (
        "user",
        """
        Analyze and summarize the stock-related information provided below.

        Your summary must include:
        - Current stock price (only if clearly available in the data)
        - Short-term price or performance trend
        - Overall market or investor sentiment (positive, neutral, or negative)

        Keep the explanation factual, neutral, and easy to understand.
        Do not speculate beyond the given data.

        Raw stock data:
        {raw}
        """
    ),
])

chain2 = prompt2 | llm | StrOutputParser()
agent2_response = chain2.invoke({"raw": raw_stock_info})

print("\nStock Summary:")
print(agent2_response)

# Agent-3 

prompt3 = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a business analyst presenting insights to non-technical
        stakeholders and decision-makers.
        Your goal is clarity, neutrality, and business relevance.
        """
    ),
    (
        "user",
        """
        Based on the information below, explain what it means from a
        business and investment perspective.

        Guidelines:
        - Combine business overview and stock information logically
        - Explain implications, not technical details
        - Keep the tone neutral and informative
        - Use simple language suitable for non-technical stakeholders
        Company Overview: {overview}
        Stock Information:{stock_data}
        """
    ),
])

chain3 = prompt3 | llm | StrOutputParser()
agent3_response = chain3.invoke({
    "overview": agent1_response,
    "stock_data": agent2_response
})

print("\n Combined Explanation:")
print(agent3_response)
