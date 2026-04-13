from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv()

model = init_llm(
    "gpt-4o",
    temperature=0.3,
    max_tokens=1500
)
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

chain = template | model | StrOutputParser()

def agent1(company):

    query = f"""
    Provide clear and structured information about the company "{company}".
    including some details like company name ,industry,core_business (list of strings),
    headquarters,parent_company etc ,in 4-5 lines"""

    result = chain.invoke({"input":query})

    print("\nAgent 1 output: Company Information\n\n",result)

    return result


def agent2(company):

    search = GoogleSerperAPIWrapper()
    query = f"{company} current stock price ticker exchange currency"
    serper_result = search.run(query)

    extract_prompt = f"""
    From the following web search result, extract stock-related information.
    Output MUST follow this exact format and order.
    Do NOT add introductions, explanations, or notes.
    Do NOT include phrases like "Here is", "Summary", or "Note".
    Return ONLY the fields below, nothing else.

    Company name:
    Stock ticker:
    Exchange:
    Approx price:
    Currency:

    Web search result:
    {serper_result}
    """

    structured_stock = chain.invoke({"input": extract_prompt})
    print("\nAgent 2 output: Stock information\n\n", structured_stock)

    return structured_stock


def agent3(company_info, stock_info):
   
    query = f"""
    
        Create a short analytical report using the inputs below.

    Rules:
    - Keep the output upto 5-6 lines
    - Do NOT copy or restate the input text.
    - Paraphrase and add interpretation.
    - Focus on insights, not description
    - Do NOT cut off any section.
    - Combine company context and stock data naturally.
    - Add interpretation, not description.
    - Keep it concise and professional.

    Inputs:
    {company_info}
    {stock_info}

    """

    result = chain.invoke({"input": query})
    print("\nAgent 3: Final Output\n\n", result)
    return result

if __name__=="__main__":

    company=input("Enter Company Name: ")

    info=agent1(company)
    stock=agent2(company)
    final=agent3(info,stock)

    

