from dotenv import load_dotenv
import os

from typing_extensions import TypedDict, NotRequired

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = init_llm("gpt-4o",max_tokens=700)

class GraphState(TypedDict, total=False):
    company_name: str
    agent1_response: NotRequired[str]
    agent2_response: NotRequired[str]
    agent3_response: NotRequired[str]


def node_agent1_company_overview(state: GraphState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business analyst."),
        ("user", "Provide a clear business overview of this company: {company}.")
    ])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"company": state["company_name"]})
    print("\nCompany Overview:")
    print(response)
    return {"agent1_response": response}


def node_agent2_stock_summary(state: GraphState) -> dict:
    search = GoogleSerperAPIWrapper(
        serper_api_key=os.getenv("SERPER_API_KEY")
    )

    response_stock_info = search.run(
        f"{state['company_name']} stock price recent performance market sentiment"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial assistant."),
        ("user", """
        Using the following raw stock information, summarize:
        Current price , Short-term trend, General market sentiment
        Stock data:
        {stock_info}
        """)
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"stock_info": response_stock_info})

    print("\nStock Summary:")
    print(response)

    return {"agent2_response": response}


def node_agent3_final_explanation(state: GraphState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business analyst explaining to non-technical stakeholders."),
        ("user", """
        Company Information:
        {agent1}
        Stock Related Data:
        {agent2}.
        Explain what this means from a business and investment perspective.
        Keep it simple.
        """)
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "agent1": state["agent1_response"],
        "agent2": state["agent2_response"]
    })

    return {"agent3_response": response}


def node_print_final(state: GraphState) -> dict:
    print("\nFinal Explanation:")
    print(state["agent3_response"])
    return {}


builder = StateGraph(GraphState)

builder.add_node("agent1_company_overview", node_agent1_company_overview)
builder.add_node("agent2_stock_summary", node_agent2_stock_summary)
builder.add_node("agent3_final_explanation", node_agent3_final_explanation)
builder.add_node("print_final", node_print_final)

builder.add_edge(START, "agent1_company_overview")
builder.add_edge("agent1_company_overview", "agent2_stock_summary")
builder.add_edge("agent2_stock_summary", "agent3_final_explanation")
builder.add_edge("agent3_final_explanation", "print_final")
builder.add_edge("print_final", END)

app = builder.compile()

company = input("Enter company name: ")
app.invoke({"company_name": company})