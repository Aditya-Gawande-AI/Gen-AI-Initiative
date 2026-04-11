from dotenv import load_dotenv
import os
import json
import requests
import urllib3
from typing_extensions import TypedDict, NotRequired
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

load_dotenv()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

llm = init_llm("gpt-4o")

SAP_URL = os.getenv("SAP_URL")
if not SAP_URL:
    raise ValueError("SAP_URL is missing. Check .env and current working directory.")

class GraphState(TypedDict, total=False):
    agent1_response: NotRequired[str]
    orders_list: NotRequired[list[dict]]
    agent2_response: NotRequired[str]
    choice: NotRequired[int]
    selected_order: NotRequired[dict]
    agent3_response: NotRequired[str]

def node_agent1(state: GraphState) -> dict:
    agent1_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SAP technical assistant."),
        ("user", "Explain how to retrieve top sales orders using API_SALES_ORDER_SRV/A_SalesOrder from SAP API Business Hub. Do not bold any text.")
    ])
    agent1_chain = agent1_prompt | llm | StrOutputParser()
    agent1_response = agent1_chain.invoke({})
    print("\nHow to view Top Sales Orders Data:")
    print(agent1_response)
    return {"agent1_response": agent1_response}

def node_fetch_orders(state: GraphState) -> dict:
    response = requests.get(
        SAP_URL,
        auth=(os.getenv("SAP_USERNAME"), os.getenv("SAP_PASSWORD")),
        headers={"Accept": "application/json"},
        verify=False
    )
    response.raise_for_status()
    sales_orders = response.json()
    orders_list = sales_orders["d"]["results"]
    return {"orders_list": orders_list}

def node_agent2(state: GraphState) -> dict:
    orders_list = state["orders_list"]
    print("\nAvailable Sales Orders:")
    for idx, order in enumerate(orders_list, start=1):
        print(f"\nOrder {idx}")
        print(f"  Sales Order   : {order.get('SalesOrder')}")
        print(f"  Sold-To Party : {order.get('SoldToParty')}")
        print(f"  Net Amount    : {order.get('TotalNetAmount')} {order.get('TransactionCurrency')}")
    return {}

def node_get_choice(state: GraphState) -> dict:
    orders_list = state["orders_list"]
    max_n = len(orders_list)
    while True:
        try:
            choice = int(input(f"\nEnter a number between 1 and {max_n} to explain that sales order: "))
            if 1 <= choice <= max_n:
                return {"choice": choice}
            print(f"Please enter a number between 1 and {max_n}.")
        except ValueError:
            print("Please enter a valid numeric value.")

def node_select_order(state: GraphState) -> dict:
    selected_order = state["orders_list"][state["choice"] - 1]
    print("\nSelected Sales Order:")
    print(f"Sales Order Number: {selected_order.get('SalesOrder')}")
    return {"selected_order": selected_order}

def node_agent3(state: GraphState) -> dict:
    agent3_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business analyst. Explain sales orders clearly for non-technical stakeholders."),
        ("user", "Explain the following SAP sales order in bullets but story like, focusing on customer, value, and business meaning. Everything should be without bold.\n{sales_order}")
    ])
    agent3_chain = agent3_prompt | llm | StrOutputParser()
    agent3_response = agent3_chain.invoke({"sales_order": json.dumps(state["selected_order"], indent=2)})
    return {"agent3_response": agent3_response}

def node_print_agent3(state: GraphState) -> dict:
    print(state["agent3_response"])
    return {}

builder = StateGraph(GraphState)
builder.add_node("agent1_explain_api", node_agent1)
builder.add_node("fetch_orders", node_fetch_orders)
builder.add_node("print_orders", node_agent2)
builder.add_node("get_choice", node_get_choice)
builder.add_node("select_order", node_select_order)
builder.add_node("agent3_explain_order", node_agent3)
builder.add_node("print_explanation", node_print_agent3)

builder.add_edge(START, "agent1_explain_api")
builder.add_edge("agent1_explain_api", "fetch_orders")
builder.add_edge("fetch_orders", "print_orders")
builder.add_edge("print_orders", "get_choice")
builder.add_edge("get_choice", "select_order")
builder.add_edge("select_order", "agent3_explain_order")
builder.add_edge("agent3_explain_order", "print_explanation")
builder.add_edge("print_explanation", END)

app = builder.compile()
# print(app.get_graph().draw_mermaid())
app.invoke({})

