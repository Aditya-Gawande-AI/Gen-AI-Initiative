from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
from typing import TypedDict, Optional, Any

#Langchain / LLM
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#langGraph
from langgraph.graph import StateGraph

#Optional MongoDB
try:
    from pymongo import MongoClient
    mongo_available = True
except:
    mongo_available = False

#Load ENV Variables
load_dotenv()

#Initialize LLM
model = init_llm(
    "gpt-4o",
    temperature=0,
    max_tokens=1500
)

template = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant."),
    ("user","{input}")
    ])

chain = template | model | StrOutputParser()

#MongoDB setup
collection = None

if mongo_available:
    try:
        client = MongoClient("mongodb://localhost:27017/",serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client["sap_ai"]
        collection = db["results"]
        print("MongoDB Connected")
    except:
        print("MongoDB not Running")

# Define Graph State
class GraphState(TypedDict):
    agent2: Optional[Any]
    agent3: Optional[Any]
    agent1: Optional[Any]

#Node 1 -> AGENT 1(LLM)
def agent1_node(state):
    print("\nRunning Agent 1.............")

    query = """Explain step-by-step how to retrieve top sales orders using API_SALES_ORDER_SRV?A_SalesOrder service in SAP API Business Hub."""

    response = chain.invoke({"input":query})

    print("\nAgent 1 output:\n",response)

    if collection is not None:
        collection.insert_one({"agent":"agent1","data":response})

    state["agent1"] = response
    return state
    
# Node 2 -> AGENT 2(API CALL)
def agent2_node(state):
    print("\nRunning Agent 2.............")

    url = os.getenv("SAP_API_URL")
    username = os.getenv("SAP_USERNAME")
    password = os.getenv("SAP_PASSWORD")

    try:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(username, password),
            headers={"Accept":"application/json"},
            verify=False
        )

        data = response.json()

        print("\nAgent 2 Output:\n")

        sales_orders = data["d"]["results"]

        formatted_orders = []

        for order in sales_orders:
            formatted_order = {
                "salesOrder":order.get("SalesOrder"),
                "Type":order.get("SalesOrderType"),
                "Organization":order.get("SalesOrganization"),
                "Channel":order.get("DistributionChannel"),
                "Amount":order.get("TotalNetAmount"),
                "Currency":order.get("TransactionCurrency")
            }
            formatted_orders.append(formatted_order)
            print(formatted_order)

        if collection is not None:
            collection.insert_one({"agent":"agent2","data":formatted_order})
        
        state["agent2"] = data
    
    except Exception as e:
        print("API Error:",e)
        state["agent2"] = None
    
    return state

#Node 3 -> AGENT 3(LLM Explain)
def agent3_node(state):
    print("\n Running Agent 3.............")

    data = state.get("agent2")

    if data:
        try:
            sales_order = data["d"]["results"]
            first_order = sales_order[0]

            query = f"""Explain this SAP sales order in simple terms: {first_order}"""

            response = chain.invoke({"input":query})

            print("\nAgent 3 Output:\n",response)

            if collection is not None:
                collection.insert_one({"agent":"agent3","data":response})

            state["agent3"] = response
        
        except Exception as e:
            print("Agent 3 Error:",e)

    else:
        print("No data availble for Agent 3")

    return state

# Build Graph
graph = StateGraph(GraphState)

graph.add_node("agent1",agent1_node)
graph.add_node("agent2",agent2_node)
graph.add_node("agent3",agent3_node)

#Define flow
graph.set_entry_point("agent1")
graph.add_edge("agent1","agent2")
graph.add_edge("agent2","agent3")

app = graph.compile()

# Run Graph
if __name__ == "__main__":
    app.invoke({
        "agent1":None,
        "agent2":None,
        "agent3":None
    })