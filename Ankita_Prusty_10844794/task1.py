from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
from typing import TypedDict, Optional, Any
 
#-------------------------------------- 
# ---------- LangChain / LLM ----------
#--------------------------------------

from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#--------------------------------
# ---------- LangGraph ----------
#--------------------------------

from langgraph.graph import StateGraph

#---------------------------------------
# ---------- Optional MongoDB ----------
#---------------------------------------

try:
    from pymongo import MongoClient
    MONGO_ENABLED = True
except ImportError:
    MONGO_ENABLED = False


#---------------------------
# Load environment variables
#---------------------------

load_dotenv()

# Initialize GPT model

llm = init_llm(
    model_name="gpt-4o",
    temperature=0,
    max_tokens=1500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a knowledgeable SAP assistant."),
        ("user", "{question}")
    ]
)

llm_chain = prompt | llm | StrOutputParser()

#--------------------------------------------
# ---------- MongoDB Configuration ----------
#--------------------------------------------

mongo_collection = None
if MONGO_ENABLED:
    try:
        mongo_client = MongoClient(
            "mongodb://localhost:27017/",
            serverSelectionTimeoutMS=3000
        )

        mongo_client.server_info()
        mongo_db = mongo_client["sap_ai"]
        mongo_collection = mongo_db["agent_logs"]
        print("MongoDB connection successful")

    except Exception:
        print(" MongoDB is not available")


#----------------------------------
# ---------- Graph State ----------
#----------------------------------

class WorkflowState(TypedDict):
    step1: Optional[Any]
    step2: Optional[Any]
    step3: Optional[Any]

#--------------------------------------------
# ---------- Agent 1 : Explanation ----------
#--------------------------------------------

def explanation_agent(state: WorkflowState):
    print("\n Executing Agent 1 (LLM Explanation)----------------------")
    question = (
        "Explain step-by-step how to retrieve top sales orders "
        "using API_SALES_ORDER_SRV A_SalesOrder service from SAP API Hub."
    )

    answer = llm_chain.invoke({"question": question})
    print("\n Agent 1 Response:\n", answer)

    if mongo_collection:
        mongo_collection.insert_one({"agent": "agent1", "output": answer})
    state["step1"] = answer
    return state

#----------------------------------------- 
# ---------- Agent 2 : API Call ----------
#-----------------------------------------

def sales_order_agent(state: WorkflowState):
    print("\n Executing Agent 2 (SAP API Call)-----------------------")
    api_url = os.getenv("SAP_API_URL")
    user = os.getenv("SAP_USERNAME")
    pwd = os.getenv("SAP_PASSWORD")
    try:

        api_response = requests.get(
            api_url,
            auth=HTTPBasicAuth(user, pwd),
            headers={"Accept": "application/json"},
            verify=False
        )

        json_data = api_response.json()
        orders = json_data["d"]["results"]
        print("\nAgent 2 Sales Orders:\n")
        parsed_orders = []
        for so in orders:
            entry = {
                "SalesOrder": so.get("SalesOrder"),
                "OrderType": so.get("SalesOrderType"),
                "SalesOrg": so.get("SalesOrganization"),
                "DistChannel": so.get("DistributionChannel"),
                "NetValue": so.get("TotalNetAmount"),
                "Currency": so.get("TransactionCurrency")
            }

            parsed_orders.append(entry)
            print(entry)
        if mongo_collection:
            mongo_collection.insert_one({"agent": "agent2", "output": parsed_orders})
        state["step2"] = json_data
    except Exception as err:
        print(" API failed:", err)
        state["step2"] = None
    return state

#---------------------------------------------------
# ---------- Agent 3 : Simple Explanation ----------
#---------------------------------------------------

def interpretation_agent(state: WorkflowState):
    print("\n Executing Agent 3 (LLM Interpretation)-----------------------------")
    api_data = state.get("step2")
    if not api_data:
        print("No API data available for Agent 3")
        return state
    try:
        first_so = api_data["d"]["results"][0]
        explain_prompt = f"""
        Explain the following SAP Sales Order in simple language:
        {first_so}

        """

        explanation = llm_chain.invoke({"question": explain_prompt})

        print("\nAgent 3 Explanation:\n", explanation)
        if mongo_collection:
            mongo_collection.insert_one({"agent": "agent3", "output": explanation})
        state["step3"] = explanation
    except Exception as ex:
        print("Agent 3 Error:", ex)
    return state

#--------------------------------------
# ---------- Build LangGraph ----------
#--------------------------------------

workflow = StateGraph(WorkflowState)
workflow.add_node("agent1", explanation_agent)
workflow.add_node("agent2", sales_order_agent)
workflow.add_node("agent3", interpretation_agent)
workflow.set_entry_point("agent1")
workflow.add_edge("agent1", "agent2")
workflow.add_edge("agent2", "agent3")
compiled_app = workflow.compile()

#---------------------------------------
# ---------- Execute Workflow ----------
#---------------------------------------

if __name__ == "__main__":
    compiled_app.invoke(
        {
            "step1": None,
            "step2": None,
            "step3": None
        }
    )
 