from dotenv import load_dotenv
import os
import requests
from typing import TypedDict, Optional
 
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
 
from langgraph.graph import StateGraph
 
# Load ENV
load_dotenv()
 
# LLM
model = init_llm("gpt-4o", temperature=0.3, max_tokens=1500)
 
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
 
chain = template | model | StrOutputParser()
 
# ------------------ STATE ------------------
class GraphState(TypedDict):
    company: str
    agent1: Optional[str]
    agent2: Optional[str]
    agent3: Optional[str]
 
# ------------------ AGENT 1 ------------------
def agent1_node(state):
    print("\n🔹 Agent 1: Getting company info...")
    company = state["company"]
    query = f"Give a short overview of {company} company."
    response = chain.invoke({"input": query})
    state["agent1"] = response
    print("Agent1 Output:", response)
    return state
 
# ------------------ AGENT 2 ------------------
def agent2_node(state):
    print("\n🔹 Agent 2: Fetching stock price...")
    company = state["company"]
    api_key = os.getenv("MARKETSTACK_API_KEY")
 
    try:
        url = f"http://api.marketstack.com/v1/tickers?access_key={api_key}&search={company}"
        res = requests.get(url)
        data = res.json()
 
        if data["data"]:
            symbol = data["data"][0]["symbol"]
            price_url = f"http://api.marketstack.com/v1/eod/latest?access_key={api_key}&symbols={symbol}"
            price_res = requests.get(price_url)
            price_data = price_res.json()
            price = price_data["data"][0]["close"]
            result = f"Stock price of {company} ({symbol}) is {price}"
        else:
            result = f"No stock data found for {company}"
 
    except Exception as e:
        result = f"Error fetching stock data: {e}"
 
    state["agent2"] = result
    print("Agent2 Output:", result)
    return state
 
# ------------------ AGENT 3 ------------------
def agent3_node(state):
    print("\n🔹 Agent 3: Final analysis...")
 
    info = state["agent1"]
    stock = state["agent2"]
 
    query = f"""
    Based on the following:
    Company Info: {info}
    Stock Info: {stock}
 
    Give a final summary in simple words.
    """
 
    response = chain.invoke({"input": query})
 
    state["agent3"] = response
    print("Final Output:", response)
 
    return state
 
# ------------------ GRAPH ------------------
graph = StateGraph(GraphState)
 
graph.add_node("agent1", agent1_node)
graph.add_node("agent2", agent2_node)
graph.add_node("agent3", agent3_node)
 
graph.set_entry_point("agent1")
 
graph.add_edge("agent1", "agent2")
graph.add_edge("agent2", "agent3")
 
app = graph.compile()
 
# ------------------ RUN ------------------
if __name__ == "__main__":
    company_name = input("Enter company name: ")
 
    app.invoke({
        "company": company_name,
        "agent1": None,
        "agent2": None,
        "agent3": None
    })
 