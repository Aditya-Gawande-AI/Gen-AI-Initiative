from dotenv import load_dotenv
import os
import requests
from typing import TypedDict, Optional
 
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph
 
# ------------------ LOAD ENV ------------------
load_dotenv()
 
# ------------------ LLM SETUP ------------------
model = init_llm("gpt-4o", temperature=0.3, max_tokens=1500)
 
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and professional business analyst."),
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
    print("\n Agent 1: Getting Company info........")

    company = state["company"]

    query = f"""
    Provide a concise overview of {company}.
    Include:
    - Industry
    - Core business
    - Global presence
    """
 
    try:
        response = chain.invoke({"input": query})
        state["agent1"] = response
 
        print("\n Agent 1 Output:")
        print(response)
 
    except Exception as e:
        state["agent1"] = f"Error in Agent 1: {e}"
 
    print("\n" + "-"*50)
    return state
 
# ------------------ AGENT 2 ------------------
def agent2_node(state):
    print("\n Agent 2: Fetching stock price.........")

    company = state["company"]
    api_key = os.getenv("api_key")
 
    if not api_key:
        result = " API key missing in .env"
    else:
        try:
            url = f"http://api.marketstack.com/v1/tickers?access_key={api_key}&search={company}"
            res = requests.get(url)
            data = res.json()
 
            if data.get("data"):
                symbol = data["data"][0]["symbol"]
 
                price_url = f"http://api.marketstack.com/v1/eod/latest?access_key={api_key}&symbols={symbol}"
                price_res = requests.get(price_url)
                price_data = price_res.json()
 
                if price_data.get("data"):
                    price = price_data["data"][0]["close"]
                    result = f"Stock price of {company} ({symbol}) is {price}"
                else:
                    result = f"No price data found for {company}"
            else:
                result = f"No stock symbol found for {company}"
 
        except Exception as e:
            result = f"Error fetching stock data: {e}"
 
    state["agent2"] = result
 
    print("\n Agent 2 Output:")
    print(result)
 
    print("\n" + "-"*50)
    return state
 
# ------------------ AGENT 3 ------------------
def agent3_node(state):
    print("\n Agent 3: Final analysis.............")

    info = state.get("agent1", "")
    stock = state.get("agent2", "")
 
    query = f"""
    Analyze the company based on:
    Company Info: {info}
    Stock Info: {stock}
 
    Provide:
    - Brief summary
    - Market position
    - Simple investment insight (1 line)
    """
 
    try:
        response = chain.invoke({"input": query})
        state["agent3"] = response
 
        print("\n🔹 Final Analysis (Agent 3):")
        print(response)
 
    except Exception as e:
        state["agent3"] = f"Error in Agent 3: {e}"
 
    print("\n" + "-"*50)
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
    company_name = input("Enter company name: ").strip()
 
    if not company_name:
        print(" Please enter a valid company name")
        exit()
 
    app.invoke({
        "company": company_name,
        "agent1": None,
        "agent2": None,
        "agent3": None
    })
