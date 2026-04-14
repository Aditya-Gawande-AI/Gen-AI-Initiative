from dotenv import load_dotenv
import requests
from requests.auth import HTTPBasicAuth
import urllib3

# LangChain / SAP GenAI Hub
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Disable SSL warnings (internal SAP system)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -------------------------------------------------
# ENV SETUP
# -------------------------------------------------
load_dotenv()

llm = init_llm("gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful SAP assistant."),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

# =================================================
# AGENT 1 – LLM explains SAP Sales Order API
# =================================================
print("\n================ AGENT 1 ================\n")

agent1_query = (
    "How to retrieve top sales orders using "
    "API_SALES_ORDER_SRV/A_SalesOrder service "
    "from SAP API Business Hub? Explain step by step."
)

agent1_response = chain.invoke({"input": agent1_query})
print(agent1_response)

# =================================================
# AGENT 2 – Call SAP OData API
# =================================================
print("\n================ AGENT 2 ================\n")

url = (
    "https://172.19.151.9:44302"
    "/sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10"
)

try:
    response = requests.get(
        url,
        auth=HTTPBasicAuth("Developer", "dev@S09"),
        headers={"Accept": "application/json"},
        verify=False
    )
    response.raise_for_status()
    sales_data = response.json()
    print(sales_data)

except Exception as e:
    print("Agent 2 Error:", e)
    sales_data = None

# =================================================
# AGENT 3 – Explain one Sales Order
# =================================================
print("\n================ AGENT 3 ================\n")

if sales_data:
    try:
        first_order = sales_data["d"]["results"][0]
        agent3_query = (
            f"Explain this SAP sales order in simple business terms:\n{first_order}"
        )
        agent3_response = chain.invoke({"input": agent3_query})
        print(agent3_response)

    except Exception as e:
        print("Agent 3 Error:", e)
else:
    print("No sales order data available for Agent 3")

print("\n✅ TASK COMPLETED SUCCESSFULLY")