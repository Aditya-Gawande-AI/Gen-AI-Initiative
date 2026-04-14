

import json
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

INPUT_FILE = "agent2_output.json"

def main():
    print("\n▶ Agent 3: Explain Sales Order (LLM Powered)\n")

    # Load sales order from Agent 2 output
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        orders = json.load(f)

    if not orders:
        print("❌ No sales orders available.")
        return

    sales_order = orders[0]

    # ✅ Fake LLM simulating explanation
    llm = FakeListLLM(responses=[
        f"""
        This Sales Order represents a customer order processed in SAP.

        Sales Order Number: {sales_order.get('SalesOrder')}
        Sold-To Party: {sales_order.get('SoldToParty')}
        Sales Organization: {sales_order.get('SalesOrganization')}
        Net Amount: {sales_order.get('TotalNetAmount')}

        This order is used for delivery, billing, and revenue tracking.
        """
    ])

    # ✅ PromptTemplate
    prompt = PromptTemplate(
        input_variables=[],
        template="Explain the given SAP sales order in business terms."
    )

    # ✅ LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    explanation = chain.run()

    print(explanation)
    print("✅ Agent 3 completed.\n")

if __name__ == "__main__":
    main()
