

from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def main():
    print("\n==============================================")
    print(" SAP LangChain Multi-Agent Task")
    print("==============================================\n")

    print("▶ Agent 1: API Knowledge (LLM Powered)\n")

    # ✅ Fake LLM to simulate LLM behaviour (no internet / no API key needed)
    llm = FakeListLLM(responses=[
        """
        The API_SALES_ORDER_SRV is an SAP OData service used to fetch Sales Order data.
        The entity A_SalesOrder represents sales order header records.

        To retrieve the top sales orders, use the $top query option:
        Example:
        /sap/opu/odata/sap/API_SALES_ORDER_SRV/A_SalesOrder?$top=10

        Other useful query options:
        - $select: choose required fields
        - $filter: filter records by conditions
        - $orderby: sort the results
        """
    ])

    # ✅ PromptTemplate (LangChain concept)
    prompt = PromptTemplate(
        input_variables=[],
        template="Explain how to retrieve top sales orders using SAP OData API."
    )

    # ✅ LLMChain (this is what makes it a LangChain project)
    chain = LLMChain(llm=llm, prompt=prompt)

    # ✅ Run LLM
    response = chain.run()

    print(response)
    print("✅ Agent 1 completed.\n")

if __name__ == "__main__":
    main()
