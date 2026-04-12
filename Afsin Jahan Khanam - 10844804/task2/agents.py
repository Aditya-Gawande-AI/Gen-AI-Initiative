from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.init_models import init_llm
from langchain_core.prompts import PromptTemplate
from tools import get_stock_price

load_dotenv()


model = init_llm("gpt-4o")

#agent-1
def agent_company_info(company_name):
    prompt = PromptTemplate(
        input_variables=["company"],
        template="""
        Explain what the company {company} does.
        Include industry and headquarters.
        """
    )

    prompt_text = prompt.format(company=company_name)
    response = model.invoke(prompt_text)   
    return response.content                


#agent-2
def agent_stock_price(company_name):
    symbol_prompt = f"""
    Identify the stock ticker symbol for the company '{company_name}'.
    If not publicly traded, say 'Not Public'.
    Only return the symbol.
    """

    symbol_response = model.invoke(symbol_prompt)
    symbol = symbol_response.content.strip()

    if symbol.lower() == "not public":
        return f"{company_name} is not publicly traded."

    stock_data = get_stock_price(symbol)
    return f"Stock Symbol: {symbol}\nStock Data: {stock_data}"


#agent-3
def agent_final_report(company, info, stock):
    prompt = PromptTemplate(
        input_variables=["company", "info", "stock"],
        template="""
        Company Information:
        {info}

        Stock Information:
        {stock}

        Explain both together in a clear summary.
        """
    )

    prompt_text = prompt.format(
        company=company,
        info=info,
        stock=stock
    )

    response = model.invoke(prompt_text)   
    return response.content                
