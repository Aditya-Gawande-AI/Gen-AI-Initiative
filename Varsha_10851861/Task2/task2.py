
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.utilities import GoogleSerperAPIWrapper

# SAP GenAI Hub / AI Core (as in your setup)
from gen_ai_hub.proxy.langchain.init_models import init_llm


# =========================================================
# 1) ENV SETUP (IMPORTANT: force-load .env from same folder)
# =========================================================
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)


def env_health_check() -> None:
    """Prints only True/False (no secrets) to confirm env is loaded."""
    required = [
        "AICORE_AUTH_URL",
        "AICORE_CLIENT_ID",
        "AICORE_CLIENT_SECRET",
        "AICORE_BASE_URL",
        "AICORE_RESOURCE_GROUP",
        "SERPER_API_KEY",
    ]
    missing = [k for k in required if not os.getenv(k)]
    print("\nENV file used:", str(ENV_PATH))
    if missing:
        print("⚠️ Missing ENV keys:", ", ".join(missing))
    else:
        print("✅ ENV check OK (all required keys found).")


# =========================================================
# 2) MANAGER-STYLE LLM CONFIG (temperature, tokens, etc.)
# =========================================================
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))


def get_llm() -> Optional[object]:
    """
    Creates SAP GenAI Hub LLM with explicit configuration.
    Returns None if init fails (so we still print fallback output instead of crashing).
    """
    base_url = (os.getenv("AICORE_BASE_URL") or "").strip()
    resource_group = (os.getenv("AICORE_RESOURCE_GROUP") or "").strip()

    try:
        # Many environments accept base_url; some accept resource_group too.
        # We'll try with both, then fallback if resource_group isn't supported.
        try:
            return init_llm(
                LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                base_url=base_url,
                resource_group=resource_group
            )
        except TypeError:
            # resource_group not supported in some versions
            return init_llm(
                LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                base_url=base_url
            )

    except Exception as e:
        print("\n⚠️ LLM initialization failed. Running in FALLBACK mode.")
        print("Reason:", str(e))
        return None


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


# =========================================================
# 3) AGENT 1 — Company Information (LLM)
# =========================================================
def agent_1_logic(input_data: dict) -> str:
    company = input_data["company"].strip()
    llm = get_llm()

    if llm is None:
        return (
            f"[FALLBACK MODE]\n"
            f"{company} is a known enterprise operating across multiple business areas. "
            f"To generate a detailed company overview, ensure AI Core configuration works in .env."
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional business analyst."),
        ("user",
         "Explain the company {company} in a detailed and descriptive report style. "
         "Cover background, core business, key segments, market positioning, and future outlook. "
         "Write in clear paragraphs (professional tone).")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"company": company})


agent_1 = RunnableLambda(agent_1_logic)


# =========================================================
# 4) AGENT 2 — Stock Market Summary (Serper Tool + LLM)
# =========================================================
def agent_2_logic(input_data: dict) -> str:
    company = input_data["company"].strip()
    serper_key = (os.getenv("SERPER_API_KEY") or "").strip()

    if not serper_key:
        return (
            f"[FALLBACK MODE]\n"
            f"SERPER_API_KEY is missing, so live stock info cannot be fetched for {company}."
        )

    # Ensure wrapper sees the key (extra-safe)
    os.environ["SERPER_API_KEY"] = serper_key

    # 1) Tool: fetch search results
    try:
        search = GoogleSerperAPIWrapper()
        query = f"{company} stock price NSE"
        raw_results = search.run(query)
    except Exception as e:
        return f"Stock search error (Serper): {e}"

    # 2) LLM: summarize the tool output
    llm = get_llm()
    if llm is None:
        return (
            f"[PARTIAL MODE]\n"
            f"Fetched stock search data but cannot summarize using LLM.\n\n"
            f"Raw search output:\n{raw_results}"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst."),
        ("user",
         "From the following Google search results, extract the latest stock price/market snapshot "
         "for the company and summarize it neatly in 2–4 lines. "
         "If exact price is unclear, provide a best-effort snapshot.\n\n{data}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"data": raw_results})


agent_2 = RunnableLambda(agent_2_logic)


# =========================================================
# 5) AGENT 3 — Final Report (LLM aggregator)
# =========================================================
def agent_3_logic(input_data: dict) -> str:
    company = input_data["company"].strip()
    company_info = input_data["company_info"]
    stock_info = input_data["stock_info"]

    llm = get_llm()
    if llm is None:
        return (
            "========== FINAL REPORT (FALLBACK MODE) ==========\n\n"
            f"Company: {company}\n\n"
            "Company Information (Agent 1):\n"
            f"{company_info}\n\n"
            "Stock Market Summary (Agent 2):\n"
            f"{stock_info}\n\n"
            "Conclusion:\n"
            "This report is generated without LLM due to AI Core initialization issues.\n"
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior business analyst preparing an executive report."),
        ("user",
         "Company Name: {company}\n\n"
         "Agent 1 Output (Company Info):\n{company_info}\n\n"
         "Agent 2 Output (Stock Summary):\n{stock_info}\n\n"
         "Generate a neat final report with headings:\n"
         "1) Overview\n"
         "2) Stock Snapshot\n"
         "3) Combined Insight\n"
         "4) Conclusion\n\n"
         "Write professionally and clearly (descriptive tone).")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "company": company,
        "company_info": company_info,
        "stock_info": stock_info
    })


agent_3 = RunnableLambda(agent_3_logic)


# =========================================================
# 6) MAIN EXECUTION (neat terminal output)
# =========================================================
def main():
    env_health_check()

    company = input("\nEnter company name: ").strip() or "Reliance Industries Limited"

    banner("AGENT 1 OUTPUT (Company Information)")
    company_info = agent_1.invoke({"company": company})
    print(company_info)

    banner("AGENT 2 OUTPUT (Stock Market Summary via Serper)")
    stock_info = agent_2.invoke({"company": company})
    print(stock_info)

    banner("AGENT 3 OUTPUT (FINAL REPORT)")
    final_report = agent_3.invoke({
        "company": company,
        "company_info": company_info,
        "stock_info": stock_info
    })
    print(final_report)

    print("\n✅ Done. Final report generated.\n")


if __name__ == "__main__":
    main()
