from agents import (
    agent_company_info,
    agent_stock_price,
    agent_final_report
)

def main():
    company_name = input("Enter company name: ")

    print("\nAgent 1 working...")
    company_info = agent_company_info(company_name)
    print("\nCompany Info:\n", company_info)

    print("\nAgent 2 working...")
    stock_info = agent_stock_price(company_name)
    print("\nStock Info:\n", stock_info)

    print("\nAgent 3 working...")
    final_output = agent_final_report(
        company_name,
        company_info,
        stock_info
    )

    print("\nFINAL OUTPUT:\n")
    print(final_output)


if __name__ == "__main__":
    main()
