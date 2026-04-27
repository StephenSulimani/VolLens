import requests


def get_risk_free_rate() -> float:
    """
    Fetches the 3 month risk-free rate from Treasury.gov

    Returns:
        float: Risk-free rate


    Raises:
        Exception: If the request fails or the risk free rate is not found
    """
    response = requests.get(
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates?sort=-record_date&format=json&filter="
    )

    if response.status_code != 200:
        raise Exception(f"Error fetching risk free rate: {response.text}")

    response_json = response.json()["data"]

    for item in response_json:
        if item["security_desc"] == "Treasury Bills":
            return float(item["avg_interest_rate_amt"]) / 100

    raise Exception("Risk free rate not found")
