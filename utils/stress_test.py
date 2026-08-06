import pandas as pd


def stress_test_portfolio(holdings, scenario):

    stressed = holdings.copy()

    stressed["Scenario Price"] = (
        stressed["Current Price"] * (1 + scenario)
    )

    stressed["Scenario Value"] = (
        stressed["Scenario Price"] * stressed["Shares"]
    )

    stressed["Loss"] = (
        stressed["Scenario Value"]
        - stressed["Market Value"]
    )

    return stressed