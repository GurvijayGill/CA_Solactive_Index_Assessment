from index_model.index import IndexModel
import pandas as pd

if __name__ == "__main__":

    # 1. initialize variables
    shares_outstanding = 1000

    # Change these paths please
    out_path = "C:/Users/Gurvijay/PyCharmProjects_New/CA_Solactive/data_sources/index_values_calcs_output_gurvijay.csv"
    stock_prices_path = "C:/Users/Gurvijay/PyCharmProjects_New/CA_Solactive/data_sources/stock_prices.csv"

    # Dates
    data_start_date = pd.Timestamp("2019-12-31")
    backtest_start = pd.Timestamp("2020-01-01")
    backtest_end = pd.Timestamp("2020-12-31")

    # 2. Load the stock prices csv
    prices_df = pd.read_csv(stock_prices_path)

    # 3. Create market cap data frame, assuming each stock has same shares outstanding.
    prices_df_market_cap = prices_df.copy()
    prices_df_market_cap.loc[:, prices_df_market_cap.columns != "Date"] = prices_df_market_cap.loc[:, prices_df_market_cap.columns != "Date"].mul(shares_outstanding)

    # 4. Re-index data frames to Date column
    prices_df["Date"] = pd.to_datetime(prices_df["Date"], dayfirst=True)
    prices_df_market_cap["Date"] = pd.to_datetime(prices_df_market_cap["Date"], dayfirst=True)
    prices_df = prices_df.set_index("Date").sort_index()
    prices_df_market_cap = prices_df_market_cap.set_index("Date").sort_index()

    # 5. Capture stock universe
    index_universe = list(prices_df.columns)


    # 6. Initialize the IndexModel class and calculate index levels
    index = IndexModel(prices_df, prices_df_market_cap, index_universe)
    levels = index.calc_index_level(data_start_date.date(), backtest_start.date(), backtest_end.date())

    # 7. Save and print the output
    index.export_values(out_path)
    print(f"\nSaved index+weights CSV to: {out_path}")