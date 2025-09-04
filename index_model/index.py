import pandas as pd
import numpy as np
import datetime as dt
from pandas.tseries.offsets import BDay

# set the display option to max
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

class IndexModel:
    """
    Solactive Notes :
    The index is a total return index.
    The index universe consists of all stocks from "Stock_A" to including "Stock_J".
    Every first business day of a month the index selects from the universe the top three stocks based on their market capitalization, based on the close of business values as of the last business day of the immediately preceding month.
    The selected stock with the highest market capitalization gets assigned a 50% weight, while the second and third each get assigned 25%.
    The selection becomes effective close of business on the first business date of each month.
    The index starts with a level of 100.
    The index start date is January 1st 2020.
    The index business days are Monday to Friday.
    There are no additional holidays.
    """

    def __init__(self, prices: pd.DataFrame,
                 market_caps: pd.DataFrame | None = None,
                 universe: list[str] | None = None,
                 index_start_level: float = 100.0,
                 data_start_date: dt.date = dt.date(2019, 12, 31),
                 index_start_date: dt.date = dt.date(2020, 1, 1),
                 index_end_date: dt.date = dt.date(2020, 12, 31),
    ) -> None:
        self.prices_raw = prices.copy()
        self.market_caps_raw = market_caps.copy() if market_caps is not None else None
        self.index_start_level = float(index_start_level)
        self.data_start_date = pd.to_datetime(data_start_date)
        self.index_start_date = pd.to_datetime(index_start_date)
        self.index_end_date = pd.to_datetime(index_end_date)
        self.universe = universe

        self._clean_market_caps()
        self.weights_daily: pd.DataFrame | None = None
        self.index_levels: pd.Series | None = None

    # ---------- Public API ----------
    def calc_index_level(self, data_start_date: dt.date, index_start_date: dt.date, end_date: dt.date) -> pd.Series:
        """
        Computes and returns the daily index levels over [start_date, end_date].
        Also stores the result in self.index_levels and the weights in self.weights_daily.

        """
        data_start = pd.to_datetime(data_start_date)
        index_start = pd.to_datetime(index_start_date)
        end = pd.to_datetime(end_date)

        # 1. align to available price dates (business days only)

        # 1.a get the business days between the start and end date
        dates = self._business_days_between(data_start, end)

        # 1.b - Important step that does the following :
        # --> re-align the original input stock prices data frame with the dates times series generated in above step.
        # --> forward fill prices for missing dates
        # --> drop na values when all the column values are Nan for a specific row.
        prices = self.prices.reindex(dates).ffill().dropna(how="all")
        dates = prices.index


        # 2. call the class method to calculate weight for the date, pass the index start date and end date.
        self.weights_daily = self.calc_constituent_weights(start_date=index_start, end_date=end)


        # daily returns (close-to-close)
        daily_stock_returns = prices.pct_change().fillna(0.0)

        # portfolio daily returns: dot of weights_t with returns_t
        aligned_weights = self.weights_daily.reindex(daily_stock_returns.index).ffill().fillna(0.0)
        port_rets = (aligned_weights * daily_stock_returns).sum(axis=1)
        port_rets = port_rets.iloc[2:]

        # build index
        base_level_date = dates[1]
        base_level = self.index_start_level if base_level_date == self.index_start_date else self.index_start_level

        # If the very first date isn't exactly the formal start date, we still start the series at 'base_level'
        levels = (1.0 + port_rets).cumprod() * base_level
        levels.loc[index_start] = base_level  # add (or overwrite) the value
        levels = levels.sort_index()
        self.index_levels = levels

        return levels

    def calc_constituent_weights(self, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """
        Returns a DataFrame of daily weights for each constituent over [start_date, end_date].
        Implements:
          - Selection date: first business day (FBD) each month
          - Ranking date: last business day (LBD) of prior month (or nearest available earlier date)
          - Weights effective AFTER close of FBD => from (FBD + 1 BDay) onward
          - Initial period (up to and including first FBD): use prior-month selection
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)


        # Align to business days & price coverage
        dates = self._business_days_between(start, end)
        prices = self.prices.reindex(dates).ffill().dropna(how="all")
        dates = prices.index

        # Prepare container
        weights = pd.DataFrame(0.0, index=dates, columns=self.universe, dtype=float)

        # Build monthly schedule of first business days within [start, end]
        month_starts = pd.to_datetime(sorted({dt.date(d.year, d.month, 1) for d in dates}))
        fbd_per_month = []
        for ms in month_starts:
            # first business day >= month start and within our dates
            fbd = ms
            while fbd not in dates:
                fbd += BDay(1)
                if fbd > dates[-1]:
                    break
            if fbd in dates:
                fbd_per_month.append(pd.Timestamp(fbd))

        if not fbd_per_month:
            # No monthly boundaries inside range => assign single static weight based on prior month
            initial_weights = self._weights_from_prior_month_selection(fallback_date=dates[0])
            weights.loc[:, :] = 0.0
            for s, w in initial_weights.items():
                if s in weights.columns:
                    weights.loc[:, s] = w
            return weights

        # Initial weights apply from start through first FBD (inclusive)
        first_fbd = fbd_per_month[0]
        initial_weights = self._weights_from_prior_month_selection(fallback_date=first_fbd)
        weights.loc[:first_fbd, :] = 0.0
        for s, w in initial_weights.items():
            if s in weights.columns:
                weights.loc[:first_fbd, s] = w

        # For each month FBD, compute weights effective from FBD+1 BDay through next FBD (inclusive)
        for i, fbd in enumerate(fbd_per_month):
            eff_start = self._next_business_day(fbd, dates)
            if eff_start is None:
                continue  # no day after FBD in our window

            if i + 1 < len(fbd_per_month):
                next_fbd = fbd_per_month[i + 1]
                eff_end = next_fbd  # inclusive
            else:
                eff_end = dates[-1]

            w_month = self._weights_from_prior_month_selection(fallback_date=fbd)
            # Assign
            idx_slice = (weights.index >= eff_start) & (weights.index <= eff_end)
            weights.loc[idx_slice, :] = 0.0
            for s, w in w_month.items():
                if s in weights.columns:
                    weights.loc[idx_slice, s] = w

        return weights

    def export_values(self, file_name: str) -> None:
        """
        Exports a CSV with columns: Date, Ticker, Weight, Index_Level.
        If index levels or weights haven't been computed, this function computes them
        from the earliest available date >= 2020-01-01 through the last available date.

        """
        # 1. Get index weights previously calculated
        output_weights = self.weights_daily.copy()
        output_weights["Date"] = output_weights.index

        # 2. Get Index levels previously calculated
        output_levels = self.index_levels.rename("Index_Level").reset_index().rename(columns={"index": "Date"})

        # 3. Merge levels and weights
        output_sheet = output_weights.merge(output_levels, on="Date", how="left").sort_values(["Date"])
        output_sheet.to_csv(file_name, index=False)



    # ---------- Private methods that are intended to be internal helpers ----------
    def _weights_from_prior_month_selection(self, fallback_date: pd.Timestamp) -> dict[str, float]:
        """
        Determine top 3 by market cap as of the LAST business day of the prior month (relative to fallback_date).
        If unavailable, fall back to the most recent date strictly before fallback_date.
        If still unavailable (e.g., very start), use the earliest available date.
        Returns a dict {ticker: weight}.
        """
        # Find the prior month last business day (within available dates)
        last_business_day = self._prior_month_last_business_day(fallback_date)
        if last_business_day is None:
            last_business_day = self._last_business_day_before(fallback_date) or self.prices.index[0]

        # Ranking snapshot: market caps if available, else prices
        snapshot = (
            self.market_caps.loc[last_business_day] if self.market_caps is not None else self.prices.loc[last_business_day]
        )

        # Rank descending, ignore missing
        ranked = snapshot.dropna().sort_values(ascending=False)
        top3 = list(ranked.index[:3])

        # Assign weights
        w = {ticker: 0.0 for ticker in self.universe}
        if len(top3) >= 1:  w[top3[0]] = 0.50
        if len(top3) >= 2:  w[top3[1]] = 0.25
        if len(top3) >= 3:   w[top3[2]] = 0.25
        return w


    def _business_days_between(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        '''

        :param start: start date
        :param end: end date
        :return: pandas date index of business days between start and end

        '''
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if start > end:
            raise ValueError("start_date must be <= end_date")

        # Mon-Fri only, no additional holidays, however you can supply your own holiday calendar in the frequency parameter
        return pd.bdate_range(start=start, end=end, freq="C")

    def _last_business_day_before(self, ref_date: pd.Timestamp) -> pd.Timestamp | None:
        # Find the last available date strictly before ref_date in self.prices index
        idx = self.prices.index
        pos = idx.searchsorted(ref_date, side="left")
        if pos == 0:
            return None
        return idx[pos - 1]

    def _prior_month_last_business_day(self, fbd_of_month: pd.Timestamp) -> pd.Timestamp | None:
        # Last business day of prior month based on our available dates
        # Start from the day before FBD and step backwards to find the first available date in the prior month
        idx = self.prices.index
        pos = idx.get_indexer_for([fbd_of_month])[0]
        # Step back at least one day
        i = pos - 1
        while i >= 0:
            d = idx[i]
            if d.month != fbd_of_month.month:
                # keep going until we hit the last date that is still in the prior month
                return d
            i -= 1
        return None

    def _next_business_day(self, ref_date: pd.Timestamp, universe_index: pd.DatetimeIndex) -> pd.Timestamp | None:
        pos = universe_index.get_indexer_for([ref_date])[0]
        if pos + 1 < len(universe_index):
            return universe_index[pos + 1]
        return None

    def _clean_market_caps(self) -> None:
        '''
        Purpose of this method is to cleanse the Market Caps, if there is a mismatch of universe between supplied market caps and stock price universe, we create new columns
        for that stock and insert it into market_data data frame and finally initialize the parameter.
        :return:
        '''

        data_df = self.prices_raw.copy()
        start = self.data_start_date

        # calculate business dates
        dates = self._business_days_between(start, data_df.index.max())

        df = data_df.reindex(dates).ffill()
        self.prices = data_df

        # make a data frane copy and reindex to the business dates
        market_caps = self.market_caps_raw.copy()
        market_caps = market_caps.reindex(dates).ffill()

        # Check if the columns in universe of stocks and their market caps are the same if not then fill with Nan
        for stock_col in self.universe:
            if stock_col not in market_caps.columns:
                market_caps[stock_col] = np.nan
        self.market_caps = market_caps[self.universe]



