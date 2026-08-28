"""
Point-in-time universe constituent manager, alias resolution, and sector mapping.

Point-in-time membership tables track additions and deletions with effective
dates so historical backtests avoid reading modern index lists backwards in time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Set

import pandas as pd

# Comprehensive alias map for renames, mergers, and demergers on NSE
KNOWN_RENAMES: Dict[str, str] = {
    "TATAMOTORS": "TMPV",
    "ZOMATO": "ETERNAL",
    "MOTHERSUMI": "MOTHERSON",
    "CADILAHC": "ZYDUSLIFE",
    "MINDTREE": "LTIM",
    "LTI": "LTIM",
    "GMRINFRA": "GMRAIRPORT",
    "L&TFH": "LTF",
    "SRTRANSFIN": "SHRIRAMFIN",
    "HDFC": "HDFCBANK",
    "ADANITRANS": "ADANIENSOL",
    "IBULHSGFIN": "SAMMAANCAP",
    "JUBLFOOD": "JUBLFOOD",
    "HEXAWARE": "HEXAWARE",
    "DHFL": "DHFL",
    "YESBANK": "YESBANK",
    "RCOM": "RCOM",
    "SUZLON": "SUZLON",
    "UNITECH": "UNITECH",
    "JPASSOCIAT": "JPASSOCIAT",
    "SINTEX": "SINTEX",
    "BALLARPUR": "BALLARPUR",
    "RELINFRA": "RELINFRA",
    "RNRL": "RNRL",
    "RPOWER": "RPOWER",
    "ABIRLANUVO": "GRASIM",
    "IDFC": "IDFCFIRSTB",
}

# Default NIFTY 100/200 point-in-time historical constituent records (sample dates)
# date_added is None if in index since before start of backtest window (~2014)
# date_removed is None if currently active
DEFAULT_PIT_CONSTITUENTS = [
    {"symbol": "RELIANCE", "date_added": "2000-01-01", "date_removed": None, "sector": "Energy"},
    {"symbol": "TCS", "date_added": "2004-08-25", "date_removed": None, "sector": "IT"},
    {"symbol": "HDFCBANK", "date_added": "2000-01-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "ICICIBANK", "date_added": "2000-01-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "INFY", "date_added": "2000-01-01", "date_removed": None, "sector": "IT"},
    {"symbol": "HINDUNILVR", "date_added": "2000-01-01", "date_removed": None, "sector": "FMCG"},
    {"symbol": "ITC", "date_added": "2000-01-01", "date_removed": None, "sector": "FMCG"},
    {"symbol": "SBIN", "date_added": "2000-01-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "BHARTIARTL", "date_added": "2002-02-18", "date_removed": None, "sector": "Telecom"},
    {"symbol": "KOTAKBANK", "date_added": "2010-04-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "LT", "date_added": "2000-01-01", "date_removed": None, "sector": "Capital Goods"},
    {"symbol": "AXISBANK", "date_added": "2009-03-27", "date_removed": None, "sector": "Financials"},
    {"symbol": "BAJFINANCE", "date_added": "2017-09-29", "date_removed": None, "sector": "Financials"},
    {"symbol": "ASIANPAINT", "date_added": "2012-04-27", "date_removed": None, "sector": "Consumer"},
    {"symbol": "MARUTI", "date_added": "2004-06-04", "date_removed": None, "sector": "Auto"},
    {"symbol": "SUNPHARMA", "date_added": "2002-05-31", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "TITAN", "date_added": "2018-04-02", "date_removed": None, "sector": "Consumer"},
    {"symbol": "HCLTECH", "date_added": "2003-05-30", "date_removed": None, "sector": "IT"},
    {"symbol": "WIPRO", "date_added": "2000-01-01", "date_removed": None, "sector": "IT"},
    {"symbol": "NTPC", "date_added": "2005-09-23", "date_removed": None, "sector": "Utilities"},
    {"symbol": "ONGC", "date_added": "2000-01-01", "date_removed": None, "sector": "Energy"},
    {"symbol": "POWERGRID", "date_added": "2008-03-28", "date_removed": None, "sector": "Utilities"},
    {"symbol": "M&M", "date_added": "2000-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "TATAMOTORS", "date_added": "2000-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "TMPV", "date_added": "2024-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "COALINDIA", "date_added": "2011-04-01", "date_removed": None, "sector": "Metals & Mining"},
    {"symbol": "TATASTEEL", "date_added": "2000-01-01", "date_removed": None, "sector": "Metals & Mining"},
    {"symbol": "JSWSTEEL", "date_added": "2018-09-28", "date_removed": None, "sector": "Metals & Mining"},
    {"symbol": "HINDALCO", "date_added": "2000-01-01", "date_removed": None, "sector": "Metals & Mining"},
    {"symbol": "VEDL", "date_added": "2000-01-01", "date_removed": "2020-03-27", "sector": "Metals & Mining"},
    {"symbol": "GRASIM", "date_added": "2000-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "ULTRACEMCO", "date_added": "2012-09-28", "date_removed": None, "sector": "Materials"},
    {"symbol": "TECHM", "date_added": "2014-03-28", "date_removed": None, "sector": "IT"},
    {"symbol": "LTIM", "date_added": "2022-12-01", "date_removed": None, "sector": "IT"},
    {"symbol": "BAJAJFINSV", "date_added": "2018-09-28", "date_removed": None, "sector": "Financials"},
    {"symbol": "BAJAJ-AUTO", "date_added": "2010-10-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "NESTLEIND", "date_added": "2019-09-27", "date_removed": None, "sector": "FMCG"},
    {"symbol": "BRITANNIA", "date_added": "2019-04-01", "date_removed": None, "sector": "FMCG"},
    {"symbol": "DABUR", "date_added": "2005-01-01", "date_removed": None, "sector": "FMCG"},
    {"symbol": "GODREJCP", "date_added": "2010-01-01", "date_removed": None, "sector": "FMCG"},
    {"symbol": "CIPLA", "date_added": "2000-01-01", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "DRREDDY", "date_added": "2000-01-01", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "APOLLOHOSP", "date_added": "2022-03-31", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "DIVISLAB", "date_added": "2020-09-25", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "EICHERMOT", "date_added": "2016-04-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "HEROMOTOCO", "date_added": "2000-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "INDUSINDBK", "date_added": "2013-04-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "SBILIFE", "date_added": "2020-09-25", "date_removed": None, "sector": "Financials"},
    {"symbol": "HDFCLIFE", "date_added": "2020-03-27", "date_removed": None, "sector": "Financials"},
    {"symbol": "ADANIENT", "date_added": "2022-09-30", "date_removed": None, "sector": "Metals & Mining"},
    {"symbol": "ADANIPORTS", "date_added": "2015-09-28", "date_removed": None, "sector": "Services"},
    {"symbol": "BPCL", "date_added": "2000-01-01", "date_removed": None, "sector": "Energy"},
    {"symbol": "IOC", "date_added": "2000-01-01", "date_removed": "2022-03-31", "sector": "Energy"},
    {"symbol": "GAIL", "date_added": "2000-01-01", "date_removed": "2021-03-31", "sector": "Energy"},
    {"symbol": "BEL", "date_added": "2015-01-01", "date_removed": None, "sector": "Capital Goods"},
    {"symbol": "TRENT", "date_added": "2015-01-01", "date_removed": None, "sector": "Consumer"},
    {"symbol": "HAL", "date_added": "2018-03-28", "date_removed": None, "sector": "Capital Goods"},
    {"symbol": "SOLARINDS", "date_added": "2016-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "TATAELXSI", "date_added": "2015-01-01", "date_removed": None, "sector": "IT"},
    {"symbol": "DIXON", "date_added": "2017-09-18", "date_removed": None, "sector": "Consumer"},
    {"symbol": "POLYCAB", "date_added": "2019-04-16", "date_removed": None, "sector": "Capital Goods"},
    {"symbol": "PERSISTENT", "date_added": "2015-01-01", "date_removed": None, "sector": "IT"},
    {"symbol": "KPITTECH", "date_added": "2019-04-22", "date_removed": None, "sector": "IT"},
    {"symbol": "COFORGE", "date_added": "2015-01-01", "date_removed": None, "sector": "IT"},
    {"symbol": "CHOLAFIN", "date_added": "2015-01-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "ASTRAL", "date_added": "2015-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "DEEPAKNTR", "date_added": "2015-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "TIINDIA", "date_added": "2017-11-03", "date_removed": None, "sector": "Auto"},
    {"symbol": "SUPREMEIND", "date_added": "2015-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "JUBLFOOD", "date_added": "2015-01-01", "date_removed": None, "sector": "Consumer"},
    {"symbol": "TVSMOTOR", "date_added": "2015-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "PIDILITIND", "date_added": "2015-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "HAVELLS", "date_added": "2015-01-01", "date_removed": None, "sector": "Consumer"},
    {"symbol": "SRF", "date_added": "2015-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "MAXHEALTH", "date_added": "2020-08-21", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "PAGEIND", "date_added": "2015-01-01", "date_removed": None, "sector": "Consumer"},
    {"symbol": "BOSCHLTD", "date_added": "2015-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "CUMMINSIND", "date_added": "2015-01-01", "date_removed": None, "sector": "Capital Goods"},
    {"symbol": "FEDERALBNK", "date_added": "2015-01-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "IDFCFIRSTB", "date_added": "2015-11-06", "date_removed": None, "sector": "Financials"},
    {"symbol": "ASHOKLEY", "date_added": "2015-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "OBEROIRLTY", "date_added": "2015-01-01", "date_removed": None, "sector": "Realty"},
    {"symbol": "VOLTAS", "date_added": "2015-01-01", "date_removed": None, "sector": "Consumer"},
    {"symbol": "TORNTPHARM", "date_added": "2015-01-01", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "AUROPHARMA", "date_added": "2015-01-01", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "LUPIN", "date_added": "2015-01-01", "date_removed": None, "sector": "Healthcare"},
    {"symbol": "CONCOR", "date_added": "2015-01-01", "date_removed": None, "sector": "Services"},
    {"symbol": "NAUKRI", "date_added": "2015-01-01", "date_removed": None, "sector": "IT"},
    {"symbol": "MUTHOOTFIN", "date_added": "2015-01-01", "date_removed": None, "sector": "Financials"},
    {"symbol": "PIIND", "date_added": "2015-01-01", "date_removed": None, "sector": "Materials"},
    {"symbol": "BALKRISIND", "date_added": "2015-01-01", "date_removed": None, "sector": "Auto"},
    {"symbol": "GODREJPROP", "date_added": "2015-01-01", "date_removed": None, "sector": "Realty"},
    {"symbol": "SHREECEM", "date_added": "2020-03-27", "date_removed": "2022-09-30", "sector": "Materials"},
    {"symbol": "INFRATEL", "date_added": "2015-03-27", "date_removed": "2020-09-25", "sector": "Telecom"},
    {"symbol": "INDUSTOWER", "date_added": "2020-11-20", "date_removed": None, "sector": "Telecom"},
    {"symbol": "YESBANK", "date_added": "2015-03-27", "date_removed": "2020-03-27", "sector": "Financials"},
    {"symbol": "DHFL", "date_added": "2014-04-01", "date_removed": "2019-06-28", "sector": "Financials"},
    {"symbol": "IBULHSGFIN", "date_added": "2017-09-29", "date_removed": "2019-09-27", "sector": "Financials"},
    {"symbol": "ZEEL", "date_added": "2010-04-01", "date_removed": "2020-03-27", "sector": "Media"},
    {"symbol": "RCOM", "date_added": "2006-03-31", "date_removed": "2014-03-28", "sector": "Telecom"},
    {"symbol": "SUZLON", "date_added": "2006-03-31", "date_removed": "2012-09-28", "sector": "Capital Goods"},
    {"symbol": "JPASSOCIAT", "date_added": "2007-09-28", "date_removed": "2014-09-26", "sector": "Materials"},
    {"symbol": "UNITECH", "date_added": "2007-03-30", "date_removed": "2011-04-01", "sector": "Realty"},
    {"symbol": "RELINFRA", "date_added": "2000-01-01", "date_removed": "2015-03-27", "sector": "Utilities"},
    {"symbol": "RPOWER", "date_added": "2008-03-28", "date_removed": "2014-03-28", "sector": "Utilities"},
]


class PointInTimeUniverse:
    """Manages point-in-time constituents, historical renames, and sector labels."""

    def __init__(self, table: Optional[List[dict]] = None):
        self.table = pd.DataFrame(table or DEFAULT_PIT_CONSTITUENTS)
        self.table["date_added"] = pd.to_datetime(self.table["date_added"])
        self.table["date_removed"] = pd.to_datetime(self.table["date_removed"])
        self._sector_map = dict(zip(self.table["symbol"], self.table["sector"].fillna("Unmapped")))

    def resolve_symbol(self, symbol: str) -> str:
        """Map legacy or demerged names to their traded ticker."""
        s = symbol.strip().upper()
        return KNOWN_RENAMES.get(s, s)

    def members_asof(self, asof_date: pd.Timestamp) -> List[str]:
        """Return symbols that were ACTIVE constituents on `asof_date`."""
        ts = pd.to_datetime(asof_date)
        added_mask = self.table["date_added"].isna() | (self.table["date_added"] <= ts)
        removed_mask = self.table["date_removed"].isna() | (self.table["date_removed"] > ts)
        active = self.table.loc[added_mask & removed_mask, "symbol"].unique()
        resolved = [self.resolve_symbol(s) for s in active]
        return list(dict.fromkeys(resolved))

    def all_historical_symbols(self) -> List[str]:
        """Return every symbol that was EVER a constituent."""
        resolved = [self.resolve_symbol(s) for s in self.table["symbol"].unique()]
        return list(dict.fromkeys(resolved))

    def get_sector(self, symbol: str) -> str:
        s = self.resolve_symbol(symbol)
        return self._sector_map.get(s, "Unmapped")


__all__ = ["PointInTimeUniverse", "KNOWN_RENAMES", "DEFAULT_PIT_CONSTITUENTS"]
