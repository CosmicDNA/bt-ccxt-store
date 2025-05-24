#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2017 Ed Bartosh <bartosh@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time
from datetime import datetime
from functools import wraps

import backtrader as bt
import ccxt
import pandas as pd
from backtrader.metabase import MetaParams
from backtrader.utils.py3 import with_metaclass
from ccxt.base.errors import ExchangeError, NetworkError, NotSupported


class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(cls, name, bases, dct):
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super(MetaSingleton, cls).__call__(*args, **kwargs)

        return cls._singleton


class CCXTStore(with_metaclass(MetaSingleton, object)):
    """API provider for CCXT feed and broker classes.

    Added a new get_wallet_balance method. This will allow manual checking of the balance.
        The method will allow setting parameters. Useful for getting margin balances

    Added new private_end_point method to allow using any private non-unified end point

    """

    # Supported granularities
    _GRANULARITIES = {
        (bt.TimeFrame.Minutes, 1): "1m",
        (bt.TimeFrame.Minutes, 3): "3m",
        (bt.TimeFrame.Minutes, 5): "5m",
        (bt.TimeFrame.Minutes, 15): "15m",
        (bt.TimeFrame.Minutes, 30): "30m",
        (bt.TimeFrame.Minutes, 60): "1h",
        (bt.TimeFrame.Minutes, 90): "90m",
        (bt.TimeFrame.Minutes, 120): "2h",
        (bt.TimeFrame.Minutes, 180): "3h",
        (bt.TimeFrame.Minutes, 240): "4h",
        (bt.TimeFrame.Minutes, 360): "6h",
        (bt.TimeFrame.Minutes, 480): "8h",
        (bt.TimeFrame.Minutes, 720): "12h",
        (bt.TimeFrame.Days, 1): "1d",
        (bt.TimeFrame.Days, 3): "3d",
        (bt.TimeFrame.Weeks, 1): "1w",
        (bt.TimeFrame.Weeks, 2): "2w",
        (bt.TimeFrame.Months, 1): "1M",
        (bt.TimeFrame.Months, 3): "3M",
        (bt.TimeFrame.Months, 6): "6M",
        (bt.TimeFrame.Years, 1): "1y",
    }

    BrokerCls = None  # broker class will auto register
    DataCls = None  # data class will auto register

    @classmethod
    def getdata(cls, *args, **kwargs):
        """Returns ``DataCls`` with args, kwargs"""
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        """Returns broker with *args, **kwargs from registered ``BrokerCls``"""
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self, exchange, currency, config, retries, sandbox=False):
        self.exchange = getattr(ccxt, exchange)(config)
        if sandbox:
            self.exchange.set_sandbox_mode(True)
        self.currency = currency
        self.retries = retries
        self.logger = logging.getLogger(self.__class__.__name__)
        balance = self.exchange.fetch_balance() if "secret" in config else 0
        try:
            if balance == 0 or not balance["free"][currency]:
                self._cash = 0
            else:
                self._cash = balance["free"][currency]
        except KeyError:  # never funded or eg. all USD exchanged
            self._cash = 0
        try:
            if balance == 0 or not balance["total"][currency]:
                self._value = 0
            else:
                self._value = balance["total"][currency]
        except KeyError:
            self._value = 0

    def get_granularity(self, timeframe, compression):
        if not self.exchange.has["fetchOHLCV"]:
            raise NotImplementedError("'%s' exchange doesn't support fetching OHLCV data" % self.exchange.name)

        granularity = self._GRANULARITIES.get((timeframe, compression))
        if granularity is None:
            raise ValueError(
                "backtrader CCXT module doesn't support fetching OHLCV "
                "data for time frame %s, comression %s" % (bt.TimeFrame.getname(timeframe), compression)
            )

        if self.exchange.timeframes and granularity not in self.exchange.timeframes:
            raise ValueError(
                "'%s' exchange doesn't support fetching OHLCV data for "
                "%s time frame" % (self.exchange.name, granularity)
            )

        return granularity

    def retry(method):
        @wraps(method)
        def retry_method(self, *args, **kwargs):
            for i in range(self.retries):
                self.logger.debug("%s - %s - Attempt %s", datetime.now(), method.__name__, i)
                time.sleep(self.exchange.rateLimit / 1000)
                try:
                    return method(self, *args, **kwargs)
                except (NetworkError, ExchangeError):
                    if i == self.retries - 1:
                        raise

        return retry_method

    @retry
    def get_wallet_balance(self, currency, params=None):
        balance = self.exchange.fetch_balance(params)
        return balance

    @retry
    def get_balance(self):
        balance = self.exchange.fetch_balance()

        cash = balance["free"][self.currency]
        value = balance["total"][self.currency]
        # Fix if None is returned
        self._cash = cash if cash else 0
        self._value = value if value else 0

    @retry
    def getposition(self):
        return self._value
        # return self.getvalue(currency)

    @retry
    def create_order(self, symbol, order_type, side, amount, price, params):
        # returns the order
        return self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount,
            price=price,
            params=params,
        )

    @retry
    def cancel_order(self, order_id, symbol):
        return self.exchange.cancel_order(order_id, symbol)

    @retry
    def fetch_trades(self, symbol):
        return self.exchange.fetch_trades(symbol)

    @retry
    def fetch_ohlcv(self, symbol, timeframe, since, limit, params=None):
        params = params if params is not None else {}
        self.logger.debug(
            "Fetching: %s, TF: %s, Since: %s, Limit: %s, Params: %s",
            symbol,
            timeframe,
            since,
            limit,
            params,
        )
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit, params=params)

    @retry
    def fetch_order(self, oid, symbol):
        return self.exchange.fetch_order(oid, symbol)

    @retry
    def fetch_open_orders(self, symbol=None):
        if symbol is None:
            return self.exchange.fetchOpenOrders()
        else:
            return self.exchange.fetchOpenOrders(symbol)

    @retry
    def private_end_point(self, type, endpoint, params):
        """
        Open method to allow calls to be made to any private end point.
        See here: https://github.com/ccxt/ccxt/wiki/Manual#implicit-api-methods

        - type: String, 'Get', 'Post','Put' or 'Delete'.
        - endpoint = String containing the endpoint address eg. 'order/{id}/cancel'
        - Params: Dict: An implicit method takes a dictionary of parameters, sends
          the request to the exchange and returns an exchange-specific JSON
          result from the API as is, unparsed.

        To get a list of all available methods with an exchange instance,
        including implicit methods and unified methods you can simply do the
        following:

        print(dir(ccxt.hitbtc()))
        """
        return getattr(self.exchange, endpoint)(params)

    def get_historical_dataframe(
        self, symbol: str, interval_str: str, num_candles: int, drop_incomplete_candle: bool = True
    ) -> pd.DataFrame:
        """
        Fetches a block of historical OHLCV data directly into a Pandas DataFrame.

        Args:
            symbol (str): The trading symbol (e.g., 'BTC/USDT').
            interval_str (str): The timeframe interval string (e.g., '1m', '5m', '1h', '1d').
                                This should be a timeframe string recognized by CCXT.
                                It will be validated against the exchange's capabilities.
            num_candles (int): The number of candles to fetch.
            drop_incomplete_candle (bool): If True (default), fetches num_candles + 1 and
                                           drops the newest one to ensure only completed
                                           candles are returned.

        Returns:
            pd.DataFrame: A Pandas DataFrame with a DatetimeIndex (UTC) named 'timestamp'
                          and columns ['open', 'high', 'low', 'close', 'volume'].
                          Returns an empty DataFrame with this structure if data cannot be fetched.
                          Raises NotSupported or ValueError for configuration issues.
        """
        if not self.exchange.has["fetchOHLCV"]:
            msg = f"{self.exchange.id} does not support fetchOHLCV."
            self.logger.error(msg)
            raise NotSupported(msg)

        if self.exchange.timeframes and interval_str not in self.exchange.timeframes:
            msg = (
                f"'{self.exchange.id}' exchange doesn't support fetching OHLCV data for "
                f"{interval_str} time frame. Supported: {list(self.exchange.timeframes.keys())}"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        df_columns = ["open", "high", "low", "close", "volume"]
        empty_df_dtypes = {col: "float" for col in df_columns}
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC", name="timestamp"), columns=df_columns).astype(
            empty_df_dtypes
        )

        limit_to_fetch = num_candles
        if drop_incomplete_candle:
            limit_to_fetch += 1

        if limit_to_fetch <= 0:
            self.logger.warning(
                f"Calculated limit_to_fetch is {limit_to_fetch} (num_candles: {num_candles}, "
                f"drop_incomplete_candle: {drop_incomplete_candle}). "
                f"Returning empty DataFrame for {symbol} {interval_str}."
            )
            return empty_df

        try:
            # Use the existing self.fetch_ohlcv which includes retry logic
            # CCXT's limit parameter means "max number of candles to return"
            # 'since=None' fetches the most recent candles
            ohlcv_data = self.fetch_ohlcv(
                symbol=symbol,
                timeframe=interval_str,
                since=None,
                limit=limit_to_fetch,
                params={},  # Add if any specific params are needed for this direct call
            )

            if not ohlcv_data:
                self.logger.warning(
                    f"No data returned by exchange for {symbol}, interval {interval_str}, limit {limit_to_fetch}."
                )
                return empty_df

            df = pd.DataFrame(ohlcv_data, columns=["timestamp"] + df_columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            for col in df_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=df_columns, inplace=True)

            if df.empty:
                self.logger.warning(f"DataFrame became empty after processing for {symbol}, interval {interval_str}.")
                return empty_df

            if drop_incomplete_candle and not df.empty:
                df = df.iloc[:-1]  # Drop the last (newest) candle

            df = df.tail(num_candles)  # Ensure we return at most num_candles

            return df.astype(empty_df_dtypes) if not df.empty else empty_df.copy()

        except (NetworkError, ExchangeError) as e:  # Should be caught by self.fetch_ohlcv retry, but as a safeguard
            self.logger.error(
                f"CCXT Network/Exchange Error in get_historical_dataframe for {symbol} ({interval_str}): {e}"
            )
            return empty_df
        except Exception as e:  # Catch any other unexpected errors
            self.logger.error(
                f"Unexpected error in get_historical_dataframe for {symbol} ({interval_str}): {e}", exc_info=True
            )
            return empty_df
