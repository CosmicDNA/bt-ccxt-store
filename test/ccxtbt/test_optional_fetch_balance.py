import time
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
from backtrader import Cerebro, Strategy, TimeFrame
from ccxt.base.errors import ExchangeError, NetworkError, NotSupported

from ccxtbt import CCXTFeed, CCXTStore


class TestFeedInitialFetchBalance(unittest.TestCase):
    """
    At least at Binance and probably on other exchanges too fetching ohlcv data doesn't need authentication
    while obviously fetching the balance of ones account does need authentication.
    Usually the CCXTStore fetches the balance when it is initialized which is not a problem during live trading
    operation.
    But the store is also initialized when the CCXTFeed is created and used during unit testing and backtesting.
    For this case it is beneficial to turn off the initial fetching of the balance as it is not really needed and
    it avoids needing to have api keys.
    This makes it possible for users that don't have a Binance api key to run backtesting and unit tests with real
    ohlcv data to try out this lib.
    """

    def setUp(self):
        """
        The initial balance is fetched in the context of the initialization of the CCXTStore.
        But as the CCXTStore is a singleton it's normally initialized only once and the instance is reused
        causing side effects.
        If the  first test run initializes the store without fetching the balance a subsequent test run
        would not try to fetch the balance again as the initialization won't happen again.
        Resetting the singleton to None here causes the initialization of the store to happen in every test method.
        """
        CCXTStore._singleton = None

    @patch("ccxt.binance.binance.fetch_balance")
    def test_fetch_balance_throws_error(self, fetch_balance_mock):
        """
        If API keys are provided the store is expected to fetch the balance.
        """

        config = {
            "apikey": "an-api-key",
            "secret": "an-api-secret",
            "enableRateLimit": True,
            "nonce": lambda: str(int(time.time() * 1000)),
        }
        backtesting(config)

        fetch_balance_mock.assert_called_once()

    def test_default_fetch_balance_param(self):
        """
        If API keys are provided the store is expected to
        not fetch the balance and load the ohlcv data without them.
        """
        config = {
            "enableRateLimit": True,
            "nonce": lambda: str(int(time.time() * 1000)),
        }
        finished_strategies = backtesting(config)
        self.assertEqual(finished_strategies[0].next_runs, 3)


class TestGetHistoricalDataFrame(unittest.TestCase):
    def setUp(self):
        CCXTStore._singleton = None
        self.config = {}  # Minimal config, no API keys to avoid fetch_balance call
        self.currency = "BTC"
        self.exchange_name = "binance"  # Example exchange
        self.symbol = "BTC/USDT"
        self.interval_str = "1h"
        self.df_columns = ["open", "high", "low", "close", "volume"]
        self.start_ts_ms = int(datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        self.interval_ms = 60 * 60 * 1000  # 1 hour

    def _generate_ohlcv_data(self, num_candles, start_time_ms=None, interval_ms=None, with_none_row_at_idx=None):
        start_time_ms = start_time_ms if start_time_ms is not None else self.start_ts_ms
        interval_ms = interval_ms if interval_ms is not None else self.interval_ms
        data = []
        for i in range(num_candles):
            ts = start_time_ms + i * interval_ms
            if with_none_row_at_idx == i:
                # o, h, l, c, v - with a None in 'close'
                data.append([ts, 100.0 + i, 110.0 + i, 90.0 + i, None, 1000.0 + i * 10])
            else:
                data.append([ts, 100.0 + i, 110.0 + i, 90.0 + i, 105.0 + i, 1000.0 + i * 10])
        return data

    def _assert_dataframe_structure(self, df, expected_rows):
        self.assertIsInstance(df, pd.DataFrame)
        self.assertListEqual(list(df.columns), self.df_columns)
        self.assertEqual(len(df), expected_rows)
        if expected_rows > 0:
            self.assertIsInstance(df.index, pd.DatetimeIndex)
            self.assertEqual(df.index.name, "timestamp")
            self.assertEqual(df.index.tz, timezone.utc)
            for col in self.df_columns:
                self.assertTrue(pd.api.types.is_float_dtype(df[col]))
        else:  # Empty DataFrame
            self.assertTrue(df.empty)
            self.assertIsInstance(df.index, pd.DatetimeIndex)  # Should still have index structure
            self.assertEqual(df.index.tz, timezone.utc)
            for col in self.df_columns:
                self.assertTrue(pd.api.types.is_float_dtype(df[col]))

    @patch("ccxt.binance")
    def test_successful_fetch_drop_incomplete_true(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name

        num_candles_to_request = 5
        # fetch_ohlcv will be called with limit = num_candles_to_request + 1 = 6
        fetched_data = self._generate_ohlcv_data(num_candles_to_request + 1)
        mock_exchange.fetch_ohlcv.return_value = fetched_data

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        df = store.get_historical_dataframe(
            self.symbol, self.interval_str, num_candles_to_request, drop_incomplete_candle=True
        )

        self._assert_dataframe_structure(df, num_candles_to_request)
        mock_exchange.fetch_ohlcv.assert_called_once_with(
            self.symbol, timeframe=self.interval_str, since=None, limit=num_candles_to_request + 1, params={}
        )
        # Check if the last element of fetched_data was dropped
        self.assertEqual(
            df.iloc[-1]["close"], fetched_data[num_candles_to_request - 1][4]
        )  # Compare with the correct pre-drop element

    @patch("ccxt.binance")
    def test_successful_fetch_drop_incomplete_false(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name

        num_candles_to_request = 5
        fetched_data = self._generate_ohlcv_data(num_candles_to_request)
        mock_exchange.fetch_ohlcv.return_value = fetched_data

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        df = store.get_historical_dataframe(
            self.symbol, self.interval_str, num_candles_to_request, drop_incomplete_candle=False
        )

        self._assert_dataframe_structure(df, num_candles_to_request)
        mock_exchange.fetch_ohlcv.assert_called_once_with(
            self.symbol, timeframe=self.interval_str, since=None, limit=num_candles_to_request, params={}
        )
        if not df.empty:  # Ensure df is not empty before iloc
            self.assertEqual(df.iloc[-1]["close"], fetched_data[-1][4])

    @patch("ccxt.binance")
    def test_fetch_fewer_candles_than_requested(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name

        num_candles_to_request = 10
        candles_returned_by_exchange = 3
        fetched_data = self._generate_ohlcv_data(candles_returned_by_exchange)
        mock_exchange.fetch_ohlcv.return_value = fetched_data

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        df = store.get_historical_dataframe(
            self.symbol, self.interval_str, num_candles_to_request, drop_incomplete_candle=False
        )

        self._assert_dataframe_structure(df, candles_returned_by_exchange)

    @patch("ccxt.binance")
    def test_fetch_fewer_candles_with_drop_incomplete(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name

        num_candles_to_request = 10
        candles_returned_by_exchange = 3  # limit will be 11, exchange returns 3
        fetched_data = self._generate_ohlcv_data(candles_returned_by_exchange)
        mock_exchange.fetch_ohlcv.return_value = fetched_data

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        df = store.get_historical_dataframe(
            self.symbol, self.interval_str, num_candles_to_request, drop_incomplete_candle=True
        )

        # 3 returned, 1 dropped -> 2 expected
        self._assert_dataframe_structure(df, candles_returned_by_exchange - 1)

    @patch("ccxt.binance")
    def test_num_candles_zero(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)

        # Test with drop_incomplete_candle=False
        df_false = store.get_historical_dataframe(self.symbol, self.interval_str, 0, drop_incomplete_candle=False)
        self._assert_dataframe_structure(df_false, 0)
        mock_exchange.fetch_ohlcv.assert_not_called()  # Should return early

        # Test with drop_incomplete_candle=True
        mock_exchange.fetch_ohlcv.reset_mock()
        fetched_data = self._generate_ohlcv_data(1)  # limit will be 1
        mock_exchange.fetch_ohlcv.return_value = fetched_data
        df_true = store.get_historical_dataframe(self.symbol, self.interval_str, 0, drop_incomplete_candle=True)
        self._assert_dataframe_structure(df_true, 0)
        mock_exchange.fetch_ohlcv.assert_called_once_with(
            self.symbol, timeframe=self.interval_str, since=None, limit=1, params={}
        )

    @patch("ccxt.binance")
    def test_exchange_returns_no_data(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name
        mock_exchange.fetch_ohlcv.return_value = []  # No data

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        df = store.get_historical_dataframe(self.symbol, self.interval_str, 5)
        self._assert_dataframe_structure(df, 0)

    @patch("ccxt.binance")
    def test_fetchohlcv_not_supported(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": False}  # Not supported
        mock_exchange.id = self.exchange_name

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        with self.assertRaises(NotSupported):
            store.get_historical_dataframe(self.symbol, self.interval_str, 5)

    @patch("ccxt.binance")
    def test_timeframe_not_supported(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {"5m": "5m"}  # '1h' is not supported
        mock_exchange.id = self.exchange_name

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        with self.assertRaises(ValueError):
            store.get_historical_dataframe(self.symbol, self.interval_str, 5)

    @patch("ccxt.binance")
    @patch("logging.getLogger")  # Patch the getLogger call
    def test_network_error_returns_empty_df(self, mock_getLogger, MockExchangeCls):
        # Setup the mock logger instance that getLogger will return
        mock_logger_instance = MagicMock()
        mock_getLogger.return_value = mock_logger_instance

        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name
        mock_exchange.fetch_ohlcv.side_effect = NetworkError("Simulated network error")

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        df = store.get_historical_dataframe(self.symbol, self.interval_str, 5)

        self._assert_dataframe_structure(df, 0)
        self.assertTrue(mock_logger_instance.error.called)

    @patch("ccxt.binance")
    def test_data_with_none_values_is_dropped(self, MockExchangeCls):
        mock_exchange = MockExchangeCls.return_value
        mock_exchange.has = {"fetchOHLCV": True}
        mock_exchange.timeframes = {self.interval_str: self.interval_str}
        mock_exchange.id = self.exchange_name

        num_candles_to_request = 3
        # Generate 3 candles, the middle one (idx=1) will have a None in 'close'
        fetched_data = self._generate_ohlcv_data(num_candles_to_request, with_none_row_at_idx=1)
        mock_exchange.fetch_ohlcv.return_value = fetched_data

        store = CCXTStore(exchange=self.exchange_name, currency=self.currency, config=self.config, retries=1)
        df = store.get_historical_dataframe(
            self.symbol, self.interval_str, num_candles_to_request, drop_incomplete_candle=False
        )

        self._assert_dataframe_structure(df, num_candles_to_request - 1)  # One row should be dropped
        # Ensure the row with None was indeed dropped, and others remain
        self.assertNotIn(None, df["close"].values)  # Check no None in close after dropna
        if len(df) == 2:  # Check only if expected number of rows are present
            self.assertAlmostEqual(df.iloc[0]["close"], fetched_data[0][4])
            self.assertAlmostEqual(df.iloc[1]["close"], fetched_data[2][4])


class _TestStrategy(Strategy):
    def __init__(self):
        self.next_runs = 0

    def next(self, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print("%s closing price: %s" % (dt.isoformat(), self.datas[0].close[0]))
        self.next_runs += 1


def backtesting(config):
    cerebro = Cerebro()

    cerebro.addstrategy(_TestStrategy)

    cerebro.adddata(
        CCXTFeed(
            exchange="binance",
            dataname="BNB/USDT",
            timeframe=TimeFrame.Minutes,
            fromdate=datetime(2019, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            todate=datetime(2019, 1, 1, 0, 2, tzinfo=timezone.utc),
            compression=1,
            ohlcv_limit=2,
            currency="BNB",
            config=config,
            retries=5,
        )
    )

    finished_strategies = cerebro.run()
    return finished_strategies


if __name__ == "__main__":
    unittest.main()
