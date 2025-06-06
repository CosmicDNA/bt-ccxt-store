import logging
from datetime import datetime, timedelta, timezone
from os import environ

import backtrader as bt

from ccxtbt import CCXTStore

# Set a general level (e.g., INFO) for other loggers
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# Specifically set the CCXTFeed logger to DEBUG level
logging.getLogger("TestStrategy").setLevel(logging.DEBUG)


class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data, period=21)
        self.logger = logging.getLogger(self.__class__.__name__)

    def next(self):
        # Get cash and balance
        # New broker method that will let you get the cash and balance for
        # any wallet. It also means we can disable the getcash() and getvalue()
        # rest calls before and after next which slows things down.

        # NOTE: If you try to get the wallet balance from a wallet you have
        # never funded, a KeyError will be raised! Change LTC below as approriate
        if self.live_data:
            cash, value = self.broker.get_wallet_balance("LTC")
        else:
            # Avoid checking the balance during a backfill. Otherwise, it will
            # Slow things down.
            cash = "NA"

        for data in self.datas:
            self.logger.info(
                "{} - {} | Cash {} | O: {} H: {} L: {} C: {} V:{} SMA:{}".format(
                    data.datetime.datetime(),
                    data._name,
                    cash,
                    data.open[0],
                    data.high[0],
                    data.low[0],
                    data.close[0],
                    data.volume[0],
                    self.sma[0],
                )
            )

    def notify_data(self, data, status, *args, **kwargs):
        self.logger.info(f"{data._name} Data Status: {data._getstatusname(status)}")
        if data._getstatusname(status) == "LIVE":
            self.live_data = True
        else:
            self.live_data = False


# Load API key and secret from environment variables
apikey = environ.get("KRAKEN_API_KEY")
secret = environ.get("KRAKEN_API_SECRET")

cerebro = bt.Cerebro(quicknotify=True)


# Add the strategy
cerebro.addstrategy(TestStrategy)

# Create our store
config = {"apiKey": apikey, "secret": secret, "enableRateLimit": True}

# IMPORTANT NOTE - Kraken (and some other exchanges) will not return any values
# for get cash or value if You have never held any LTC coins in your account.
# So switch LTC to a coin you have funded previously if you get errors
store = CCXTStore(exchange="kraken", currency="LTC", config=config, retries=5)


# Get the broker and pass any kwargs if needed.
# ----------------------------------------------
# Broker mappings have been added since some exchanges expect different values
# to the defaults. Case in point, Kraken vs Bitmex. NOTE: Broker mappings are not
# required if the broker uses the same values as the defaults in CCXTBroker.
broker_mapping = {
    "order_types": {
        bt.Order.Market: "market",
        bt.Order.Limit: "limit",
        bt.Order.Stop: "stop-loss",  # stop-loss for kraken, stop for bitmex
        bt.Order.StopLimit: "stop limit",
    },
    "mappings": {
        "closed_order": {"key": "status", "value": "closed"},
        "canceled_order": {"key": "result", "value": 1},
    },
}

broker = store.getbroker(broker_mapping=broker_mapping)
cerebro.setbroker(broker)

# Get our data
# Drop newest will prevent us from loading partial data from incomplete candles
hist_start_date = datetime.now(tz=timezone.utc) - timedelta(minutes=50)
data = store.getdata(
    dataname="LTC/USD",
    name="LTCUSD",
    timeframe=bt.TimeFrame.Minutes,
    fromdate=hist_start_date,
    compression=1,
    ohlcv_limit=50,
    drop_newest=True,
)  # , historical=True)

# Add the feed
cerebro.adddata(data)

# Run the strategy
cerebro.run()
