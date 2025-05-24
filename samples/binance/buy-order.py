import json
import os
import time
import logging
from datetime import datetime, timedelta, timezone

import backtrader as bt

from backtrader import Order

from ccxtbt import CCXTStore


# Set a general level (e.g., INFO) for other loggers
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Specifically set the CCXTFeed logger to DEBUG level
logging.getLogger("TestStrategy").setLevel(logging.DEBUG)


class TestStrategy(bt.Strategy):
    def __init__(self):
        self.bought = False
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def next(self):
        if self.live_data and not self.bought:
            # Buy
            # size x price should be >10 USDT at a minimum at Binance
            # make sure you use a price that is below the market price if you don't want to actually buy
            self.order = self.buy(size=2.0, exectype=Order.Limit, price=5.4326)
            # And immediately cancel the buy order
            self.cancel(self.order)
            self.bought = True

        for data in self.datas:
            self.logger.info(
                "{} - {} | O: {} H: {} L: {} C: {} V:{}".format(
                    data.datetime.datetime(),
                    data._name,
                    data.open[0],
                    data.high[0],
                    data.low[0],
                    data.close[0],
                    data.volume[0],
                )
            )

    def notify_data(self, data, status, *args, **kwargs):
        self.logger.info(
            f"{data._name} Data Status: {data._getstatusname(status)}, Order Status: {status}"
        )
        if data._getstatusname(status) == "LIVE":
            self.live_data = True
        else:
            self.live_data = False


# absolute dir the script is in
script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, "../params.json")
with open(abs_file_path, "r") as f:
    params = json.load(f)

cerebro = bt.Cerebro(quicknotify=True)

cerebro.broker.setcash(10.0)

# Add the strategy
cerebro.addstrategy(TestStrategy)

# Create our store
config = {
    "apiKey": params["binance"]["apikey"],
    "secret": params["binance"]["secret"],
    "enableRateLimit": True,
    "nonce": lambda: str(int(time.time() * 1000)),
}

# Set a general level (e.g., INFO) for other loggers
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Specifically set the CCXTFeed logger to DEBUG level
logging.getLogger("CCXTStore").setLevel(logging.DEBUG)

store = CCXTStore(exchange="binance", currency="BNB", config=config, retries=5)

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
        "canceled_order": {"key": "status", "value": "canceled"},
    },
}

broker = store.getbroker(broker_mapping=broker_mapping)
cerebro.setbroker(broker)

# Get our data
# Drop newest will prevent us from loading partial data from incomplete candles
hist_start_date = datetime.now(tz=timezone.utc) - timedelta(minutes=50)
data = store.getdata(
    dataname="BNB/USDT",
    name="BNBUSDT",
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
