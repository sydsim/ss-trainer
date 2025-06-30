import numpy as np


class BasicState:
    def __init__(self, initial_balance, fee=0.0005, leverage=1, trade_lifecycle=20, verbose=True):
        self.prob_cur = 0
        self.position_price = 0
        self.position_side = 0
        self.position_life = 0
        self.position_created = 0
        self.position_amount = 0

        self.order_price_threshold = None
        self.order_side = 0
        self.order_life = 0
        self.order_created = 0

        self.current_bid_price = None
        self.current_bid_amount = None
        self.current_ask_price = None
        self.current_ask_amount = None

        self.balance = initial_balance
        self.fee = fee
        self.leverage = leverage
        self.trade_lifecycle = trade_lifecycle
        self.verbose = verbose

        self.trade_history = []
        self.order_history = []

    def close_position(self, timestamp, force_all=False):
        if self.position_side == 0:
            return

        if force_all:
            a_total = 0
            p_total = 0
            if self.position_side == 1:
                for p, a in zip(self.current_bid_price, self.current_bid_amount):
                    p_total += p * a
                    a_total += a
            elif self.position_side == 2:
                for p, a in zip(self.current_ask_price, self.current_ask_amount):
                    p_total += p * a
                    a_total += a
            p_total = p_total / a_total
            self.balance += self.position_amount * p_total
            self.position_amount = 0

        else:
            a_total = 0
            p_total = 0
            if self.position_side == 1:
                for p, a in zip(self.current_bid_price, self.current_bid_amount):
                    a_open = min(self.position_amount, a)
                    v_open = a_open * p
                    p_total += p * a_open
                    a_total += a_open
                    self.position_amount -= a_open
                    self.balance += v_open * (1 - self.fee)
            elif self.position_side == 2:
                for p, a in zip(self.current_ask_price, self.current_ask_amount):
                    a_open = min(self.position_amount, a)
                    v_open = a_open * p
                    p_total += p * a_open
                    a_total += a_open
                    self.position_amount -= a_open
                    self.balance += v_open * (1 - self.fee)

        self.trade_history.append((self.position_side, self.position_created, timestamp, self.position_amount, self.balance))
        if self.verbose:
            print(self.position_side, self.position_created, timestamp, self.position_amount, self.balance)

        if self.position_amount == 0:
            self.position_life = 0
            self.position_side = 0

    def update_current_price(self, timestamp, bid_price, bid_amount, ask_price, ask_amount):
        self.current_bid_price = bid_price
        self.current_bid_amount = bid_amount
        self.current_ask_price = ask_price
        self.current_ask_amount = ask_amount

        if self.order_life > 0:
            a_total = 0
            p_total = 0
            if self.order_side == 1:
                for p, a in zip(ask_price, ask_amount):
                    if p < self.order_price_threshold:
                        v_open = min(self.balance, p * a)
                        a_open = v_open / (1 + self.fee) / p
                        p_total += p * a_open
                        a_total += a_open
                        self.balance -= v_open
            elif self.order_side == 2:
                for p, a in zip(bid_price, bid_amount):
                    if p > self.order_price_threshold:
                        v_open = min(self.balance, p * a)
                        a_open = v_open / (1 + self.fee) / p
                        p_total += p * a_open
                        a_total += a_open
                        self.balance -= v_open

            if a_total > 0:
                p_total = p_total / a_total
                current_amount = self.position_amount + a_total
                current_volume = self.position_amount * self.position_price + p_total * a_total
                self.position_price = current_volume / current_amount
                self.position_created = timestamp
                self.position_life = self.trade_lifecycle
                self.position_side = self.order_side
                self.position_amount = current_amount

                if self.balance == 0:
                    self.order_life = 0

    def open_order(
        self, timestamp, base_price, prob_long, prob_short,
        order_threshold, threshold_long, threshold_short, signal_threshold
    ):
        if prob_long > order_threshold:
            pred_side = 1
            expected_profit = threshold_long
        elif prob_short > order_threshold:
            pred_side = 2
            expected_profit = threshold_short

        if self.position_side == 0:
            if expected_profit > signal_threshold:
                self._create_order(timestamp, base_price, pred_side, signal_threshold)
        else:
            if self.position_side != pred_side and expected_profit > signal_threshold:
                self.close_position(timestamp)
                self._create_order(timestamp, base_price, pred_side, signal_threshold)
            elif self.position_side == pred_side:
                self.position_life = self.trade_lifecycle

    def _create_order(self, timestamp, base_price, side, signal_threshold):
        if side == 1:
            self.order_price_threshold = base_price * (np.exp(signal_threshold) - self.fee)
        elif side == 2:
            self.order_price_threshold = base_price * (np.exp(-signal_threshold) + self.fee)
        self.order_side = side
        self.order_life = self.trade_lifecycle
        self.order_created = timestamp

    def close_order(self, timestamp):
        self.order_price_threshold = None
        self.order_side = 0
        self.order_life = 0
        self.order_created = 0

    def step(self, timestamp):
        if self.position_life > 0:
            self.position_life -= 1
        if self.position_life == 0 and self.position_amount > 0:
            self.close_position(timestamp)
        if self.order_life > 0:
            self.order_life -= 1
            if self.order_life == 0:
                self.close_order(timestamp)
