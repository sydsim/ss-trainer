
class DataPoint:
    def __init__(self, timestamp, base_price, threshold_long, threshold_short, index):
        self.timestamp = timestamp
        self.base_price = base_price
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.index = index
