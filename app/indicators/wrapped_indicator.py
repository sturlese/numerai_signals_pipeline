import logging

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

class WrappedIndicator:
    wrapped_class = None
    intervals = None
    lags = None
    def __init__(self, wrapped_class, intervals):
        self.wrapped_class = wrapped_class
        self.intervals = intervals

    def build_indicators(self, tiny_df, df):   
        for interval in self.intervals:
            obj = self.wrapped_class()
            obj.set_interval(interval)
            obj.rewrite_name(interval)
            obj.build(tiny_df, df)
        return tiny_df
