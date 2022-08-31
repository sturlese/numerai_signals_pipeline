import logging
from app.utils.name_utils import TA_FEATURE_PREFIX
import sys

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)

class IndicatorBase:
    NAME = ''
    interval = -1
    def __init__(self):
        return

    def get_name(self):
        return self.NAME

    def set_interval(self, interval):
        self.interval = interval
        
    def rewrite_name(self, interval):
        self.NAME = f"{TA_FEATURE_PREFIX}_"  + self.NAME + "_" + str(interval)
        
    def build(self, df):
        raise NotImplementedError("Please Implement this method")

class IndicatorDynamic(IndicatorBase):
    def build_extra_features(self, tiny_df):
        for i in self.indicator_lags:
            pos = int(self.interval * i)
            tiny_df[f'{self.NAME}_lag_{pos}'] = tiny_df[self.NAME].shift(pos).astype('float32')

class IndicatorStatic(IndicatorBase):
    def build_extra_features(self, tiny_df):
        for i in self.indicator_lags:
            pos = i
            tiny_df[f'{self.NAME}_lag_{pos}'] = tiny_df[self.NAME].shift(pos).astype('float32')

class IndicatorLaggerBase:
    NAME = ''
    interval = -1
    TYPE = ""
    lags = []
    def __init__(self, name, interval, type, lags):
        self.NAME = name
        self.interval = interval
        self.TYPE = type
        self.lags = lags

    def get_name(self):
        return self.NAME

    def get_interval(self):
        return self.interval

    def get_type(self):
        return self.TYPE

    def get_lags(self):
        return self.lags
        