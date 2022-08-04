import pandas as pd
import logging
from app.utils.name_utils import TA_FEATURE_PREFIX

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

class IndicatorBase:
    NAME = ''
    interval = -1
    indicator_lags = None
    def __init__(self):
        return

    def get_name(self):
        raise NotImplementedError("Please Implement this method")

    def set_interval(self, interval):
        self.interval = interval
        
    def rewrite_name(self, interval):
        self.NAME = f"{TA_FEATURE_PREFIX}_"  + self.NAME + "_" + str(interval)
    
    def set_lags(self, lags):
        self.indicator_lags = lags
        
    def build(self, df):
        raise NotImplementedError("Please Implement this method")

    def build_extra_features(self, tiny_df):
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
