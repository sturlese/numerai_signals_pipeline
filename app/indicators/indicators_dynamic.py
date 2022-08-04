import logging
import numpy as np
from app.indicators.indicator_base import IndicatorDynamic
from ta.trend import SMAIndicator, EMAIndicator, WMAIndicator

import logging
logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

class KMA(IndicatorDynamic):
    NAME = 'KMA'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        num = ticker_df['adj_close'].astype('float32')
        denom = ticker_df['adj_close'].rolling(self.interval).mean().astype('float32')
        tiny_df[self.NAME] = num/denom
        self.build_extra_features(tiny_df)
        return tiny_df

class VOL(IndicatorDynamic): #volatility
    NAME = 'VOL'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        tiny_df[self.NAME] = np.log1p(ticker_df['adj_close']).pct_change().rolling(self.interval).std().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class RET(IndicatorDynamic): #returns
    NAME = 'RET'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        tiny_df[self.NAME] = ticker_df['adj_close'].pct_change(self.interval).astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class SHARP(IndicatorDynamic): #sharp ratio
    NAME = 'SHP'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        num = np.log1p(ticker_df['adj_close']).pct_change().rolling(self.interval).mean().astype('float32')
        denom = np.log1p(ticker_df['adj_close']).pct_change().rolling(self.interval).std().astype('float32')
        tiny_df[self.NAME] = num/denom
        self.build_extra_features(tiny_df)
        return tiny_df

class SMA(IndicatorDynamic): #normalized
    NAME = 'SMA'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        sma = SMAIndicator(close=ticker_df['adj_close'], window=self.interval)
        tiny_df[self.NAME] = ticker_df['adj_close'] / sma.sma_indicator().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class EMA(IndicatorDynamic): #normalized
    NAME = 'EMA'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        ema = EMAIndicator(close=ticker_df['adj_close'], window=self.interval)
        tiny_df[self.NAME] = ticker_df['adj_close'] / ema.ema_indicator().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class WMA(IndicatorDynamic): #normalized
    NAME = 'WMA'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        wma = WMAIndicator(close=ticker_df['adj_close'], window=self.interval)
        tiny_df[self.NAME] = ticker_df['adj_close'] / wma.wma().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

