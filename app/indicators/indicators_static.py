import logging
from app.indicators.indicator_base import IndicatorStatic
import pandas as pd
import numpy as np
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator, WilliamsRIndicator, KAMAIndicator, StochRSIIndicator, UltimateOscillator, AwesomeOscillatorIndicator, PercentagePriceOscillator, PercentageVolumeOscillator, TSIIndicator
from ta.volatility import UlcerIndex, bollinger_pband, keltner_channel_pband, donchian_channel_pband
from ta.trend import ADXIndicator, TRIXIndicator, CCIIndicator, DPOIndicator, VortexIndicator, KSTIndicator, stc, MassIndex
from ta.volume import NegativeVolumeIndexIndicator, VolumePriceTrendIndicator, OnBalanceVolumeIndicator, AccDistIndexIndicator, MFIIndicator, ForceIndexIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, VolumeWeightedAveragePrice

import logging
logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

def ema1(adj_close, span):
    a= 2/(span+1)
    return pd.Series(adj_close).ewm(alpha=a).mean()

class MACDS(IndicatorStatic):
    NAME = 'MACDS'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        span1=12
        span2=26
        span3=9

        exp1 = ema1(ticker_df['adj_close'], span1)
        exp2 = ema1(ticker_df['adj_close'], span2)
        macd = 100 * (exp1 - exp2) / exp2
        signal = ema1(macd, span3)
        tiny_df[self.NAME] = signal.astype('float32')
        self.build_extra_features(tiny_df)

        return tiny_df

class MACD(IndicatorStatic):
    NAME = 'MACD'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        span1=12
        span2=26

        exp1 = ema1(ticker_df['adj_close'], span1)
        exp2 = ema1(ticker_df['adj_close'], span2)
        macd = 100 * (exp1 - exp2) / exp2
        tiny_df[self.NAME] = macd.astype('float32')
        self.build_extra_features(tiny_df)

        return tiny_df

class CPC(IndicatorStatic): #close percentage change
    NAME = 'CPC'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        tiny_df[self.NAME] = ticker_df['adj_close']
        self.build_extra_pct(tiny_df)
        tiny_df.drop([self.NAME], axis=1, inplace=True)
        return tiny_df

class ADI(IndicatorStatic): 
    NAME = 'ADI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        adi = AccDistIndexIndicator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = adi.acc_dist_index().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class OBV(IndicatorStatic):
    NAME = 'OBV'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        obv = OnBalanceVolumeIndicator(close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = obv.on_balance_volume().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class VPT(IndicatorStatic):
    NAME = 'VPT'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        vpt = VolumePriceTrendIndicator(close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = vpt.volume_price_trend().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class NVI(IndicatorStatic):
    NAME = 'NVI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        nvi = NegativeVolumeIndexIndicator(close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = nvi.negative_volume_index().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class KCP(IndicatorStatic):
    NAME = 'KCP'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        window=20
        window_atr=10

        kcp = keltner_channel_pband(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'], 
            window=window, window_atr=int(window_atr), fillna=False, original_version=True)

        tiny_df[self.NAME] = kcp.astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class KST(IndicatorStatic):
    NAME = 'KST'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        window1 = 10
        window2 = 10
        window3 = 10
        window4 = 15

        kst = KSTIndicator(close=ticker_df['adj_close'],
            window1=window1,
            window2=window2,
            window3=window3,
            window4=window4)
        tiny_df[self.NAME] = kst.kst_diff().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class STC(IndicatorStatic):
    NAME = 'STC'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        window_slow = 50
        window_fast = 23
        cycle = 10

        stc_ = stc(close=ticker_df['adj_close'], 
            window_slow=window_slow, 
            window_fast=window_fast, 
        cycle=cycle, smooth1=3, smooth2=3, fillna=False)

        tiny_df[self.NAME] = stc_.astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class MI(IndicatorStatic):
    NAME = 'MI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):

        window_fast = 9
        window_slow = 25

        mi = MassIndex(high=ticker_df['high'], low=ticker_df['low'], 
            window_fast=window_fast, 
            window_slow=window_slow)
        tiny_df[self.NAME] = mi.mass_index().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class UO(IndicatorStatic):
    NAME = 'UO'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        window1=7
        window2=14
        window3=28

        uo = UltimateOscillator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'],
            window1 = window1,
            window2 = window2,
            window3 = window3)
        tiny_df[self.NAME] = uo.ultimate_oscillator().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class AO(IndicatorStatic):
    NAME = 'AO'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        window1 = 5
        window2 = 34
        ao = AwesomeOscillatorIndicator(high=ticker_df['high'], low=ticker_df['low'],
            window1 = window1,
            window2 = window2)
        tiny_df[self.NAME] = ao.awesome_oscillator().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class PPO(IndicatorStatic):
    NAME = 'PPO'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        window_slow = 26
        window_fast = 12
        window_sign = 9

        ppo = PercentagePriceOscillator(close=ticker_df['adj_close'], 
            window_fast=window_fast, 
            window_slow=window_slow, 
            window_sign=window_sign)
        tiny_df[self.NAME] = ppo.ppo_hist().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class PVO(IndicatorStatic):
    NAME = 'PVO'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):

        window_slow = 26
        window_fast = 12
        window_sign = 9

        pvo = PercentageVolumeOscillator(volume=ticker_df['volume'], 
            window_fast=window_fast, 
            window_slow=window_slow, 
            window_sign=window_sign)
        tiny_df[self.NAME] = pvo.pvo_hist().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class TSI(IndicatorStatic):
    NAME = 'TSI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        window_slow = 25
        window_fast = 13

        tsi = TSIIndicator(close=ticker_df['adj_close'], 
            window_fast=window_fast, 
            window_slow=window_slow)
        tiny_df[self.NAME] = tsi.tsi().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class VI(IndicatorStatic):
    NAME = 'VI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        vi = VortexIndicator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'])
        tiny_df[self.NAME] = vi.vortex_indicator_diff().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class DCP(IndicatorStatic):
    NAME = 'DCP'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        dcp = donchian_channel_pband(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'], 
        offset=0, fillna=False)

        tiny_df[self.NAME] = dcp.astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df


class BBP(IndicatorStatic):
    NAME = 'BBP'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        bbp = bollinger_pband(close=ticker_df['adj_close'], fillna=False)
        tiny_df[self.NAME] = bbp.astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class ROC(IndicatorStatic):
    NAME = 'ROC'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        roc = ROCIndicator(close=ticker_df['adj_close'])
        tiny_df[self.NAME] = roc.roc().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class UI(IndicatorStatic):
    NAME = 'UI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        ui = UlcerIndex(close=ticker_df['adj_close'])
        tiny_df[self.NAME] = ui.ulcer_index().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df
    
class ADX(IndicatorStatic):
    NAME = 'ADX'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        try:
            adx = ADXIndicator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'])
            tiny_df[self.NAME] = adx.adx_pos().astype('float32')
            self.build_extra_features(tiny_df)
        except:
            logger.info(f'**** WARNING, could not create {self.NAME}. Its been excluded!')
        return tiny_df

def _get_min_max(series1, series2, function: str = "min"):
    """Find min or max value between two lists for each index"""
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)

class Impl_ADX(IndicatorStatic):
    def __init__(self):
        return
    
    def impl(self, ticker_df, interval):
        np.seterr(divide='ignore', invalid='ignore')
        fillna = False
        closes = ticker_df['adj_close']
        high = ticker_df['high']
        low = ticker_df['low']
        

        close_shift = closes.shift(1)
        pdm = _get_min_max(high, close_shift, "max")
        pdn = _get_min_max(low, close_shift, "min")
        diff_directional_movement = pdm - pdn

        trs_initial = np.zeros(interval - 1)
        if len(closes) <= (interval * 2): #by marc
            val_return = np.zeros(len(closes))
            val_return[:] = np.NaN
            return val_return
        tmp_val = len(closes) - (interval - 1)
        if tmp_val <= 0:
            return
        trs = np.zeros(len(closes) - (interval - 1))
        trs[0] = diff_directional_movement.dropna()[0 : interval].sum()
        diff_directional_movement = diff_directional_movement.reset_index(drop=True)

        for i in range(1, len(trs) - 1):
            trs[i] = (
                trs[i - 1]
                - (trs[i - 1] / float(interval))
                + diff_directional_movement[interval + i]
            )

        diff_up = high - high.shift(1)
        diff_down = low.shift(1) - low
        pos = abs(((diff_up > diff_down) & (diff_up > 0)) * diff_up)
        neg = abs(((diff_down > diff_up) & (diff_down > 0)) * diff_down)

        dip = np.zeros(len(closes) - (interval - 1))
        dip[0] = pos.dropna()[0 : interval].sum()

        pos = pos.reset_index(drop=True)

        for i in range(1, len(dip) - 1):
            dip[i] = (
                dip[i - 1]
                - (dip[i - 1] / float(interval))
                + pos[interval + i]
            )

        din = np.zeros(len(closes) - (interval - 1))
        din[0] = neg.dropna()[0 : interval].sum()

        neg = neg.reset_index(drop=True)

        for i in range(1, len(din) - 1):
            din[i] = (
                din[i - 1]
                - (din[i - 1] / float(interval))
                + neg[interval + i]
            )        

        dip2 = np.zeros(len(trs))
        for i in range(len(trs)):
            dip2[i] = 100 * (dip[i] / trs[i])

        din2 = np.zeros(len(trs))
        for i in range(len(trs)):
            din2[i] = 100 * (din[i] / trs[i])

        directional_index = 100 * np.abs((dip2 - din2) / (dip2 + din2))

        adx_series = np.zeros(len(trs))
        adx_series[interval] = directional_index[0 : self.interval].mean()

        for i in range(interval + 1, len(adx_series)):
            adx_series[i] = (
                (adx_series[i - 1] * (interval - 1)) + directional_index[i - 1]
            ) / float(interval)

        adx_series = np.concatenate((trs_initial, adx_series), axis=0)
            
        return adx_series

class ADX(IndicatorStatic):
    NAME = 'ADX'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        adx = Impl_ADX()
        tiny_df[self.NAME] = adx.impl(ticker_df, 14)
        self.build_extra_features(tiny_df)
        return tiny_df

class Impl_ATR():
    def __init__(self):
        return
    
    def _true_range(self, high, low, prev_close):
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range
    
    def impl(self, ticker_df, interval):
        closes = ticker_df['adj_close']
        highs = ticker_df['high']
        lows = ticker_df['low']

        length = len(closes.values)
        if length < (interval):
            atr = np.zeros(length)
            atr.fill(np.nan)
            return atr

        closes_shift = closes.shift(1)
        true_range = self._true_range(highs, lows, closes_shift)
        atr = np.zeros(len(closes))
        atr[:] = np.NaN #hack
        atr[interval - 1] = true_range[0 : interval].mean()
        for i in range(interval, len(atr)):
            atr[i] = (atr[i - 1] * (interval - 1) + true_range.iloc[i]) / float(interval)

        return atr

class ATR(IndicatorStatic): #normalized
    NAME = 'ATR'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        atr = Impl_ATR()
        tiny_df[self.NAME] = atr.impl(ticker_df, 14) / ticker_df['adj_close']
        self.build_extra_features(tiny_df)
        return tiny_df

class RSI(IndicatorStatic):
    NAME = 'RSI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        rsi = RSIIndicator(close=ticker_df['adj_close'], window=14)
        tiny_df[self.NAME] = rsi.rsi().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class SRSI(IndicatorStatic):
    NAME = 'SRSI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        srsi = StochRSIIndicator(close=ticker_df['adj_close'], 
            window=14, 
            smooth1=3, smooth2=3)
        tiny_df[self.NAME] = srsi.stochrsi().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class MFI(IndicatorStatic):
    NAME = 'MFI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        mfi = MFIIndicator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = mfi.money_flow_index().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class FI(IndicatorStatic):
    NAME = 'FI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        fi = ForceIndexIndicator(close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = fi.force_index().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class CMF(IndicatorStatic):
    NAME = 'CMF'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        cmf = ChaikinMoneyFlowIndicator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = cmf.chaikin_money_flow().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class VWAP(IndicatorStatic):
    NAME = 'VWAP'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        vwap = VolumeWeightedAveragePrice(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = vwap.volume_weighted_average_price().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class TRIX(IndicatorStatic):
    NAME = 'TRIX'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        trix = TRIXIndicator(close=ticker_df['adj_close'])
        tiny_df[self.NAME] = trix.trix().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class CCI(IndicatorStatic):
    NAME = 'CCI'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        cci = CCIIndicator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'])
        tiny_df[self.NAME] = cci.cci().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class DPO(IndicatorStatic):
    NAME = 'DPO'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        dpo = DPOIndicator(close=ticker_df['adj_close'])
        tiny_df[self.NAME] = dpo.dpo().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class SR(IndicatorStatic):
    NAME = 'SR'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        sr = StochasticOscillator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'])
        tiny_df[self.NAME] = sr.stoch_signal().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class WR(IndicatorStatic):
    NAME = 'WR'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        wr = WilliamsRIndicator(high=ticker_df['high'], low=ticker_df['low'], close=ticker_df['adj_close'])
        tiny_df[self.NAME] = wr.williams_r().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class KAMA(IndicatorStatic): #normalized
    NAME = 'KAMA'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        kama = KAMAIndicator(close=ticker_df['adj_close'])
        tiny_df[self.NAME] = ticker_df['adj_close'] / kama.kama().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df

class EMV(IndicatorStatic):
    NAME = 'EMV'
    def __init__(self):
        return
        
    def get_name(self):
        return self.NAME
    
    def build(self, tiny_df, ticker_df):
        emv = EaseOfMovementIndicator(high=ticker_df['high'], low=ticker_df['low'], volume=ticker_df['volume'])
        tiny_df[self.NAME] = emv.sma_ease_of_movement().astype('float32')
        self.build_extra_features(tiny_df)
        return tiny_df