from dateutil.relativedelta import FR
from app.indicators.indicator_base import IndicatorLaggerBase
from app.indicators.wrapped_indicator import WrappedIndicator
from app.indicators.indicators_dynamic import KMA, VOL, RET, SHARP, SMA, EMA, WMA
from app.indicators.indicators_static import MACD, MACDS, VI, DCP, KCP, BBP, SR, ROC, UI, ADX, ATR, RSI, MFI, FI, CMF, VWAP, TRIX, CCI, DPO, WR, KAMA, KST, STC, SRSI, MI, TSI, PPO, PVO, UO, AO, EMV
import logging
import sys
from app.utils.name_utils import INDICATOR_STATIC, INDICATOR_DYAMIC, TRANSFORMATION_BIN, TRANSFORMATION_ZSCORE, TRANSFORMATION_RANK

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)

class SubmissionConfiguration:
    public_id = ""
    model_id = ""
    secret_key = ""
    skip_check_needs_submission = False #do not check if the model has already been submitted
    numerai_submit = True  #make sure it's True if we want to submit predictions

    def __init__(self, properties):
        self.public_id = properties["public_id"]
        self.model_id = properties["model_id"]
        self.secret_key = properties["secret_key"]

class IndicatorConfiguration:
    indicators_dynamic = [KMA, VOL, RET, SHARP, SMA, EMA, WMA]
    indicators_static = [MACD, MACDS, VI, KCP, BBP, SR, ROC, UI, ADX, ATR, RSI, MFI, VWAP, TRIX, CCI, DPO, KST, MI, TSI, PPO, PVO, UO, AO, EMV] 

    static_lags = [5, 10, 20]
    dynamic_lags = [0.25, 0.5, 0.75, 1.0]
    dynamic_windows = [14, 20, 40, 60]
    
    def get_indicators_lag(self, windows, indicators, indicator_type, lags):
        lagged_indicators  = []
        for interval in windows:
            for indicator_class in indicators:
                obj = indicator_class()
                obj.set_interval(interval)
                obj.rewrite_name(interval)
                i_l_b = IndicatorLaggerBase(obj.get_name(), interval, indicator_type, lags)
                lagged_indicators.append(i_l_b)
        return lagged_indicators

    def get_indicators_lag_dynamics(self):
        return self.get_indicators_lag(self.dynamic_windows, self.indicators_dynamic, INDICATOR_DYAMIC, self.dynamic_lags)

    def get_indicators_lag_static(self):
        return self.get_indicators_lag([0], self.indicators_static, INDICATOR_STATIC, self.static_lags)

    def get_indicators(self, indicators, windows):
        wrapped_indicator_instances = []
        for c in indicators:
            w_i = WrappedIndicator(c, windows)
            wrapped_indicator_instances.append(w_i)
        return wrapped_indicator_instances
    
    def get_indicators_dynamic(self):
        return self.get_indicators(self.indicators_dynamic, self.dynamic_windows)

    def get_indicators_static(self):
        return self.get_indicators(self.indicators_static, [0])

    #contains all indicators needed for the indicator_builder phase
    def get_indicator_list(self):
        return (self.get_indicators_dynamic() + self.get_indicators_static())

    #contains all indicators needed for the indicator_lagger phase
    def get_lagged_indicator_list(self):
        return (self.get_indicators_lag_dynamics() + self.get_indicators_lag_static())

class DataConfiguration:
    ROOT_PATH = 'data'
    AWS_BASE_URL='https://numerai-signals-public-data.s3-us-west-2.amazonaws.com'
    skip_download = False #Skip download the raw data. Useful as it takes a while
    static_data = False #In case we want to use a static train, validation and live datasets (if we have it placed in the right folder)
    raw_data = 'yahoo'
    target_name = "target_20d" #Numerai's target 
    week_day = 4
    delta_day = FR
    TRANSFORMATION_TYPE = TRANSFORMATION_BIN

    def __init__(self, static_data):
        self.static_data = static_data
        if self.static_data:
            self.skip_download = True

def get_indicator_config():
    config = IndicatorConfiguration()
    return config

def get_data_config(static_data):
    config = DataConfiguration(static_data)
    return config

def get_submission_config(properties):
    config = SubmissionConfiguration(properties)
    return config
