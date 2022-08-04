from dateutil.relativedelta import FR
from app.indicators.wrapped_indicator import WrappedIndicator
from app.indicators.indicators_dynamic import KMA, VOL, RET, SHARP, SMA, EMA, WMA
from app.indicators.indicators_static import MACD, MACDS, VI, DCP, KCP, BBP, SR, ROC, UI, ADX, ATR, RSI, MFI, FI, CMF, VWAP, TRIX, CCI, DPO, WR, KAMA, KST, STC, SRSI, MI, TSI, PPO, PVO, UO, AO, EMV
import logging

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

class SubmissionConfiguration:
    public_id = ""
    model_id = ""
    secret_key = ""
    skip_check_needs_submission = True #do not check if the model has already been submitted
    numerai_submit = True  #make sure it's True if we want to submit predictions

    def __init__(self, properties):
        self.public_id = properties["public_id"]
        self.model_id = properties["model_id"]
        self.secret_key = properties["secret_key"]

class IndicatorConfiguration:
    indicators_dynamic = [KMA, VOL, RET, SHARP, SMA, EMA, WMA]
    indicators_static = [MACD, MACDS, VI, DCP, KCP, BBP, SR, ROC, UI, ADX, ATR, RSI, MFI, FI, CMF, VWAP, TRIX, CCI, DPO, WR, KAMA, KST, STC, SRSI, MI, TSI, PPO, PVO, UO, AO, EMV]

    static_windows = [0] #won't be used as window is the "default" one
    static_lags = [5, 10, 20]
    
    dynamic_windows = [14, 20, 40, 60]
    dynamic_lags = [0.25, 0.5, 0.75, 1.0]

    def get_indicators_dynamic(self):
        wrapped_indicator_instances = []
        for c in self.indicators_dynamic:
            wi = WrappedIndicator(c, self.dynamic_windows, self.dynamic_lags)
            wrapped_indicator_instances.append(wi)
        return wrapped_indicator_instances

    def get_indicators_static(self):
        wrapped_indicator_instances = []
        for c in self.indicators_static:
            wi = WrappedIndicator(c, self.static_windows, self.static_lags)
            wrapped_indicator_instances.append(wi)
        return wrapped_indicator_instances

    #must contain all indicators I want to process
    def get_indicator_list(self):
        return (self.get_indicators_dynamic() + self.get_indicators_static())

class DataConfiguration:
    ROOT_PATH = 'data'
    AWS_BASE_URL='https://numerai-signals-public-data.s3-us-west-2.amazonaws.com'
    skip_download = False #Skip download the raw data. Useful as it takes a while
    static_data = False #In case we want to use a static train, validation and live datasets (if we have it placed in the right folder)
    raw_data = 'yahoo'
    target_name = "target_20d" #Numerai's target 
    week_day = 4
    delta_day = FR

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

