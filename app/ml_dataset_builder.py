from app.utils.utils import get_ticker_data
from app.utils.name_utils import ML_DATA_FILE
import pandas as pd
import logging
from datetime import datetime
import gc
from dateutil.relativedelta import relativedelta, FR
import logging

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

def get_data(db_dir, config):

    SIGNALS_UNIVERSE=f'{config.AWS_BASE_URL}/latest_universe.csv'
    SIGNALS_TARGETS=f'{config.AWS_BASE_URL}/signals_train_val_bbg.csv'
    STATIC_SIGNALS_UNIVERSE = f'{config.ROOT_PATH}/static_data/live_universe.csv'
    STATIC_SIGNALS_TARGETS = f'{config.ROOT_PATH}/static_data/historical_targets.csv'
    
    last_friday = datetime.today() - relativedelta(weekday=config.delta_day(-1))
    if config.static_data: 
        last_friday = datetime(2021, 6, 18, 0, 0)

    logger.info("Loadin data...")
    ticker_data = get_ticker_data(db_dir)
    logger.info(f'Shape ticker data before dropping NaN {ticker_data.shape}')
    feature_names = [f for f in ticker_data.columns if f.startswith("feature")]

    ticker_data = ticker_data.dropna(subset=feature_names)
    logger.info(f'Shape ticker data after dropping NaN {ticker_data.shape}')
    ticker_universe = pd.DataFrame()
    targets = pd.DataFrame()
    if config.static_data:
        path_ticker_universe = STATIC_SIGNALS_UNIVERSE
        path_targets = STATIC_SIGNALS_TARGETS
    else:
        path_ticker_universe = SIGNALS_UNIVERSE
        path_targets = SIGNALS_TARGETS

    logger.info (f'Path targets: {path_targets}')
    logger.info (f'Path tickers universe: {path_ticker_universe}')
    ticker_universe = pd.read_csv(path_ticker_universe)
    ticker_data = ticker_data[ticker_data.bloomberg_ticker.isin(ticker_universe['bloomberg_ticker'])]
    logger.info("Merging targets...")
    gc.collect()

    ml_data = pd.DataFrame()

    targets = pd.read_csv(path_targets) 
    targets['date'] = pd.to_datetime(targets['friday_date'],format='%Y%m%d')  
    ml_data = pd.merge(
        ticker_data, targets,
        on=['date', 'bloomberg_ticker'],
        how='left'
    )

    gc.collect()    
    logger.info(f'Shape ML dataset before cleaning {ml_data.shape}')
    logger.info(f'Found {ml_data[config.target_name].isna().sum()} rows without target in ML dataset')
    ml_data = ml_data.dropna(subset=feature_names) #already done in ticker_data
    ml_data = ml_data.set_index('date')
    ml_data = ml_data[ml_data.index.weekday == config.week_day]
    ml_data = ml_data[ml_data.index.value_counts() > 50]
    logger.info(f"Shape ML dataset after cleaning {ml_data.shape}")

    train_data = ml_data[ml_data['data_type'] == 'train']
    logger.info(f'Shape train data {train_data.shape}. Min date {train_data.index.min()} and max date {train_data.index.max()}')
    test_data = ml_data[ml_data['data_type'] == 'validation']
    logger.info(f'Shape validation data {test_data.shape}. Min date {test_data.index.min()} and max date {test_data.index.max()}')

    logger.info(f'Train data, found {train_data[config.target_name].isna().sum()} rows without target')
    logger.info(f'Validation data, found {test_data[config.target_name].isna().sum()} rows without target')

    # generate live data
    date_string = last_friday.strftime('%Y-%m-%d')
    live_data = ticker_data[ticker_data.date == date_string].copy()

    ################ Start getting data from the previous day ################ 
    # get data from the day before, for markets that were closed
    last_thursday = last_friday - relativedelta(days=1)
    thursday_date_string = last_thursday.strftime('%Y-%m-%d')
    thursday_data = ticker_data[ticker_data.date == thursday_date_string].copy()

    logger.info(f'Last friday: {date_string} and last thursday: {thursday_date_string}')

    # Only select tickers than aren't already present in live_data
    thursday_data = thursday_data[
        ~thursday_data.bloomberg_ticker.isin(live_data.bloomberg_ticker.values)
    ].copy()

    live_data = pd.concat([live_data, thursday_data])
    ################ Finished getting data from the previous day ################

    live_data['last_friday'] = date_string #Live data needs to have 'last friday' not to crash
    live_data = live_data.set_index('date')
    logger.info(f'Live data, num rows before dropping NaN: {live_data.shape}')
    live_data.dropna(subset=feature_names, inplace=True)
    logger.info(f'Live data, num rows after dropping NaN: {live_data.shape}')
    logger.info(f"NUMBER OF LIVE TICKETS TO SUBMIT WITH ALL DATA: {live_data.shape[0]}")

    return train_data, test_data, live_data
    
def create_colab_csv_io(db_input, db_output, config):
    train_data, test_data, live_data = get_data(db_input, config)
    train_data['date'] = train_data.index
    test_data['date'] = test_data.index
    live_data['date'] = live_data.index
    live_data['data_type'] = 'live'

    full_data = pd.concat([train_data, test_data, live_data])

    clean_full_data = pd.DataFrame()
    clean_full_data['date'] = full_data["date"]
    clean_full_data['bloomberg_ticker'] = full_data["bloomberg_ticker"]
    clean_full_data[config.target_name] = full_data[config.target_name]
    clean_full_data['data_type'] = full_data["data_type"]
    clean_full_data['friday_date'] = full_data["friday_date"]
    feature_names = [f for f in train_data.columns if f.startswith("feature")]
    for f in feature_names:
        clean_full_data[f] = full_data[f] 
    clean_full_data.to_csv(db_output / f'{ML_DATA_FILE}', index=False)







