from app.utils.utils import get_ticker_data, parquet_concat
import logging
import gc
from tqdm import tqdm
import pickle
from app.utils.name_utils import COLUMNS_LAGGED_FILE, INDICATOR_STATIC, INDICATOR_DYAMIC, TA_FEATURE_PREFIX
from multiprocessing import Pool, cpu_count
from app.configuration import get_indicator_config
import logging
import sys

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)

def add_lagged_indicators(tiny_ticker_df, lagged_indicator_objects, indicator_name):
    for indicator_object in lagged_indicator_objects:
        if indicator_name == indicator_object.get_name():
            if indicator_object.get_type() == INDICATOR_DYAMIC:
                for lag in indicator_object.get_lags():
                    val = int(indicator_object.get_interval() * lag)
                    tiny_ticker_df[f'{indicator_name}_lag_{val}'] = tiny_ticker_df[indicator_name].shift(val).astype('float32')
                return tiny_ticker_df
            elif indicator_object.get_type() == INDICATOR_STATIC:
                for lag in indicator_object.get_lags():
                    val = int(lag)
                    tiny_ticker_df[f'{indicator_name}_lag_{val}'] = tiny_ticker_df[indicator_name].shift(val).astype('float32')
                return tiny_ticker_df
            else:
                logger.info(f"ERROR: Wrong type of indicator {indicator_object.get_type()}. Going to exit the program.")
                sys.exit()

    logger.info(f"ERROR: Indicator {indicator_name} not found. Going to exit the program.")
    sys.exit()

def build_lagged_indicators(ticker_df):
    conf = get_indicator_config()
    lagged_indicator_objects = conf.get_lagged_indicator_list()

    all_columns_list = ticker_df.columns.values.tolist()
    indicator_column_names = []
    for col in all_columns_list:
        if col.startswith(f'{TA_FEATURE_PREFIX}_'):
            indicator_column_names.append(col)

    tiny_ticker_df = ticker_df.copy()
    tiny_ticker_df['date'] = ticker_df.index
    tiny_ticker_df.set_index('date',inplace=True)
    tiny_ticker_df['date'] = tiny_ticker_df.index

    for indicator_name in indicator_column_names:
        tiny_ticker_df = add_lagged_indicators(tiny_ticker_df, lagged_indicator_objects, indicator_name)
    
    return tiny_ticker_df

def lag_indicators_dfs(df_input):
    ticker_group = df_input.groupby('bloomberg_ticker')
    dfs_parallel = []

    for not_used, ticker_df in ticker_group:
        tmp_ticker_df = ticker_df.copy(deep=True)
        tmp_ticker_df.sort_index(inplace=True, ascending=True)
        dfs_parallel.append(tmp_ticker_df)

    gc.collect()
    logger.info(f"Found {cpu_count()} cpus")
        
    with Pool(cpu_count() * 2) as p:
        feature_dfs = list(tqdm(p.imap(build_lagged_indicators, dfs_parallel), total=len(dfs_parallel)))

    return feature_dfs

def create_lagged_indicators_io(db_input, db_output, db_pickle):
    raw_data = get_ticker_data(db_input)
    raw_data = raw_data.set_index('date')
    raw_data['date'] = raw_data.index
    dfs = lag_indicators_dfs(raw_data)
    path = db_output / f'indicators_lagged_data.parquet'
    pickle_cols(dfs, db_pickle)
    parquet_concat(dfs, path)

def pickle_cols(dfs, db_pickle):
    pickled_cols = set()
    for df in dfs:
        col_list = df.columns.values.tolist()
        for col in col_list:
            if col.startswith(TA_FEATURE_PREFIX):
                pickled_cols.add(col)
    pickled_cols = sorted(list(pickled_cols))
    logger.info(f"Pickling columns {pickled_cols}")
    pickle.dump(pickled_cols, open(f"{db_pickle}/{COLUMNS_LAGGED_FILE}", 'wb'), protocol=-1)
    
