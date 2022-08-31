from distutils.command.config import config
from app.utils.utils import get_ticker_data, parquet_concat
import pandas as pd
import logging
import gc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from app.configuration import get_indicator_config
import logging
import sys

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)

def build_ticker_indicators(ticker_df):
    conf = get_indicator_config()
    global_indicator_list = conf.get_indicator_list()

    tiny_ticker_df = pd.DataFrame()
    tiny_ticker_df['bloomberg_ticker'] = ticker_df['bloomberg_ticker']
    tiny_ticker_df['country'] = ticker_df['country']
    tiny_ticker_df['sector'] = ticker_df['sector']
    tiny_ticker_df['industry'] = ticker_df['industry']
    tiny_ticker_df['currency'] = ticker_df['currency']
    tiny_ticker_df['date'] = ticker_df.index
    tiny_ticker_df.set_index('date',inplace=True)
    tiny_ticker_df['date'] = tiny_ticker_df.index
    for wrapped_indicator in global_indicator_list:
        tiny_ticker_df = wrapped_indicator.build_indicators(tiny_ticker_df, ticker_df)

    return tiny_ticker_df

def build_indicators_dfs(df_input):
    ticker_group = df_input.groupby('bloomberg_ticker')
    dfs_parallel = []

    for not_used, ticker_df in ticker_group:
        tmp_ticker_df = ticker_df.copy(deep=True)
        tmp_ticker_df.sort_index(inplace=True, ascending=True)
        dfs_parallel.append(tmp_ticker_df)

    gc.collect()
    logger.info(f"Found {cpu_count()} cpus")
        
    with Pool(cpu_count() * 2) as p:
        feature_dfs = list(tqdm(p.imap(build_ticker_indicators, dfs_parallel), total=len(dfs_parallel)))

    return feature_dfs

def create_indicators_io(db_input, db_output):
    raw_data = get_ticker_data(db_input)
    raw_data = raw_data.set_index('date')
    raw_data['date'] = raw_data.index
    dfs = build_indicators_dfs(raw_data)
    path = db_output / f'indicators_data.parquet'
    parquet_concat(dfs, path)
