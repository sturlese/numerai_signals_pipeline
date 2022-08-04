import pandas as pd
import logging
import numpy as np
import csv
import logging

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

def read_csv(file_path):
    with open(file_path, 'r') as f:
        column_names = next(csv.reader(f))

    dtypes = {x: np.float16 for x in column_names if x.startswith(('feature', 'target'))}
    df = pd.read_csv(file_path, dtype=dtypes, header=0)
    return df

def get_ticker_data(db_dir): 
    ticker_data = pd.DataFrame({
        'bloomberg_ticker' : pd.Series([], dtype='str'),
        'date' : pd.Series([], dtype='datetime64[ns]')
    })
    if len(list(db_dir.rglob('*.parquet'))) > 0:
        ticker_data = pd.read_parquet(db_dir)

    return ticker_data
def get_ticker_data_cols(db_dir, cols): 
    ticker_data = pd.DataFrame({
        'bloomberg_ticker' : pd.Series([], dtype='str'),
        'date' : pd.Series([], dtype='datetime64[ns]')
    })
    if len(list(db_dir.rglob('*.parquet'))) > 0:
        ticker_data = pd.read_parquet(db_dir, columns=cols)

    return ticker_data
