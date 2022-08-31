import pandas as pd
import logging
import numpy as np
import csv
import logging
import pyarrow as pa
import pyarrow.parquet as pq
import sys

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)

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

def parquet_concat(dfs, path):
    pqwriter = None
    first_time = True
    logger.info(f'Persisting dataframes...')
    for df in dfs:
        df.reset_index(drop=True, inplace=True)
        table = pa.Table.from_pandas(df)
        if first_time:
            pqwriter = pq.ParquetWriter(path, table.schema, compression='brotli')
            first_time = False
        pqwriter.write_table(table)
    if pqwriter:
        pqwriter.close()