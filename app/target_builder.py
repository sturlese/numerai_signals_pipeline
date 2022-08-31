from app.utils.utils import get_ticker_data
import pandas as pd
import logging
from datetime import datetime
from datetime import datetime
import logging
import sys

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)

def build_target(prices):
    prices_change = prices.pct_change(5)
    pct_change_ranked = prices_change.rank(pct=True, method="first")
    target_raw = pct_change_ranked.shift(periods=-5)
    return target_raw

def train_val(df):
    df['date_obj'] = pd.to_datetime(df['date'],format='%Y-%m-%d')
    date_str = '20121228'
    date_time_obj = datetime.strptime(date_str, '%Y%m%d')
    val_data = df[df['date_obj'] > date_time_obj]
    val_data = val_data[val_data['target_custom'].notnull()]
    val_data['data_type'] = 'validation'
    train_data = df[df['date_obj'] < date_time_obj]
    train_data = train_data[train_data['target_custom'].notnull()]
    train_data['data_type'] = 'train'
    data_out = pd.concat([train_data, val_data], ignore_index=True)

    data_out['friday_date'] = data_out['date_obj'].dt.strftime('%Y%m%d')
    data_out.drop('date_obj', axis=1, inplace=True)
    return data_out
    
def create_target(df_input):
    ticker_groups = df_input.groupby('bloomberg_ticker')
    df_input['target_raw'] = ticker_groups['adj_close'].transform(lambda x: build_target(x))
    
    date_groups = df_input.groupby('date')
    df_input['target_custom'] = date_groups['target_raw'].transform(
        lambda group: pd.cut(
            group,
            bins=[0, 0.05, 0.25, 0.75, 0.95, 1],
            right=True,
            labels=[0, 0.25, 0.50, 0.75, 1],
            include_lowest=True)    
        )
    df_target = pd.DataFrame()
    df_target['date'] = df_input['date']
    df_target['bloomberg_ticker'] = df_input['bloomberg_ticker']
    df_target['target_custom'] = df_input['target_custom']

    data_out = train_val(df_target)
    return data_out

def create_target_io(db_input, db_output):
    full_data = get_ticker_data(db_input)
    full_data.sort_values(by=['date', 'bloomberg_ticker'], inplace=True, ascending=(True, True))
    df_target = create_target(full_data)
    df_target.to_parquet(db_output / f'data-target.parquet', index=False, compression='brotli')
