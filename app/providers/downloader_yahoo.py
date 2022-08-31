from app.utils.utils import get_ticker_data
import shutil
import pandas as pd
import logging
import sys
import time
import numpy as np
import time as _time
import requests
import random
from concurrent import futures
from tqdm import tqdm
from datetime import datetime, date, time
from dateutil.relativedelta import relativedelta, FR
import sys

USER_AGENTS = [
    (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko)'
        ' Chrome/39.0.2171.95 Safari/537.36'
    ),
    (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
        ' Chrome/42.0.2311.135 Safari/537.36 Edge/12.246'
    )
]

import logging
logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)


def get_tickers(config): #just used here so not moved to utils file
    SIGNALS_TICKER_MAP=f'{config.AWS_BASE_URL}/signals_ticker_map_w_bbg.csv'
    ticker_map = pd.read_csv(SIGNALS_TICKER_MAP)
    ticker_map = ticker_map.dropna(subset=['yahoo'])
    logger.info(f'Number of eligible tickers: {ticker_map.shape[0]}')

    if ticker_map['yahoo'].duplicated().any():
        raise Exception(
            f'Found duplicated {ticker_map["yahoo"].duplicated().values().sum()}'
            ' yahoo tickers'
        )

    if ticker_map['bloomberg_ticker'].duplicated().any():
        raise Exception(
            f'Found duplicated {ticker_map["bloomberg_ticker"].duplicated().values().sum()}'
            ' bloomberg_ticker tickers'
        )

    return ticker_map

def get_ticker_missing( #just used here so not moved to utils file
    ticker_data, ticker_map, last_friday = datetime.today() - relativedelta(weekday=FR(-1))
):
    tickers_available_data = ticker_data.groupby('bloomberg_ticker').agg({'date': [max, min]})
    tickers_available_data.columns = ['date_max', 'date_min']

    eligible_tickers_available_data = ticker_map.merge(
        tickers_available_data.reset_index(),
        on='bloomberg_ticker',
        how='left'
    )

    ticker_not_found = eligible_tickers_available_data.loc[
        eligible_tickers_available_data.date_max.isna(), ['bloomberg_ticker', 'yahoo']
    ]

    ticker_not_found['start'] = '2002-12-01'

    last_friday_20 = last_friday - relativedelta(days=20)
    tickers_outdated = eligible_tickers_available_data.loc[
        (
            (eligible_tickers_available_data.date_max < last_friday.strftime('%Y-%m-%d'))
        ),
        ['bloomberg_ticker', 'yahoo', 'date_max']
    ]

    tickers_outdated['start'] = (
        tickers_outdated['date_max'] + pd.DateOffset(1)
    ).dt.strftime('%Y-%m-%d')
    tickers_outdated.drop(columns=['date_max'], inplace=True)

    result = pd.concat([ticker_not_found, tickers_outdated])
    return result

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def download_data(db_dir, config):
    logging.info(f' Recreated folder {db_dir} from scratch ')
    shutil.rmtree(db_dir, ignore_errors=True)

    db_dir.mkdir(exist_ok=True)

    ticker_data = get_ticker_data(db_dir) #from my folder
    ticker_map = get_tickers(config) #map bloomberg yahoo from url
    ticker_missing = get_ticker_missing(ticker_data, ticker_map) #map ticker - date start download

    n_ticker_missing = ticker_missing.shape[0]
    if n_ticker_missing <= 0:
        logger.info(f'WARNING - No data to download. Stopping the program')
        sys.exit()
        return

    logger.info(f'Downloading data for {n_ticker_missing} tickers')

    ticker_missing_grouped = ticker_missing.groupby('start').apply(
        lambda x: ' '.join(x.yahoo.astype(str))
    )

    for start_date, tickers_str in ticker_missing_grouped.iteritems():

        tickers_large = tickers_str.split(' ')
        tickers_list = chunks(tickers_large, 750)
        
        for tickers in tickers_list:
            logger.info(f'Downloading {len(tickers)}')

            temp_df = download_tickers(tickers, start=start_date)
            # Yahoo Finance returning previous day in some situations (e.g. Friday in TelAviv markets)
            temp_df = temp_df[temp_df.date >= start_date]
            if temp_df.empty:
                continue

            temp_df['created_at'] = datetime.now()
            temp_df['bloomberg_ticker'] = temp_df['bloomberg_ticker'].map(
                dict(zip(ticker_map['yahoo'], ticker_map['bloomberg_ticker'])))

            temp_df_clean = clean_up(temp_df)
            temp_df_clean = adjust(temp_df_clean)
            if(temp_df_clean.shape[0] > 0):
                temp_df_clean.to_parquet(db_dir / f'yahoo-{datetime.utcnow().timestamp()}.parquet', index=False, compression='brotli')
            _time.sleep(3)

def clean_up(df):
    df_out = df.copy(deep=True)
    df_out = df_out[df_out['volume'] > 0].copy(deep=True)
    
    #neg
    df_neg = df_out[df_out['adj_close'] <= 0.0]
    NEG_TICKERS = (df_neg.bloomberg_ticker.unique()).tolist()
    
    #shift
    df_out_tmp = df_out[df_out['adj_close'] > 0].copy()
    df_out_tmp = df_out_tmp[df_out_tmp['volume'] >= 1].copy()

    ticker_groups = df_out_tmp.groupby("bloomberg_ticker")
    df_out_tmp["adj_close_shift_1"] = ticker_groups["adj_close"].shift(1)
    df_out_tmp["close_shift_1"] = ticker_groups["close"].shift(1)

    df_out_tmp = df_out_tmp[df_out_tmp["adj_close_shift_1"] > 0].copy()
    df_out_tmp = df_out_tmp[df_out_tmp["close_shift_1"] > 0].copy()

    df_outliers = df_out_tmp[df_out_tmp['adj_close'] > (df_out_tmp['adj_close_shift_1'] * 10)] 
    OUTLIERS_TICKERS = (df_outliers.bloomberg_ticker.unique()).tolist()

    df_outliers_down = df_out_tmp[(df_out_tmp['adj_close'] * 10) < df_out_tmp['adj_close_shift_1']] 
    OUTLIERS_TICKERS_DOWN = (df_outliers_down.bloomberg_ticker.unique()).tolist()
    
    ticker_list = NEG_TICKERS + OUTLIERS_TICKERS + OUTLIERS_TICKERS_DOWN
    
    mylist = ticker_list
    ticker_list = list(dict.fromkeys(mylist))
    
    logger.info(f'Number of tickers to remove due bad data {len(ticker_list)}')
    df_out = df_out[~df_out['bloomberg_ticker'].isin(ticker_list)].copy()
    return df_out

def adjust(df):
    df['high'] = df['high'] * df['adj_close'] / df['close']
    df['low'] = df['low'] * df['adj_close'] / df['close']
    df['open'] = df['open'] * df['adj_close'] / df['close']
    return df

def download_tickers(tickers, start):
    start_epoch = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
    end_epoch = int(datetime.combine(date.today(), time()).timestamp())

    pbar = tqdm(
        total=len(tickers),
        unit='tickers'
    )

    dfs = {}
    with futures.ThreadPoolExecutor() as executor:
        _futures = []
        for ticker in tickers:
            _futures.append(
                executor.submit(download_ticker, ticker=ticker, start_epoch=start_epoch, end_epoch=end_epoch)
            )

        for future in futures.as_completed(_futures):
            pbar.update(1)
            ticker, data = future.result()
            dfs[ticker] = data

    pbar.close()
    return pd.concat(dfs)

def download_ticker(ticker, start_epoch, end_epoch):
    def empty_df():
        return pd.DataFrame(columns=[
            "date", "bloomberg_ticker",
            "open", "high", "low", "close",
            "adj_close", "volume", "currency", "provider", "country", "sector", "industry"])

    ## start get sector data
    retries = 20
    tries = retries + 1
    backoff = 1
    url = f'https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile'
    user_agent = random.choice(USER_AGENTS)
    country = ""
    sector = ""
    industry = ""

    while(tries > 0):
        tries -= 1
        try:
            data = requests.get(
                url=url,
                headers={'User-Agent': user_agent}
            )
            data_json = data.json()
            info = data_json["quoteSummary"]["result"][0]['assetProfile']
            country = info["country"]
            industry = info["industry"]
            sector = info["sector"]

        except Exception as e:
            user_agent = random.choice(USER_AGENTS)
            _time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    print_data = ""
    if country == "":
        print_data = print_data + " country"

    if sector == "":
        print_data = print_data + " sector"

    if industry == "":
        print_data = print_data + " industry"

    if print_data != "":
        #logger.info(f'{ticker} lacks {print_data}')
        return ticker, empty_df()

    #Finished getting sector, industry, etc data. Now go for price data
    retries = 20
    tries = retries + 1
    backoff = 1

    url = f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}'
    user_agent = random.choice(USER_AGENTS)
    params = dict(
        period1=start_epoch,
        period2=end_epoch,
        interval='1d',
        events='div,splits',
    )
    while(tries > 0):
        tries -= 1
        try:
            data = requests.get(
                url=url,
                params=params,
                headers={'User-Agent': user_agent}
            )
            data_json = data.json()
            quotes = data_json["chart"]["result"][0]
            if "timestamp" not in quotes:
                return ticker, empty_df()

            timestamps = quotes["timestamp"]
            ohlc = quotes["indicators"]["quote"][0]
            volumes = ohlc["volume"]
            opens = ohlc["open"]
            closes = ohlc["close"]
            lows = ohlc["low"]
            highs = ohlc["high"]

            adjclose = closes
            if "adjclose" in quotes["indicators"]:
                adjclose = quotes["indicators"]["adjclose"][0]["adjclose"]
            
            #adjust with a ratio based on close and adj_close
            opens_tmp = np.array(opens, dtype='float32')
            lows_tmp = np.array(lows, dtype='float32')
            highs_tmp = np.array(highs, dtype='float32')
            adjcloses_tmp = np.array(adjclose, dtype='float32')
            closes_tmp = np.array(closes, dtype='float32')

            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s").normalize(),
                "bloomberg_ticker": ticker,
                "open": opens_tmp,
                "high": highs_tmp,
                "low": lows_tmp,
                "close": closes_tmp,
                "adj_close": adjcloses_tmp,
                "volume": np.array(volumes, dtype='float32'),
                "currency": quotes['meta']['currency'],
                "provider": 'yahoo'
            })
    
            df["country"] = country
            df["sector"] = sector
            df["industry"] = industry

            return ticker, df.drop_duplicates().dropna()

        except Exception as e:
            user_agent = random.choice(USER_AGENTS)
            _time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    return ticker, empty_df()
