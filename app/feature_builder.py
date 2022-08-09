from app.utils.utils import get_ticker_data_cols, get_ticker_data
import pandas as pd
import logging
import gc
import pickle
from sklearn import preprocessing
import app.folders as folders
import datetime as dt
from multiprocessing import Pool, cpu_count
from app.configuration import get_data_config
from tqdm import tqdm
import logging
from app.utils.name_utils import LABEL_FEATURE_PREFIX, COLUMNS_LAGGED_FILE, TRANSFORMATION_BIN, TRANSFORMATION_ZSCORE, TRANSFORMATION_RANK
import sys

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

ENCODES = 'ENCODES'
NORMALIZERS = 'NORMALIZERS'

def get_ticker_data_file(file_path): 
    ticker_data = pd.DataFrame({
        'bloomberg_ticker' : pd.Series([], dtype='str'),
        'date' : pd.Series([], dtype='datetime64[ns]')
    })
    ticker_data = pd.read_parquet(file_path)
    return ticker_data

def check_dups(df):
    df_out = df[df.duplicated(['date', 'bloomberg_ticker'])]
    num_dups = df_out.shape[0]
    if num_dups > 0:
        logger.info("Num duplicates found: " + str(df_out))
        df.drop_duplicates(inplace=True)
        raise Exception("CRITICAL: Found duplicate rows. Stopping execution")

def create_normalizers(full_data, config_data):
    indicators = full_data.columns.values.tolist()  
    indicators.remove('bloomberg_ticker')
    
    date_groups = full_data.groupby(full_data.index)
    for feature in indicators:
        indicator_name = feature
        full_feature_name = f'feature_{indicator_name}_normal'
        try:
            if config_data.TRANSFORMATION_TYPE == TRANSFORMATION_BIN:
                full_data[full_feature_name] = date_groups[indicator_name].transform(
                    lambda group: pd.qcut(group, 9, labels=False, duplicates='drop')).astype('float32')
            elif config_data.TRANSFORMATION_TYPE == TRANSFORMATION_ZSCORE:
                full_data[full_feature_name] = date_groups[indicator_name].transform(
                    lambda x: (x - x.mean())/x.std(ddof=0)).astype('float32')
            elif config_data.TRANSFORMATION_TYPE == TRANSFORMATION_RANK:
                full_data[full_feature_name] = date_groups[indicator_name].transform(
                    lambda group: group.rank()).astype('float32')
            else:
                logger.info(f"Unrecognised transformation type {config_data.TRANSFORMATION_TYPE}")
                sys.exit()
        except:
            logger.info(f'WARNING, could not do quintiles for {full_feature_name}. It has been excluded!')

    return full_data

def create_encodes(full_data):
    label_encoder = preprocessing.LabelEncoder()
    full_data[f'feature_{LABEL_FEATURE_PREFIX}_industry'] = label_encoder.fit_transform(full_data['industry'])
    label_encoder = preprocessing.LabelEncoder()
    full_data[f'feature_{LABEL_FEATURE_PREFIX}_country'] = label_encoder.fit_transform(full_data['country']) 
    label_encoder = preprocessing.LabelEncoder()
    full_data[f'feature_{LABEL_FEATURE_PREFIX}_sector'] = label_encoder.fit_transform(full_data['sector'])
    label_encoder = preprocessing.LabelEncoder()
    full_data[f'feature_{LABEL_FEATURE_PREFIX}_currency'] = label_encoder.fit_transform(full_data['currency'])    
    return full_data 

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

def encodes(features_batch_list):
    build_features(features_batch_list, ENCODES)

def normalizers(features_batch_list):
    build_features(features_batch_list, NORMALIZERS)

def build_features(features_batch_list, action):
    full_data = pd.DataFrame()
    full_data_io = pd.DataFrame()

    config_data = get_data_config(False)
    paths = folders.PathSignals(config_data)
    paths.setup()
    unique_id = dt.datetime.utcnow().timestamp()

    cols = ['bloomberg_ticker', 'date']
    for ele in features_batch_list:
        cols.append(ele)
 
    full_data = get_ticker_data_cols(paths.db_indicators_lagged, cols)
    full_data['date'] = pd.to_datetime(full_data['date'],format='%Y%m%d')
    full_data = full_data.set_index('date')
    full_data.sort_index(inplace=True, ascending=True)

    if action == NORMALIZERS:
        full_data = create_normalizers(full_data, config_data)
    elif action == ENCODES:
        full_data = create_encodes(full_data)
    else:
        raise Exception(f"CRITICAL: Unrecognized action {action}")
    
    full_data['date'] = full_data.index
    full_data = full_data[(full_data.index.weekday == config_data.week_day) | (full_data.index.weekday == (config_data.week_day - 1))] #to save time and space we do this here
    col_list = [f for f in full_data.columns if f.startswith("feature")]
    col_list= ['bloomberg_ticker', 'date'] + col_list
    full_data_io = full_data[col_list].copy()
    
    full_data_io.to_parquet(paths.db_engineered / f'feature_data_partial-{unique_id}.parquet', index=False, compression='brotli')
    del full_data
    del full_data_io
    gc.collect()  

def feature_engineering_io(db_pickle):
    logger.info("Engineering features...")
    full_indicator_list = pickle.load(open(f"{db_pickle}/{COLUMNS_LAGGED_FILE}", 'rb'))
    indicators_batches = chunks(full_indicator_list, 30)

    indicator_batches_list = []
    for btch in indicators_batches:
        indicator_batches_list.append(btch)

    #using threads seems to corrupt parquet files that's why we use a pool of 1
    logger.info("Generating normalization...")
    with Pool(1) as p:
        list(tqdm(p.imap(normalizers, indicator_batches_list), total=len(indicator_batches_list)))
    
    features_batch_list = ["country", "sector", "industry", "currency"]
    logger.info(f"Generating label encodes for {features_batch_list}")
    encodes(features_batch_list)

def compact_files_io(db_input, db_output):
    df_tmp = pd.DataFrame()
    df_all_raw = pd.DataFrame()
    file_list = list(db_input.rglob('*.parquet'))
    logger.info(f"Number of files to process {len(file_list)}")
    count = 0
    first_time = True
    for ffile in file_list:
        count += 1
        gc.collect()
        df_tmp = get_ticker_data_file(ffile)
        df_tmp.sort_values(by=['date', 'bloomberg_ticker'], inplace=True, ascending=(True, True))
        if first_time:
            df_all_raw['bloomberg_ticker'] = df_tmp['bloomberg_ticker']
            df_all_raw['date'] = df_tmp['date']
            first_time = False
        feature_names = [f for f in df_tmp.columns if f.startswith("feature")]
        logger.info(f'Processing file num {count} with {len(feature_names)} features. Name is {ffile}')
        for f in feature_names:
            df_all_raw[f] = df_tmp[f].copy(deep=True)
    check_dups(df_all_raw)

    df_all_raw.to_parquet(db_output / f'feature_data_packed.parquet', index=False, compression='brotli')
    return df_all_raw
