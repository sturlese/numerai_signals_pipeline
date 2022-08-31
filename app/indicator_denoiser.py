from app.utils.utils import get_ticker_data
from app.utils.name_utils import TA_FEATURE_PREFIX
import logging
import gc
import sys

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)

def denoise_indicators_io(db_input, db_output):
    raw_data_df = get_ticker_data(db_input)
    feature_to_denoise = [f for f in raw_data_df.columns if f.startswith(TA_FEATURE_PREFIX)]
    logger.info (f"Num features to denoise {len(feature_to_denoise)}")

    #shouldn't be done here
    label_encodings = ["country", "industry", "sector", "currency"]
    raw_data_df.dropna(inplace=True, subset=label_encodings)

    upper_thresh = 0.999
    lower_thresh = 0.001

    logger.info(f"Shape of the df BEFORE denoising {raw_data_df.shape}")
    logger.info(f"")

    df_result = raw_data_df.copy()
    #remove extreme outlier rows
    for feature in feature_to_denoise:
        start_rows = raw_data_df.shape[0]
        upper_q = raw_data_df[feature].quantile(upper_thresh)
        lower_q = raw_data_df[feature].quantile(lower_thresh)    
        df_result = df_result.query('@lower_q < ' + feature + ' < @upper_q')
        end_rows = df_result.shape[0]
        deleted_rows = start_rows - end_rows
        pct_dropped = round((start_rows/end_rows-1)*100,1)
        logger.info(f"Initial rows are {start_rows}. Deleted {deleted_rows} rows for feature {feature} representing {pct_dropped}% of the total rows.")
        gc.collect()

    logger.info(f"")
    logger.info(f"Shape of the df AFTER denoising {df_result.shape}")
    rows_before_denoising = raw_data_df.shape[0]
    rows_after_denoising = df_result.shape[0]
    dropped_rows = rows_before_denoising - rows_after_denoising
    pct_rows_dropped = round((rows_before_denoising/rows_after_denoising-1)*100,1)
    logger.info(f"Dropped {pct_rows_dropped}% of rows. A total of {dropped_rows}.")

    df_result.sort_values(by=['date', 'bloomberg_ticker'], inplace=True, ascending=(True, True))

    #replace outliers by quantile values
    for feature in feature_to_denoise:
        df_result[feature] = df_result[feature].transform(lambda x: x.clip(*x.quantile([0.005, 0.995])))

    df_result.sort_values(by=['date', 'bloomberg_ticker'], inplace=True, ascending=(True, True))
    df_result.to_parquet(db_output / f'denoised_data.parquet', index=False, compression='brotli')
