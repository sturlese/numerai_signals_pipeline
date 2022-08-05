from app.utils.utils import get_ticker_data
from app.utils.name_utils import TA_FEATURE_PREFIX
import logging
import gc

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

def denoise_indicators_io(db_input, db_output):
    raw_data_df = get_ticker_data(db_input)
    raw_data_df = raw_data_df.set_index('date')
    raw_data_df['date'] = raw_data_df.index
    feature_to_denoise = [f for f in raw_data_df.columns if f.startswith(TA_FEATURE_PREFIX)]

    #shouldn't be done here
    must_have_fields = ["country", "industry", "sector", "currency"]
    raw_data_df.dropna(inplace=True, subset=must_have_fields)

    upper_thresh = 0.999
    lower_thresh = 0.001

    logger.info(f"Shape of the df BEFORE denoising {raw_data_df.shape}")

    df_result = raw_data_df.copy()
    for feature in feature_to_denoise:
        upper_q = raw_data_df[feature].quantile(upper_thresh)
        lower_q = raw_data_df[feature].quantile(lower_thresh)    
        df_result = df_result.query('@lower_q < ' + feature + ' < @upper_q')
        gc.collect()

    logger.info(f"Shape of the df AFTER denoising {df_result.shape}")
    rows_before_denoising = raw_data_df.shape[0]
    rows_after_denoising = df_result.shape[0]
    dropped_rows = rows_before_denoising - rows_after_denoising
    pct_rows_dropped = round((rows_before_denoising/rows_after_denoising-1)*100,1)
    logger.info(f"Dropped {pct_rows_dropped}% of rows. A total of {dropped_rows}.")

    df_result.to_parquet(db_output / f'denoised_data.parquet', index=False, compression='brotli')
