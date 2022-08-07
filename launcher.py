import app.providers.downloader_yahoo as yahoo_data
from app.feature_builder import feature_engineering_io, compact_files_io
from app.indicator_builder import create_indicators_io
from app.target_builder import create_target_io
from app.ml_dataset_builder import create_colab_csv_io
from app.prediction_builder import run_prediction
from app.indicator_denoiser import denoise_indicators_io
from app.indicator_lagger import create_lagged_indicators_io
import app.folders as folders
from app.configuration import get_data_config, get_submission_config
import numpy as np
import gc
import time
import numerapi
import sys
import json
import logging
import warnings
import pandas as pd

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) 

def needs_submission(config):
    napi = numerapi.SignalsAPI(config.public_id, config.secret_key)
    needs_submission = True
    try:
        status = napi.submission_status(config.model_id)
        logger.info(f"Submission status {status}")
        if status is not None:
            needs_submission = False
    except Exception as e:
        logger.info ("No submission found. Exception: " + str(e))
    return needs_submission

def run_full_pipline(data_config, submission_config):
    if not submission_config.skip_check_needs_submission:
        if not needs_submission(submission_config):
            logger.info("WARNING - This model has already been submitted. Quitting.")
            sys.exit()
        else:
            logger.info("Need submission. Let's run the pipeline")

    start_time_full = time.perf_counter()

    paths = folders.PathSignals(data_config)
    paths.setup()
    paths.create_folders()
    if not data_config.skip_download:
        logger.info("*** DOWNLOADING PHASE...")
        if 'yahoo' in data_config.raw_data:
            yahoo_data.download_data(paths.db_raw, data_config)

    logger.info("*** CREATING CUSTOM TARGET... [currently not used]")
    create_target_io(paths.db_raw, paths.db_target) #currently not used as using Numerai's
    logger.info("*** CREATING INDICATORS...")
    create_indicators_io(paths.db_raw, paths.db_indicators)
    logger.info("*** DENOISING INDICATORS...")
    denoise_indicators_io(paths.db_indicators, paths.db_denoised)
    logger.info("*** CREATING LAGS TO INDICATORS...")
    create_lagged_indicators_io(paths.db_denoised, paths.db_indicators_lagged, paths.db_pickled_cols)
    logger.info("*** FEATURE ENGINEERING...")
    np.seterr('raise')
    feature_engineering_io(paths.db_pickled_cols)
    logger.info("*** COMPRESSING ENGINEERED FEATURES FILES...")
    full_data = compact_files_io(paths.db_engineered, paths.db_packed)
    feature_names = [f for f in full_data.columns if f.startswith("feature")]
    logger.info(f'Total number of features: {len(feature_names)}')
    del full_data
    gc.collect()
    logger.info("*** BUILDING ML DATASET CSV...")
    create_colab_csv_io(paths.db_packed, paths.db_csv, data_config)
    logger.info("*** PREDICTING...")
    run_prediction(paths.db_csv, paths.db_predictions, submission_config, data_config)
    time_taken_full = time.perf_counter() - start_time_full
    time_taken_full = (time_taken_full / 60) / 60
    logger.info("TIME TAKEN RUNNING FULL PIPELINE " + str(time_taken_full) + " hours")
    paths.cleanup()

try:
    properties_file_name = sys.argv[1]
    f = open(properties_file_name)
    properties = json.load(f)
    f.close()
except:
  print("Wrong properties file as argument!")

data_config = get_data_config(False)
submission_config = get_submission_config(properties)

run_full_pipline(data_config, submission_config)
