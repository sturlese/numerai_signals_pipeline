import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from app.utils.utils import read_csv
from app.utils.name_utils import TA_FEATURE_PREFIX, ML_DATA_FILE
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numerapi
import logging

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

PREDICTION_NAME = 'signal'
MAX_CORR_ALLOWED = 0.96

def create_corrs(tmp_train_era, fs):
    tmp_train_era = tmp_train_era[fs].copy()
    tmp_train_era.reset_index(drop=True, inplace=True)
    
    corr_matrix = tmp_train_era.corr().abs()
    return corr_matrix

#calculates features corr per era (date)
def get_features_to_delete(train_data, feature_names, maximum): 
    logger.info("Removing correlated features per era") 
    train_eras = train_data.groupby('date')
    num_eras = len(train_eras)
    corrs = []
    
    for not_used, train_era in train_eras:
        tmp_train_era = train_era.copy()
        tmp_corr_matrix = create_corrs(tmp_train_era, feature_names)
        corrs.append(tmp_corr_matrix)
            
    corr_acc = corrs[0].copy(deep=True)
    for col in corr_acc.columns:
        corr_acc[col].values[:] = 0

    for curr_corr in corrs:
        corr_acc = corr_acc.add(curr_corr, fill_value=0)
        
    corr_acc = corr_acc.div(num_eras) 
    upper = corr_acc.where(np.triu(np.ones(corr_acc.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > maximum)]
    return to_drop

def remove_corr_features(train_data):
    features_corr_candidates = [f for f in train_data.columns if f.startswith(f"feature_{TA_FEATURE_PREFIX}_")]
    feature_names_all = [f for f in train_data.columns if f.startswith("feature")]
    logger.info(f"Original num features {len(feature_names_all)}")    
    train_data = train_data[["date"] + features_corr_candidates].copy()
    drop_features = get_features_to_delete(train_data, features_corr_candidates, MAX_CORR_ALLOWED)
    logger.info(f"Going to drop {len(drop_features)} features: {len(drop_features)}")
    final_features = [item for item in feature_names_all if item not in drop_features]
    logger.info(f'Num features after removing the correlated ones {len(final_features)}')
    return final_features

def build_preds_file(test_data, live_data, predictions_path, last_friday):
    diagnostic_df = pd.concat([test_data, live_data])
    diagnostic_df = diagnostic_df.rename(columns={'bloomberg_ticker': 'numerai_ticker'})
    diagnostic_df['friday_date'] = diagnostic_df.friday_date.fillna(
        last_friday.strftime('%Y%m%d')).astype(int)
    diagnostic_df['data_type'] = diagnostic_df.data_type.fillna('live')
    diagnostic_df[['numerai_ticker', 'friday_date', 
        'data_type','signal']].reset_index(drop=True).to_csv(predictions_path, index=False)
    logger.info('Signals file ready to submit')

def run_prediction(db_input, db_predictions, submission_config, data_config):
    full_data = read_csv(db_input / ML_DATA_FILE)
    full_data.head()

    train_data = full_data[full_data.data_type == 'train']
    validation_data  = full_data[full_data.data_type == 'validation']
    live_data = full_data[full_data.data_type == 'live']
    logger.info("Train shape: " + str(train_data.shape))
    logger.info("Validation shape: " + str(validation_data.shape))
    logger.info("Live shape: " + str(live_data.shape))

    feature_names = remove_corr_features(train_data.copy(deep=True))

    model = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, n_jobs=-1, colsample_bytree=0.1, tree_method='gpu_hist', predictor='gpu_predictor', random_state=1, silent=True)
    model.fit(train_data[feature_names], train_data[data_config.target_name])
    logger.info("Model trained")
    validation_data[PREDICTION_NAME] = model.predict(validation_data[feature_names])
    live_data[PREDICTION_NAME] = model.predict(live_data[feature_names])

    predictions_path = db_predictions / 'predictions_val_live.csv'
    last_friday = datetime.now() + relativedelta(weekday=data_config.delta_day(-1))
    build_preds_file(validation_data, live_data, predictions_path, last_friday)

    if submission_config.numerai_submit:
        napi = numerapi.SignalsAPI(submission_config.public_id, submission_config.secret_key)
        napi.upload_predictions(predictions_path, model_id=submission_config.model_id)
