from pathlib import Path
import os
import logging

logger = logging.getLogger()
log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
logging.basicConfig(format=log_format, level=logging.INFO)

class PathSignals():
    root_path = ""
    static_data = False
    db_tmp = None
    db_raw = None
    db_target = None
    db_indicators = None
    db_denoised = None
    db_indicators_lagged = None
    db_engineered = None
    db_packed = None
    db_csv = None
    list_paths = []

    def __init__(self, config):
        self.root_path = config.ROOT_PATH
        self.static_data = config.static_data

    def setup(self):
        self.db_tmp = Path(f'{self.root_path}/db_tmp')
        self.db_raw = Path(f'{self.root_path}/db_raw_downloaded')
        self.db_static_data = Path(f'{self.root_path}/db_static_data/yahoo') #if does not contain data pipeline can't run
        if self.static_data:
            self.db_raw = Path(f'{self.root_path}/db_static_data/yahoo')
        self.db_target = Path(f'{self.root_path}/db_targets')
        self.db_indicators = Path(f'{self.root_path}/db_indicators')
        self.db_denoised = Path(f'{self.root_path}/db_denoised')
        self.db_indicators_lagged = Path(f'{self.root_path}/db_indicators_lagged')
        self.db_engineered = Path(f'{self.root_path}/db_features')
        self.db_packed = Path(f'{self.root_path}/db_features_packed')
        self.db_csv = Path(f'{self.root_path}/db_ml_csv')
        self.db_predictions = Path(f'{self.root_path}/db_predictions')
        self.db_pickled_cols = Path(f'{self.root_path}/db_feature_column_names')
        #leaves raw yahoo downloaded files, predictions file and the ML csv (with train, val and live data). Need to remove them manually later.
        self.paths_to_clean = [self.db_target, self.db_indicators, self.db_denoised, self.db_indicators_lagged, self.db_engineered, self.db_packed, self.db_pickled_cols]
        self.paths_to_create = [self.db_static_data, self.db_tmp, self.db_raw, self.db_target, self.db_indicators, self.db_denoised, self.db_indicators_lagged, self.db_engineered, self.db_packed, self.db_pickled_cols, self.db_csv, self.db_predictions]

    def cleanup(self):
        for f in self.paths_to_clean:
            arr = os.listdir(f)
            for e in arr:
                    path = str(f) + "/" + str(e)
                    logger.info("Deleting file " + str(path))
                    file_to_rem = Path(path, missing_ok=True)
                    file_to_rem.unlink()

    def create_folders(self):
        for f in self.paths_to_create:
            if not os.path.exists(f):
                os.makedirs(f)

