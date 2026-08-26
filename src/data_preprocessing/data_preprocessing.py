import os
import sys
import pandas as pd
from utils.custom_exception import CustomException
from utils.logger import get_logger
from config.paths_config import *
from utils.custom_functions import read_yaml

logger = get_logger(__name__)

class DataPreprocessing:
    def __init__(self, config):
        try:
            self.config_pre = config["data_preprocessing"]
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            logger.info("Data Preprocessing component initialized.")
        except Exception as e:
            raise CustomException(e, sys)

    def clear_missing_value(self):
        try:
            logger.info("Data Preprocessing: Handling missing values and cleaning unneeded columns")
            
          
            train_df = pd.read_csv(TRAIN_FILE_PATH, low_memory=False)
            test_df = pd.read_csv(TEST_FILE_PATH, low_memory=False)

       
            train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
            test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

            # missing values भरना
            if self.config_pre["impute_missing_value"]:
                fill_val = self.config_pre["impute_value"]
                for col in self.config_pre["text_columns_to_impute"]:
                    if col in train_df.columns:
                        train_df[col] = train_df[col].fillna(fill_val)
                    if col in test_df.columns:
                        test_df[col] = test_df[col].fillna(fill_val)

            # 🔥 असली फिक्स: डेटा को डिस्क पर 'processed' फ़ोल्डर में परमानेंट सेव क
            train_df.to_csv(PROCESSED_TRAIN_DATA_PATH, index=False, header=True)
            test_df.to_csv(PROCESSED_TEST_DATA_PATH, index=False, header=True)

            logger.info(f"Processed train data saved to {PROCESSED_TRAIN_DATA_PATH}")
            logger.info(f"Processed test data saved to {PROCESSED_TEST_DATA_PATH}")
            
            return PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH

        except Exception as e:
            logger.error("Error occurred during data preprocessing stage")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        config_data = read_yaml(CONFIG_PATH)
        preprocessing_pipeline = DataPreprocessing(config_data)
        test_path, train_path = preprocessing_pipeline.clear_missing_value()
        
    
        print("🎉 Preprocessing Stage Completed Successfully!")
        print(f"Clean Train Saved: {train_path}")
        print(f"Clean Test Saved: {test_path}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Preprocessing runtime failed: {e}")
