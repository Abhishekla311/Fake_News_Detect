import os
import sys
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from utils.custom_exception import CustomException
from utils.logger import get_logger
from config.paths_config import *
from utils.custom_functions import read_yaml

# Ensure stopwords are downloaded quietly
nltk.download('stopwords', quiet=True)

logger = get_logger(__name__)


class FeatureEngineering:
    def __init__(self, config):
        try:
            self.config_pr = config["data_preprocessing"]
            self.ps = PorterStemmer()
            self.stop_words = set(stopwords.words(self.config_pr["text_cleaning"]["language"]))
            logger.info("FeatureEngineering component initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def clean_text_structure(self, text):
        """Cleans input text by removing non-alphabetic characters and applying stemming."""
        if not isinstance(text, str):
            return ""
        text = re.sub('[^a-zA-Z]', ' ', text).lower()
        words = text.split()
        stemmed_words = [self.ps.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(stemmed_words)

    def run_feature_building(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Creates the 'content' column from 'title' and 'text' and cleans textual structures."""
        try:
            logger.info("Feature engineering: Transforming textual structure")
            
            if self.config_pr.get("create_content_feature", True):
                # Ensure title and text columns exist, fill NaNs to avoid missing values
                train_title = train_df['title'].fillna('') if 'title' in train_df.columns else ""
                train_text = train_df['text'].fillna('') if 'text' in train_df.columns else ""
                
                test_title = test_df['title'].fillna('') if 'title' in test_df.columns else ""
                test_text = test_df['text'].fillna('') if 'text' in test_df.columns else ""

                # Combine title and text into 'content' column
                train_df['content'] = (train_title + " " + train_text).str.strip()
                test_df['content'] = (test_title + " " + test_text).str.strip()

                logger.info("Applying text cleaning and stemming across content columns...")
                train_df["content"] = train_df["content"].apply(self.clean_text_structure)
                test_df["content"] = test_df["content"].apply(self.clean_text_structure)

                logger.info("Feature construction and text stemming completed.")
                return train_df, test_df

            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)


# Standalone execution block for testing
if __name__ == "__main__":
    try:
        # 1. Load configuration
        config_data = read_yaml(CONFIG_PATH)

        # 2. Load preprocessed or raw datasets
        logger.info(f"Loading data from raw/input paths")
        train_df = pd.read_csv(TRAIN_FILE_PATH, low_memory=False)
        test_df = pd.read_csv(TEST_FILE_PATH, low_memory=False)

        # 3. Run feature engineering
        fe = FeatureEngineering(config_data)
        train_fe, test_fe = fe.run_feature_building(train_df, test_df)

        # 4. Display result summary
        print("\n" + "=" * 45)
        print("🎉 Feature Engineering Successful!")
        print(f"Train Shape: {train_fe.shape} | Test Shape: {test_fe.shape}")
        print("\n--- Engineered 'content' Column Preview ---")
        print(train_fe['content'].head(2).values)
        print("=" * 45 + "\n")

        # 5. Save processed output files
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        train_fe.to_csv(PROCESSED_TRAIN_DATA_PATH, index=False)
        test_fe.to_csv(PROCESSED_TEST_DATA_PATH, index=False)
        logger.info("Engineered features successfully saved to processed directory.")

    except Exception as e:
        print(f"Feature Engineering Execution Failed: {e}")