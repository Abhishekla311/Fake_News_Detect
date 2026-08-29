import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config.paths_config import *
from utils.logger import get_logger
from utils.custom_exception import CustomException
from utils.custom_functions import read_yaml


logger = get_logger(__name__)


class FeatureSelection:
    def __init__(self, config):
        try:
            self.target_col = config["data_preprocessing"]["target_column"]
            self.vectorizer = TfidfVectorizer(max_features=5000)
            logger.info("FeatureSelection component structural parameters mapped.")
        except Exception as e:
            raise CustomException(e, sys)

    def apply_selection_and_vectorization(self, train_df, test_df):
        try:
            logger.info("Feature Selection & Array Vectorization started...")

            # सेफ्टी चेक: अगर डेटा प्रीप्रोसेसिंग से 'content' कॉलम नहीं मिला, तो कोड में ऑन-द-फ्लाई बना लें
            if 'content' not in train_df.columns:
                logger.warning("'content' column not found in input. Engineering it now from title and text...")
                train_df['content'] = train_df['title'].fillna(" ") + " " + train_df['text'].fillna(" ")
                test_df['content'] = test_df['title'].fillna(" ") + " " + test_df['text'].fillna(" ")

            # Missing target values हटाना
            train_df = train_df.dropna(subset=[self.target_col])
            test_df = test_df.dropna(subset=[self.target_col])

            train_df['content'] = train_df['content'].fillna(" ")
            test_df['content'] = test_df['content'].fillna(" ")

            X_train_raw = train_df['content'].values.astype(str)
            y_train = train_df[self.target_col].values

            X_test_raw = test_df['content'].values.astype(str)
            y_test = test_df[self.target_col].values

            # Vectorization
            X_train_vec = self.vectorizer.fit_transform(X_train_raw)
            X_test_vec = self.vectorizer.transform(X_test_raw)

            # Binary outputs को सेव करना
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            joblib.dump((X_train_vec, y_train), PROCESSED_TRAIN_DATA_PATH)
            joblib.dump((X_test_vec, y_test), PROCESSED_TEST_DATA_PATH)

            vectorizer_pkl_path = os.path.join(PROCESSED_DIR, "vectorizer.pkl")
            joblib.dump(self.vectorizer, vectorizer_pkl_path)

            logger.info(f"Numeric arrays saved to {PROCESSED_DIR}")
            return PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH

        except Exception as e:
            logger.error("Error in Feature Selection module setup execution")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        config_data = read_yaml(CONFIG_PATH)
        logger.info("Reading structural text dataset states...")

        # 🛑 महत्वपूर्ण बदलाव: 
        # अगर आपकी 'data_preprocessing.py' फ़ाइल आउटपुट को किसी अलग पाथ पर सेव करती है (जैसे artifacts/data_preprocessing/...) 
        # तो यहाँ उसका सही पाथ दें। 
        # बैकअप के तौर पर हम अभी वही पाथ लोड कर रहे हैं लेकिन सेफ्टी चेक ऊपर लगा दिया है।
        train_dataframe = pd.read_csv(TRAIN_FILE_PATH, low_memory=False)
        test_dataframe = pd.read_csv(TEST_FILE_PATH, low_memory=False)

        fs = FeatureSelection(config=config_data)
        train_out, test_out = fs.apply_selection_and_vectorization(train_dataframe, test_dataframe)

        print("\n" + "⚙️" * 40)
        print("🎉 Feature Selection Matrix Processing Successful!")
        print(f"Vector Arrays Saved Locally to Path: {PROCESSED_DIR}")
        print(f"Saved Train Array Resource: {train_out}")
        print(f"Saved Test Array Resource: {test_out}")
        print("⚙️" * 40 + "\n")

    except Exception as main_runtime_error:
        print(f"Feature Selection validation execution broken: {main_runtime_error}")
