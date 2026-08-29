import os
import sys
import re
import json
import hashlib
import redis
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from utils.custom_exception import CustomException
from utils.logger import get_logger
from config.paths_config import *
from utils.custom_functions import read_yaml

nltk.download('stopwords', quiet=True)
logger = get_logger(__name__)


class FeatureEngineering:
    def __init__(self, config):
        try:
            self.config_pr = config["data_preprocessing"]
            self.ps = PorterStemmer()
            self.stop_words = set(stopwords.words(self.config_pr["text_cleaning"]["language"]))
            
            # Redis Client Initialization with Exception Fallback
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=0,
                    decode_responses=True
                )
                self.redis_client.ping()
                self.use_redis = True
                logger.info("Connected successfully to Redis Cache Engine.")
            except Exception as redis_err:
                self.use_redis = False
                logger.warning(f"Redis connection failed. Falling back to CPU processing: {redis_err}")
                
            logger.info("FeatureEngineering component initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def _generate_text_hash(self, text: str) -> str:
        """Generates a unique MD5 hash fingerprint for text content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def clean_text_structure(self, text: str) -> str:
        """Cleans input text with optional Redis Caching Layer."""
        if not isinstance(text, str) or not text.strip():
            return ""

        # Redis Lookup Block
        text_hash = None
        if self.use_redis:
            text_hash = f"cleaned_text:{self._generate_text_hash(text)}"
            try:
                cached_result = self.redis_client.get(text_hash)
                if cached_result:
                    return cached_result
            except Exception:
                pass  # Ignore Redis read errors quietly

        # Core Text Cleaning Strategy
        text_clean = re.sub(r'\\x[0-9a-fA-F]{2}\??\??s?', ' ', text).lower()
        text_clean = re.sub(r'[^a-zA-Z]', ' ', text_clean)
        words = text_clean.split()
        stemmed_words = [self.ps.stem(word) for word in words if word not in self.stop_words]
        cleaned_output = ' '.join(stemmed_words)

        # Redis Cache Set Block (Cache TTL: 7 Days)
        if self.use_redis and text_hash:
            try:
                self.redis_client.setex(text_hash, 604800, cleaned_output)
            except Exception:
                pass

        return cleaned_output

    def run_feature_building(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logger.info("Feature engineering: Transforming textual structure")
            
            if self.config_pr.get("create_content_feature", True):
                train_title = train_df['title'].fillna('') if 'title' in train_df.columns else ""
                train_text = train_df['text'].fillna('') if 'text' in train_df.columns else ""
                
                test_title = test_df['title'].fillna('') if 'title' in test_df.columns else ""
                test_text = test_df['text'].fillna('') if 'text' in test_df.columns else ""

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


if __name__ == "__main__":
    try:
        config_data = read_yaml(CONFIG_PATH)

        logger.info("Loading data from raw/input paths")
        train_df = pd.read_csv(TRAIN_FILE_PATH, low_memory=False)
        test_df = pd.read_csv(TEST_FILE_PATH, low_memory=False)

        fe = FeatureEngineering(config_data)
        train_fe, test_fe = fe.run_feature_building(train_df, test_df)

        print("\n" + "=" * 45)
        print("🎉 Feature Engineering Successful with Redis Cache Check!")
        print(f"Train Shape: {train_fe.shape} | Test Shape: {test_fe.shape}")
        print("\n--- Engineered 'content' Column Preview ---")
        print(train_fe['content'].head(2).values)
        print("=" * 45 + "\n")

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        train_fe.to_csv(PROCESSED_TRAIN_DATA_PATH, index=False)
        test_fe.to_csv(PROCESSED_TEST_DATA_PATH, index=False)
        logger.info("Engineered features successfully saved to processed directory.")

    except Exception as e:
        print(f"Feature Engineering Execution Failed: {e}")