import os
import pandas as pd
from sklearn.model_selection import train_test_split
from config.paths_config import *
from utils.logger import get_logger
import boto3
import sys
from utils.custom_functions import read_yaml
from utils.custom_exception import CustomException


logger = get_logger(__name__)
class Data_ingestion:
    def __init__(self, config):
        try:
            self.config = config["data_ingestion"]
            self.bucket_name = self.config["bucket_name"]
            self.file_name = self.config["bucket_file_name"]
            self.train_tratio = self.config["train_ratio"]
            os.makedirs(RAW_DIR, exist_ok=True)
            logger.info(f"DataIngestion Starting") 
        except Exception as e:
            logger.error("Error while DataIngestion ")
            raise CustomException( e, sys)

    def download_data_from_s3(self):
        try:
            s3_client = boto3.client('s3')
            s3_client.download_file(
                Bucket=self.bucket_name,
                Key=self.file_name,
                Filename=RAW_FILE_PATH
            )
            logger.info("Successfully downloaded csv form S3")
        except Exception as e:
            logger.error("Erro While downloading  the csv file")
            raise CustomException(e, sys)

    def split_data(self):
        try:
            logger.info("Starting the splitting process")
            data = pd.read_csv(RAW_FILE_PATH,encoding='latin-1')
            train_data, test_data = train_test_split(
                data, test_size=1-self.train_tratio, random_state=42
            )

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Test data saved to {TEST_FILE_PATH}")
        except Exception as e:
            logger.error("Error while splitting data")
            raise CustomException(e, sys)
        
    def run(self):
        try:
            logger.info("Starting data ingestion stage")
            self.download_data_from_s3()
            self.split_data()
            logger.info("Data ingestion completed successfully")
        except Exception as e:
            logger.error(f"CustomException in Ingestion  {str(e)}")
            raise CustomException(e, sys)

if __name__ =="__main__":
    data_ingestion = Data_ingestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()


