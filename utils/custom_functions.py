import os
import pandas 
from utils.logger import get_logger
from utils.custom_exception import CustomException
import yaml
import pandas as pd

logger = get_logger(__name__)
def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Successfully read the yaml File")
            return config
    except Exception as e:
        logger.error("Error while loading yaml file")
        raise CustomException("Failed to read the yaml file", e)

def load_data(path):
    try:
        logger.info("Loading")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading the data {e}")
        raise CustomException("Failed to load data", e)