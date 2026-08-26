import os
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from config.paths_config import *
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self):
        logger.info("ModelTrainer component initialized.")

    def start_training(self):
        try:
            logger.info("Model Training: Loading vectorized matrices...")
            # Automatically reads the path from config/paths_config.py
            X_train, y_train = joblib.load(PROCESSED_TRAIN_DATA_PATH)

            logger.info("Fitting Logistic Regression Algorithm...")
            model = LogisticRegression()
            model.fit(X_train, y_train)

            # Creating the directory path for the model artifact if it doesn't exist
            os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
            
            # Saving the trained model file securely to disk
            joblib.dump(model, MODEL_OUTPUT_PATH)
            logger.info(f"Trained model saved successfully at: {MODEL_OUTPUT_PATH}")
            
            return MODEL_OUTPUT_PATH
            
        except Exception as e:
            logger.error("Error during model training sequence")
            raise CustomException(e, sys)

# Standalone execution runtime wrapper block
if __name__ == "__main__":
    try:
        # Initialize the trainer component
        trainer = ModelTrainer()
        
        # Trigger the model training operation
        saved_model_path = trainer.start_training()
        
        print("\n" + "🎓" * 40)
        print("🎉 Machine Learning Model Training Successful!")
        print(f"Algorithm Trained: Logistic Regression")
        print(f"Model File Saved Permanently at: {saved_model_path}")
        print("🎓" * 40 + "\n")
        
    except Exception as main_runtime_error:
        print(f"Model Training validation execution failed: {main_runtime_error}")
