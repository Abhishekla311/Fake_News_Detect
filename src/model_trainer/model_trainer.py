# import os
# import sys
# import json
# import warnings
# import logging
# import joblib
# from sklearn.linear_model import LogisticRegression
# from config.paths_config import *
# from utils.logger import get_logger
# from utils.custom_exception import CustomException
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import mlflow
# import mlflow.sklearn

# from dotenv import load_dotenv
# load_dotenv()  

# logger = get_logger(__name__)
# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")
# logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "Abhishekla311"
# repo_name = "Fake_News_Detect"

# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


# class ModelTrainer:
#     def __init__(self):
#         logger.info("ModelTrainer component initialized.")

#     def start_training(self):
#         try:
#             logger.info("Model Training: Loading vectorized matrices...")
#             X_train, y_train = joblib.load(PROCESSED_TRAIN_DATA_PATH)
            
#             mlflow.sklearn.autolog()
            
#             with mlflow.start_run(run_name="Fake news prediction") as run:
#                 run_id = run.info.run_id
#                 logger.info(f"Active MLflow run initialized successfully. Run ID: {run_id}")

#                 logger.info("Fitting Logistic Regression Algorithm...")
#                 model = LogisticRegression()
#                 model.fit(X_train, y_train)
                
#                 y_pred = model.predict(X_train)
                            
#                 metrics = {
#                     "accuracy": accuracy_score(y_train, y_pred),
#                     "precision": precision_score(y_train, y_pred, pos_label='FAKE', zero_division=0),
#                     "recall": recall_score(y_train, y_pred, pos_label='FAKE', zero_division=0),
#                     "f1": f1_score(y_train, y_pred, pos_label='FAKE', zero_division=0)
#                 }
                            
#                 for k, v in metrics.items():
#                     logger.info(f"{k.capitalize()} Score: {v:.4f}")

#                 os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
#                 joblib.dump(model, MODEL_OUTPUT_PATH)
#                 logger.info(f"Trained model saved successfully at: {MODEL_OUTPUT_PATH}")
                
#                 logger.info("Uploading and Registering model to DagsHub Registry...")
                
#                 # LATEST FIX: 'registered_model_name' का उपयोग करके मॉडल यहीं रजिस्टर हो जाएगा
#                 # 'artifact_path' को हटाकर डायरेक्टरी एरर को जड़ से खत्म किया गया
#                 mlflow.sklearn.log_model(
#                     sk_model=model, 
#                     name="model",
#                     registered_model_name="FakeNewsLogisticModel",
#                     serialization_format="skops"
#                 )
#                 logger.info("Model logged and registered successfully!")
                
#                 # metadata JSON फ़ाइल जनरेट करें
#                 report_dir = "reports"
#                 os.makedirs(report_dir, exist_ok=True)
#                 experiment_info_path = os.path.join(report_dir, "experements_info.json")
                
#                 experiment_metadata = {
#                     "run_id": run_id,
#                     "model_name": "FakeNewsLogisticModel"
#                 }
                
#                 with open(experiment_info_path, "w") as f:
#                     json.dump(experiment_metadata, f, indent=4)
#                 logger.info(f"Saved experiment run details to: {experiment_info_path}")
            
#             return MODEL_OUTPUT_PATH
            
#         except Exception as e:
#             logger.error("Error during model training sequence")
#             raise CustomException(e, sys)


# if __name__ == "__main__":
#     try:
#         trainer = ModelTrainer()
#         saved_model_path = trainer.start_training()
#         print("\n🎉 Machine Learning Model Training and Registration Successful!")
#     except Exception as main_runtime_error:
#         print(f"Model Training validation execution failed: {main_runtime_error}")






import os
import sys
import json
import warnings
import logging
import joblib
from sklearn.linear_model import LogisticRegression
from config.paths_config import *
from utils.logger import get_logger
from utils.custom_exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import redis  # Redis लाइब्रेरी इम्पोर्ट की

from dotenv import load_dotenv
load_dotenv()  

logger = get_logger(__name__)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Abhishekla311"
repo_name = "Fake_News_Detect"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


class ModelTrainer:
    def __init__(self):
        logger.info("ModelTrainer component initialized.")
        # Redis कनेक्शन सेटअप (Environment variables से क्रेडेंशियल उठाएगा)
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=True
        )
        self.redis_key = "fake_news_training:status"

    def update_redis_status(self, status, **kwargs):
        """Redis में ट्रेनिंग स्टेटस और लाइव प्रोग्रेस अपडेट करने के लिए हेल्पर फ़ंक्शन"""
        try:
            data = {"status": status, **kwargs}
            self.redis_client.set(self.redis_key, json.dumps(data))
            # स्टेटस को 1 दिन (86400 सेकेंड) में एक्सपायर होने के लिए सेट करें ताकि मेमोरी क्लीन रहे
            self.redis_client.expire(self.redis_key, 86400) 
        except Exception as redis_err:
            logger.warning(f"Redis Update Failed: {redis_err}")

    def start_training(self):
        try:
            # 1. स्टेटस अपडेट: ट्रेनिंग शुरू हो गई है
            self.update_redis_status("STARTED", message="Loading vectorized matrices...")
            logger.info("Model Training: Loading vectorized matrices...")
            X_train, y_train = joblib.load(PROCESSED_TRAIN_DATA_PATH)
            
            mlflow.sklearn.autolog()
            
            with mlflow.start_run(run_name="Fake news prediction") as run:
                run_id = run.info.run_id
                logger.info(f"Active MLflow run initialized successfully. Run ID: {run_id}")
                
                # 2. स्टेटस अपडेट: मॉडल फिट होना शुरू हो गया है
                self.update_redis_status("TRAINING", run_id=run_id, message="Fitting Logistic Regression...")

                logger.info("Fitting Logistic Regression Algorithm...")
                model = LogisticRegression()
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_train)
                            
                metrics = {
                    "accuracy": float(accuracy_score(y_train, y_pred)),
                    "precision": float(precision_score(y_train, y_pred, pos_label='FAKE', zero_division=0)),
                    "recall": float(recall_score(y_train, y_pred, pos_label='FAKE', zero_division=0)),
                    "f1": float(f1_score(y_train, y_pred, pos_label='FAKE', zero_division=0))
                }
                            
                for k, v in metrics.items():
                    logger.info(f"{k.capitalize()} Score: {v:.4f}")

                os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
                joblib.dump(model, MODEL_OUTPUT_PATH)
                logger.info(f"Trained model saved successfully at: {MODEL_OUTPUT_PATH}")
                
                # 3. स्टेटस अपडेट: मॉडल लॉग और रजिस्टर हो रहा है
                self.update_redis_status("REGISTERING", run_id=run_id, metrics=metrics, message="Uploading model to DagsHub Registry...")
                logger.info("Uploading and Registering model to DagsHub Registry...")
                
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    name="model",
                    registered_model_name="FakeNewsLogisticModel",
                    serialization_format="skops"
                )
                logger.info("Model logged and registered successfully!")
                
                # metadata JSON फ़ाइल जनरेट करें
                report_dir = "reports"
                os.makedirs(report_dir, exist_ok=True)
                experiment_info_path = os.path.join(report_dir, "experements_info.json")
                
                experiment_metadata = {
                    "run_id": run_id,
                    "model_name": "FakeNewsLogisticModel"
                }
                
                with open(experiment_info_path, "w") as f:
                    json.dump(experiment_metadata, f, indent=4)
                logger.info(f"Saved experiment run details to: {experiment_info_path}")
                
                # 4. स्टेटस अपडेट: ट्रेनिंग सफलतापूर्वक पूरी हो गई है
                self.update_redis_status("COMPLETED", run_id=run_id, metrics=metrics, model_path=MODEL_OUTPUT_PATH)
            
            return MODEL_OUTPUT_PATH
            
        except Exception as e:
            # 5. स्टेटस अपडेट: कोई एरर आने पर Redis में फेलियर लॉग करें
            self.update_redis_status("FAILED", error=str(e))
            logger.error("Error during model training sequence")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        saved_model_path = trainer.start_training()
        print("\n🎉 Machine Learning Model Training and Registration Successful!")
    except Exception as main_runtime_error:
        print(f"Model Training validation execution failed: {main_runtime_error}")
