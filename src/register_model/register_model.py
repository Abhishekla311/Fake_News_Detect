import os
import json
import warnings
import mlflow
import mlflow.tracking
from utils.logger import get_logger
from utils.custom_exception import CustomException
import sys

from dotenv import load_dotenv
load_dotenv()  

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Abhishekla311"
repo_name = "Fake_News_Detect"

mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
logger = get_logger(__file__)

def register_model_latest(model_name: str, model_info: dict):
    try:
        client = mlflow.tracking.MlflowClient()
        run_id = model_info['run_id']
        
        # 1. 'name="model"' स्ट्रक्चर के लिए आधुनिक URI रूट पाथ
        model_uri = f"runs:/{run_id}/"
        logger.info(f"Attempting to register model from URI: {model_uri}")
        
        # LATEST FIX: 'mlflow.register_model' में 'artifact_path None' की जो एरर आ रही थी, 
        # उसे बायपास करने के लिए सीधे MlflowClient के 'create_model_version' का उपयोग किया गया है।
        try:
            client.create_registered_model(model_name)
            logger.info(f"Registered model container '{model_name}' created successfully.")
        except Exception:
            # अगर मॉडल कंटेनर पहले से मौजूद है, तो यह ब्लॉक उसे इग्नोर कर देगा
            logger.info(f"Registered model container '{model_name}' already exists.")

        # सीधे रन आईडी और सोर्स लिंक का उपयोग करके नया वर्ज़न बनाएं
        model_version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        logger.info(f"Successfully created version {model_version.version} for model '{model_name}'.")
        
        # 2. नए मॉडल वर्ज़न पर सीधे "Production" का उपनाम (Alias) सेट करें
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=str(model_version.version)
        )
        logger.info(f"🎉 Success! Model {model_name} version {model_version.version} is now active in Production.")
        
    except Exception as e:
        raise CustomException("Error during model registration and production assignment", e)
    

def main():
    try:
        config_path = 'reports/experements_info.json'
        fallback_path = 'reports/experiments_info.json'
        model_info = {}
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                model_info = json.load(file)
        elif os.path.exists(fallback_path):
            with open(fallback_path, 'r') as file:
                model_info = json.load(file)
        else:
            logger.warning("Metadata JSON file missing. Fetching latest run directly from MLflow...")
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name("Fake_News_Detect")
            exp_id = experiment.experiment_id if experiment else "0"
            
            latest_runs = client.search_runs(
                experiment_ids=[exp_id],
                max_results=1,
                order_by=["attributes.start_time DESC"]
            )
            
            if latest_runs:
                model_info = {
                    "run_id": latest_runs[0].info.run_id
                }
            else:
                raise FileNotFoundError("Neither metadata file found nor any active MLflow runs found.")
        
        # सुनिश्चित करें कि मॉडल का नाम सुसंगत (Consistent) है
        model_name = model_info.get("model_name", "FakeNewsLogisticModel")
        register_model_latest(model_name, model_info)
        
    except Exception as e:
        raise CustomException("Failed the registration process pipeline execution", e)


if __name__ == "__main__":
    main()
