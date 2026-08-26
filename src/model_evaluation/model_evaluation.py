import sys
import joblib
from sklearn.metrics import accuracy_score
from config.paths_config import *
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class ModelEvaluation:
    def __init__(self):
        logger.info("ModelEvaluation component initialized.")

    def execute_evaluation(self):
        try:
            logger.info("Model Evaluation started...")
            
            # Loading the test matrices and trained model using paths_config paths
            X_test, y_test = joblib.load(PROCESSED_TEST_DATA_PATH)
            model = joblib.load(MODEL_OUTPUT_PATH)

            logger.info("Making predictions on test dataset matrices...")
            predictions = model.predict(X_test)
            
            # Calculating accuracy score
            accuracy = accuracy_score(y_test, predictions)
            
            print("\n" + "★" * 40)
            print(f"🎯 PRODUCTION MODEL TEST ACCURACY: {accuracy * 100:.2f}%")
            print("★" * 40 + "\n")
            
            logger.info(f"Model Evaluation completed successfully. Score: {accuracy}")
            return accuracy
            
        except Exception as e:
            logger.error("Error during model evaluation stage")
            raise CustomException(e, sys)

# Standalone execution runtime wrapper block
if __name__ == "__main__":
    try:
        # Initialize evaluation component
        evaluation = ModelEvaluation()
        
        # Trigger evaluation workflow
        final_accuracy = evaluation.execute_evaluation()
        
    except Exception as main_runtime_error:
        print(f"Model Evaluation validation execution failed: {main_runtime_error}")
