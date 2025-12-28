from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler , OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.compose import ColumnTransformer
from xgboost import  XGBClassifier
from catboost import CatBoostClassifier
import pickle,sys


class ModelConfig:
    def __init__(self):
        self.Model_path = "Model/final_model.pkl"
        self.transformer = 'Model/preprocessor.pkl'


class PredictionPipeline():
    def __init__(self):
        self.config = ModelConfig()

        try:
            with open(self.config.Model_path,"rb") as file:
                self.model = pickle.load(file)

            with open(self.config.transformer ,"rb") as file:
                self.transformer=pickle.load(file)

            logging.info("Model and Transformater loaded")
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self, user_data):
        try:
            user_data_transformed = self.transformer.transform(user_data)
            result = self.model.predict(user_data_transformed)
            return result

        except Exception as e:
            raise CustomException(e, sys)


