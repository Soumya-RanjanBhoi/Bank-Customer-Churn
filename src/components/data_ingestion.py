import os,sys
import requests
from src.exception import CustomeException
from src.logger import logging
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str =os.path.join('artifacts','train.csv')
    test_data_path :str = os.path.join('artifacts','test.csv')
    raw_data_path :str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        self.path=""
    
    def download_data(self):
        try:
            path = kagglehub.dataset_download("saurabhbadole/bank-customer-churn-prediction-dataset")
            self.path=f"{path}/Churn_Modelling.csv"
            return f"file saved in {path}/Churn_Modelling.csv"
        except  Exception as e:
            raise CustomeException(e,sys)
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:

            logging.info('Downloading data')
            path = self.download_data()
            logging.info('Completed the downloading part')

            df=pd.read_csv(self.path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('train test split initiated')

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomeException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()




