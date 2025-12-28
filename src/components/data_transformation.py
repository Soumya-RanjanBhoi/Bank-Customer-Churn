import pandas as pd
import numpy as np
import os,sys,pickle
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomeException
from src.logger import logging
from sklearn.pipeline import Pipeline
from src.components.data_ingestion import DataIngestion

class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            transformer= ColumnTransformer(
                        transformers=[
                            ("crscore_trans",StandardScaler(),['CreditScore']),
                            ("age_trans",StandardScaler(),['Age']),
                            ("tenure_trans",StandardScaler(),['Tenure']),
                            ("balance_trans",StandardScaler(),['Balance']),
                            ("estimated_salary_trans",StandardScaler(),['EstimatedSalary']),
                            ("geo_trans",OrdinalEncoder(),['Geography']),
                            ("gender_trans",OrdinalEncoder(),['Gender']),
                            ("credit_score_trans",OrdinalEncoder(),['CreditScoreCategory']),
                            ('haszerobal_trans',OrdinalEncoder(),['hasZeroBalance'])
                        ],
                        remainder="passthrough"
                    )
            return transformer
            
        except Exception as e:
            raise CustomeException(e,sys)
        
    def feature_eng(self,df:pd.DataFrame)-> pd.DataFrame:
        try:
            avg_credit=df['CreditScore'].mean()
            df['CreditScoreCategory'] = np.where(df['CreditScore'] > avg_credit, 'Above Average', 'Below Average')
            df['hasZeroBalance']= np.where(df['Balance'] ==0.0,"Yes",'No')
            df.drop(['RowNumber','CustomerId','Surname'],axis=1,inplace=True)

            logging.info('feature engneering completed:')

            return df
        
        except Exception as e:
            raise CustomeException(e,sys)


        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read test and train data')

            train=self.feature_eng(train_df)

            logging.info('Completed feature eng of train data')

            test=self.feature_eng(test_df)

            logging.info('Completed feature eng of test data')

            input_train_features= train.drop("Exited",axis=1)
            output_train_feature= train['Exited']

            input_test_features= test.drop("Exited",axis=1)
            output_test_feature= test['Exited']

            transformer = self.get_data_transformer_object()
            logging.info('Found Column Transformer')

            input_train_features=transformer.fit_transform(input_train_features)
            input_test_features= transformer.transform(input_test_features) 
            logging.info('Preprocessing Completed')

            train_arr = np.c_[input_train_features,np.array(output_train_feature)]          

            test_arr =np.c_[input_test_features,np.array(output_test_feature)]

            with open(self.data_transformation_config.preprocessor_obj_file_path,"wb") as file:
                pickle.dump(transformer,file)
            
            logging.info('Saved transformer')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomeException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    train,test=obj.initiate_data_ingestion()

    obj1=DataTransformation()
    obj1.initiate_data_transformation(train,test)

