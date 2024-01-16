from sklearn.linear_model import LinearRegression , Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from logger import logging
from exception import CustomException
from utils import save_object
import pandas as pd 
import numpy as np
import os,sys


## Data Transformation config
@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')


## Data Transformation class
class DataTransformation:
    
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationconfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation Starts")
            # define which column should be ordinal-encoded and which should be scaled.
            categorical_col=["cut",'color', 'clarity']
            numerical_col=['carat', 'depth', 'table', 'x','y', 'z']

            # define the custome ranking for each ordinal variable
            cut_cat=["Fair","Good", "Very Good", "Premium", "Ideal"]
            color_cat=["D", "E", "F", "G", "H", "I", "J"]
            clarity_cat=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
            
            logging.info("Pipeline Start")
            ## numerical pipeline

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            ## Categorical Pipeline

            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OrdinalEncoder', OrdinalEncoder(categories=[cut_cat, color_cat, clarity_cat])),
                    ('scalar', StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_col),
                ("cat_pipeline", cat_pipeline, categorical_col)
            ])
            
            logging.info("Pipeline Start")
            
            return preprocessor
        
        except Exception as e:
                logging.info("Data Transformation Failed", e)
    
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Initiated Start")
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f'Train Dataframe Head:  \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head:  \n{test_df.head().to_string()}')
            
            logging.info("Obtaining preprocessor object")
            
            preprocessor_obj=self.get_data_transformation_object()
            
            target_column_name='price'
            drop_columns=[target_column_name, 'id']
            
            ## sorting features into independent and dependent features
            
            # for train data
            input_feature_train_df=train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            # for test data
            input_feature_test_df=test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            # applying transformation
            
            intput_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            intput_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            logging.info("Applying preprocessing object on training and test datasets")
            
            train_arr=np.c_[intput_feature_train_arr, np.array(target_feature_train_df)]
            test_arr=np.c_[intput_feature_test_arr, np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Preprocessor pickle is created and saved")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info("Data Initiated Failed", e)            
            
            raise  CustomException(e, sys)