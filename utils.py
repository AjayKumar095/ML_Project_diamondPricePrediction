import os
import sys
import pickle
import pandas as pd
import numpy as np
from logger import logging
from exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logging.info("Pickle file save")
        
    except Exception as e:
        logging.info("Failed to save pickle file")        


def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        logging.info("Model evaluation start")
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            
            # Train model
            model.fit(X_train, y_train)
            
            
            # Predicting test data
            y_test_pred=model.predict(X_test)
            
            # Get r2 scores for train and text data
            test_model_score=r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]]= test_model_score
            
        return report
    except Exception as e:
        logging.info("Model evaluation failed")
        raise CustomException(e, sys)    


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)            