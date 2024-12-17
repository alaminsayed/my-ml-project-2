import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\proprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(  self,
        area_type,
        availability, 
        location, 
        size, 
        society,
        total_sqft, 
        bath, 
        balcony ):
        self.area_type = area_type
        self.availability = availability
        self.location = location
        self.size = size
        self.society = society
        self.total_sqft = total_sqft
        self.bath = bath
        self.balcony=balcony
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "area_type": [self.area_type],
                "availability": [self.availability],
                "location": [self.location],
                "size": [self.size],
                "society": [self.society],
                "total_sqft": [self.total_sqft],
                "balcony": [self.balcony],
                "balcony": [self.balcony],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
                 