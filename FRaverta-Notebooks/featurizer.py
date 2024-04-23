import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, TransformerMixin
import requests
from io import BytesIO
import zipfile
import structlog

logger = structlog.get_logger()


DATA_SET_COLUMNS = [
    'stock_id', 
    'date_id', 
    'seconds_in_bucket', 
    'imbalance_size', 
    'imbalance_buy_sell_flag', 
    'reference_price', 
    'matched_size', 
    'far_price', 
    'near_price', 
    'bid_price', 
    'bid_size', 
    'ask_price', 
    'ask_size', 
    'wap', 
    'target', 
    'time_id', 
    'row_id'
]

NUMERICAL_FEATURES = ['seconds_in_bucket', 'imbalance_size', 'reference_price', 
                      'matched_size', 'far_price', 'near_price', 'bid_price', 
                      'bid_size', 'ask_price', 'ask_size', 'wap']
CATEGORICAL_FEATURES = ['stock_id', 'imbalance_buy_sell_flag']

OTHER_FEATURES = ['row_id', 'time_id', 'date_id']

y = ['target']

# Rolling window fetaures

categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])



# Custom transformer to add a new feature (example: combining existing features)

class FeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # call super constructor
        super(FeatureAdder, self).__init__()

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X


class TradingAtTheCloseDS:
    # Link to the ZIP file
    zip_url = "https://media.githubusercontent.com/media/fdraverta/comp642-project/main/trading_the_close_data.zip?download=true&token=A362CLYGAPLZIW6YLGSGPGDGE23EQ"

    def __init__(self, path_to_zip_with_data=None):
        self.data_set_columns = DATA_SET_COLUMNS
        self.numerical_features = NUMERICAL_FEATURES
        self.categorical_features = CATEGORICAL_FEATURES
        self.other_features = OTHER_FEATURES
        self.y_column = y

        if path_to_zip_with_data is None:
            logger.info("Downloading ZIP file with the data set...")
            response = requests.get(self.zip_url, allow_redirects=True)
            zip_data = BytesIO(response.content)
        else:
            logger.info("Using ZIP file with the data set from the provided path", path=path_to_zip_with_data)
            zip_data = path_to_zip_with_data


        # Open the ZIP file
        logger.info("Opening ZIP file...")
        with zipfile.ZipFile(zip_data, 'r') as z:
            fname = "train.csv"
            logger.info("Reading CSV file...", fname=fname)
            with z.open(fname) as file:
                self.data = pd.read_csv(file)
        logger.info("Data set loaded successfully.")
    
    def get_train_test_data(self):
        train_data = self.data.loc[self.data['date_id'] < 425]
        test_data = self.data.loc[self.data['date_id'] >= 425]
        
        return train_data, test_data


    def compute_baseline_model(self, simple_mapping=None):
        train_data, test_data = self.get_train_test_data()

        # Create explicit copies to avoid "SettingWithCopyWarning"
        train_data = train_data.copy()
        test_data = test_data.copy()

        if simple_mapping is None:
            simple_mapping = {
                1: 0.1,
                0: 0,
                -1: -0.1
            }

        train_data.loc[:, 'simple_prediction'] = train_data['imbalance_buy_sell_flag'].map(simple_mapping)
        test_data.loc[:, 'simple_prediction'] = test_data['imbalance_buy_sell_flag'].map(simple_mapping)

        train_mae = (train_data['simple_prediction'] - train_data['target']).abs().mean()
        test_mae = (test_data['simple_prediction'] - test_data['target']).abs().mean()

        logger.info("Baseline model computed.", train_mae=train_mae, test_mae=test_mae)

        return train_mae, test_mae


if __name__ == '__main__':
    data = TradingAtTheCloseDS("/home/fraverta/development/comp642-project/FRaverta-Notebooks/comp642-project/trading_the_close_data.zip")
    train_data, test_data = data.get_train_test_data()
    train_mae, test_mae = data.compute_baseline_model()



"""usage example"""
'''
# Example data
data = {
    'total_rooms': [10, 20, 30, 40],
    'households': [1, 2, 3, 4],
    'ocean_proximity': ['NEAR BAY', 'INLAND', 'NEAR OCEAN', 'INLAND']
}

df = pd.DataFrame(data)

# Define numerical and categorical features
numerical_features = ['total_rooms', 'households']
categorical_features = ['ocean_proximity']

# Numerical pipeline with feature addition
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('feature_adder', FeatureAdder()),              # Custom feature addition
    ('scaler', StandardScaler())                    # Scaling
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('onehot', OneHotEncoder())                             # One-hot encoding
])

# Combine pipelines
preprocessing_pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Applying the pipeline
transformed_data = preprocessing_pipeline.fit_transform(df)

# Output transformed data
print(transformed_data)
'''

