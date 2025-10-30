import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Adjustment:
    def __init__(self, csv_file):
        """
        Initializes the BikePathRegressor instance by loading data from a CSV file.
        
        Parameters:
        csv_file (str): Path to the CSV file containing the data.
        """
        self.df = pd.read_csv(csv_file)  # Load data from CSV file
        self.models = {}  # Initialize dictionary to store models for each bike path type

    def preprocess_data(self):
        """
        Preprocesses the data by creating dummy variables and separating features and target variable.
        """
        # Create dummy variables for 'BikePathType' column
        dummies = pd.get_dummies(self.df['BikePathType'])
        
        # Dynamically select only existing predictors
        predictors = [col for col in ['maxspeed_score', 'lane_score_normalized', 'parking_score', 'crossing_score', 'width_score'] if col in self.df.columns]
        self.X = pd.concat([self.df[predictors], dummies], axis=1)
        
        # Extract target variable 'Score'
        self.Y = self.df['Score']

    def fit(self):
        """
        Fits a linear regression model for each bike path type.
        """
        self.preprocess_data()  # Preprocess the data
        
        # Iterate over unique bike path types
        for bike_path_type in self.df['BikePathType'].unique():
            model = LinearRegression()  # Create a linear regression model
            X = self.X[self.df['BikePathType'] == bike_path_type]  # Select features for the current bike path type
            y = self.Y[self.df['BikePathType'] == bike_path_type]  # Select target variable for the current bike path type
            model.fit(X, y)  # Fit the model
            self.models[bike_path_type] = model  # Store the model in the dictionary

    def predict(self, bike_path_type, X):
        """
        Predicts scores for the given bike path type and feature matrix.
        
        predictors = [col for col in ['maxspeed_score', 'lane_score_normalized', 'parking_score', 'crossing_score', 'width_score'] if col in self.df.columns]
        self.X = pd.concat([self.df[predictors], dummies], axis=1)
        bike_path_type (str): Type of bike path for which to make predictions.
        X (array-like or DataFrame): Feature matrix for prediction.
        
        Returns:
        array-like: Predicted scores.
        """
        model = self.models.get(bike_path_type)  # Get the model for the specified bike path type
        if model is None:
            raise ValueError(f"No model found for BikePathType: {bike_path_type}")
        
        # Infer correct column count
        input_cols = X.shape[1]
        if input_cols == 5:
            columns = ['maxspeed_score', 'lane_score_normalized', 'parking_score', 'crossing_score', 'width_score']
        elif input_cols == 4:
            columns = ['maxspeed_score', 'lane_score_normalized', 'parking_score', 'crossing_score']
        else:
            raise ValueError(f"Unexpected number of columns in input X: {input_cols}")

        X = pd.DataFrame(X, columns=columns)
        
        # Add columns for the bike path type
        for bpt in ['cycle_lane', 'safety_lane', 'separated_lane']:
            X[bpt] = 1 if bpt == bike_path_type else 0
        
        # Make predictions using the model
        return np.clip(model.predict(X), 0, 1)
    
    def check_multicollinearity(self):
        """
        Check for multicollinearity among the predictor variables.
        
        Returns:
        pandas.DataFrame: The correlation matrix of the predictor variables.
        """
        self.preprocess_data()  # Preprocess the data
        
        # Get the predictor variables
        X = self.X[['maxspeed_score', 'lane_score_normalized', 'parking_score', 'crossing_score', 'width_score']]
        
        # Calculate the correlation matrix
        corr_matrix = X.corr(method='pearson')

        #print(corr_matrix)
        print(f"Correlation matrix:\n{corr_matrix}\n")
        
        return corr_matrix

    def print_coefficients(self):
        """
        Prints the coefficients of the trained linear regression models along with the corresponding feature names.
        """
        for bike_path_type, model in self.models.items():
            print(f"Coefficients for {bike_path_type}:")
            
            # Get the feature names from the column names of the feature matrix
            feature_names = list(self.X.columns)
            
            # Create a mapping between feature names and coefficients
            coef_map = dict(zip(feature_names, model.coef_))
            
            # Print the coefficient for each feature
            for feature, coef in coef_map.items():
                print(f"{feature}: {coef}")
            
            print("\n")

    def evaluate(self, bike_path_type):
        """
        Evaluates the performance of the linear regression model for a given bike path type.

        Parameters:
        bike_path_type (str): Type of bike path for which to evaluate the model.

        Returns:
        dict: A dictionary containing the mean squared error (MSE) and R-squared (R2) score for the specified bike path type.
        """
        model = self.models.get(bike_path_type)
        if model is None:
            raise ValueError(f"No model found for BikePathType: {bike_path_type}")

        X = self.X[self.df['BikePathType'] == bike_path_type]
        y = self.Y[self.df['BikePathType'] == bike_path_type]

        y_pred = model.predict(X)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return {'MSE': mse, 'R2': r2}
