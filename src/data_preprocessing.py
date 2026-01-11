import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import yaml

class DataPreprocessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self):
        """Load and initial preprocessing"""
        df = pd.read_csv(self.config['data']['raw_path'])
        
        # Clean TotalCharges (convert to numeric, handle missing)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Drop customerID (identifier)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values"""
        # Fill TotalCharges missing with 0 (new customers)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        
        # Fill other missing categorical values with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def encode_target(self, df, target_col='Churn'):
        """Encode target variable"""
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        joblib.dump(le, 'models/label_encoder.pkl')
        return df
    
    def split_data(self, df, target_col='Churn'):
        """Split data into train and test sets"""
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        import os
        os.makedirs(self.config['data']['processed_path'], exist_ok=True)
        
        X_train.to_csv(f"{self.config['data']['processed_path']}X_train.csv", index=False)
        X_test.to_csv(f"{self.config['data']['processed_path']}X_test.csv", index=False)
        y_train.to_csv(f"{self.config['data']['processed_path']}y_train.csv", index=False)
        y_test.to_csv(f"{self.config['data']['processed_path']}y_test.csv", index=False)