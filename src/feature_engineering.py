import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class FeatureEngineer:
    def __init__(self):
        self.preprocessor = None
        
    def create_features(self, df):
        """Create new engineered features"""
        df = df.copy()
        
        # 1. Tenure groups
        df['tenure_group'] = pd.cut(df['tenure'], 
                                     bins=[0, 12, 24, 48, 72, np.inf],
                                     labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr', '6+yr'])
        
        # 2. Charge ratios
        df['charges_tenure_ratio'] = df['TotalCharges'] / (df['tenure'] + 1)
        df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        
        # 3. Service count
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        df['num_services'] = df[service_columns].apply(
            lambda x: sum([1 for val in x if val == 'Yes']), axis=1
        )
        
        # 4. Binary flags
        df['has_internet'] = df['InternetService'].apply(
            lambda x: 0 if x == 'No' else 1
        )
        
        # 5. Contract length
        contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        df['contract_months'] = df['Contract'].map(contract_mapping)
        
        return df
    
    def build_preprocessor(self, numerical_features, categorical_features):
        """Build preprocessing pipeline"""
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor
    
    def fit_transform(self, X_train, numerical_features, categorical_features):
        """Fit and transform training data"""
        self.build_preprocessor(numerical_features, categorical_features)
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        
        return X_train_processed
    
    def transform(self, X):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            self.preprocessor = joblib.load('models/preprocessor.pkl')
        
        return self.preprocessor.transform(X)