import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_processed_data(self):
        """Load processed data"""
        X_train = pd.read_csv(f"{self.config['data']['processed_path']}X_train.csv")
        X_test = pd.read_csv(f"{self.config['data']['processed_path']}X_test.csv")
        y_train = pd.read_csv(f"{self.config['data']['processed_path']}y_train.csv")
        y_test = pd.read_csv(f"{self.config['data']['processed_path']}y_test.csv")
        
        return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()
    
    def get_models(self):
        """Define multiple models to compare"""
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.config['model']['random_state'],
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.config['model']['random_state'],
                class_weight='balanced',
                n_estimators=100
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.config['model']['random_state'],
                n_estimators=100
            ),
            'XGBoost': XGBClassifier(
                random_state=self.config['model']['random_state'],
                eval_metric='logloss',
                use_label_encoder=False
            ),
            'LightGBM': LGBMClassifier(
                random_state=self.config['model']['random_state'],
                class_weight='balanced'
            )
        }
        return models
    
    def train_with_cv(self, X_train, y_train):
        """Train models with cross-validation"""
        models = self.get_models()
        cv_results = {}
        
        for name, model in models.items():
            # Creating pipeline with SMOTE for imbalanced data
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=self.config['model']['random_state'])),
                ('classifier', model)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=self.config['model']['cv_folds'],
                scoring=self.config['model']['scoring']
            )
            
            cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_results
    
    def train_best_model(self, X_train, y_train):
        """Train and optimize the best model"""
        print("\nTraining Random Forest with Hyperparameter Tuning...")
        
        # Defining pipeline with SMOTE
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=self.config['model']['random_state'])),
            ('rf', RandomForestClassifier(random_state=self.config['model']['random_state'],
                                         class_weight='balanced'))
        ])
        
        # Hyperparameter grid
        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [None, 10, 20, 30],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=self.config['model']['cv_folds'],
            scoring=self.config['model']['scoring'],
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Save best model
        joblib.dump(grid_search.best_estimator_, 
                   self.config['deployment']['model_path'])
        
        return grid_search.best_estimator_