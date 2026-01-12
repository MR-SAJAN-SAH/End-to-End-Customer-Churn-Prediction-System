import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report, roc_curve, precision_recall_curve)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

class ModelEvaluator:
    def __init__(self, model_path="models/churn_model.pkl"):
        self.model = joblib.load(model_path)
        
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculating metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return metrics, report, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_test, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return fig
    
    def plot_roc_curve(self, y_test, y_pred_proba, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                mode='lines',
                                name=f'ROC curve (AUC = {roc_auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                mode='lines',
                                name='Random',
                                line=dict(dash='dash')))
        
        fig.update_layout(title='ROC Curve',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate',
                         width=600, height=500)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision,
                                mode='lines',
                                name='Precision-Recall curve'))
        
        fig.update_layout(title='Precision-Recall Curve',
                         xaxis_title='Recall',
                         yaxis_title='Precision',
                         width=600, height=500)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_dashboard(self, metrics, report, y_test, y_pred_proba, save_path='results/dashboard.html'):
        """Create interactive evaluation dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve',
                          'Precision-Recall Curve', 'Metrics Summary'),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                  [{'type': 'scatter'}, {'type': 'table'}]]
        )
        
        # Add confusion matrix
        cm = confusion_matrix(y_test, self.model.predict(X_test))
        fig.add_trace(
            go.Heatmap(z=cm, colorscale='Blues', showscale=False),
            row=1, col=1
        )
        
        # Add ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'),
            row=1, col=2
        )
        
        # Add PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        fig.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', name='PR'),
            row=2, col=1
        )
        
        # Add metrics table
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [f"{v:.4f}" for v in metrics.values()]
        })
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(metrics_df.columns),
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[metrics_df['Metric'], metrics_df['Value']],
                          fill_color='lavender',
                          align='left')
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, width=1000,
                         title_text="Model Evaluation Dashboard")
        
        fig.write_html(save_path)
        return fig