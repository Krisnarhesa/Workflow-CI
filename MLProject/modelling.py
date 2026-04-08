import json
import os
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dagshub.init(repo_owner='Krisnarhesa',
             repo_name='machine-learning-models',
             mlflow=True)


def get_or_create_experiment(name: str):
    """Set experiment, restoring it first if it was previously deleted."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        print(f"Restored deleted experiment: '{name}'")
    mlflow.set_experiment(name)


get_or_create_experiment("Base_Model_CICD")


def load_and_prepare_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load and split the preprocessed dataset."""
    df = pd.read_csv(data_path)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def create_confusion_matrix_plot(y_true, y_pred, output_path: str = "confusion_matrix.png"):
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path


def create_roc_curve_plot(y_true, y_pred_proba, output_path: str = "roc_curve.png"):
    """Create and save ROC curve visualization."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    return output_path


def create_feature_importance_plot(model, feature_names, output_path: str = "feature_importance.png"):
    """Create and save feature importance visualization."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        return output_path
    return None


def train_base_model():
    """Train the base machine learning model and log with MLflow."""
    
    data_path = Path(__file__).parent / "dataset_preprocessing.csv"
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data(
        str(data_path), test_size=0.2
    )
    
    os.environ.pop("MLFLOW_RUN_ID", None)

    with mlflow.start_run(run_name="CICD_Base_Model") as run:
        
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("feature_count", len(feature_names))
        mlflow.log_param("scaler_type", "StandardScaler")
        mlflow.log_param("stratified_split", True)
        
        model_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, str(param_value))
        
        unique, counts = np.unique(y_train, return_counts=True)
        for class_val, count in zip(unique, counts):
            mlflow.log_param(f"train_class_{int(class_val)}_count", count)
        
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        best_threshold = 0.5
        best_f1 = 0.0
        threshold_results = []
        
        for threshold in np.arange(0.2, 0.8, 0.05):
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1_thresh = f1_score(y_test, y_pred_thresh)
            threshold_results.append((threshold, f1_thresh))
            if f1_thresh > best_f1:
                best_f1 = f1_thresh
                best_threshold = threshold
        
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        mlflow.log_param("optimal_threshold", round(best_threshold, 2))
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc_score", roc_auc)
        
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        mlflow.sklearn.log_model(model, "model")
        
        artifacts_dir = "artifacts"
        os.makedirs(artifacts_dir, exist_ok=True)
        
        cm_path = create_confusion_matrix_plot(y_test, y_pred, f"{artifacts_dir}/confusion_matrix.png")
        mlflow.log_artifact(cm_path)
        
        roc_path = create_roc_curve_plot(y_test, y_pred_proba, f"{artifacts_dir}/roc_curve.png")
        mlflow.log_artifact(roc_path)
        
        fi_path = create_feature_importance_plot(model, feature_names, f"{artifacts_dir}/feature_importance.png")
        if fi_path:
            mlflow.log_artifact(fi_path)
        
        class_report = classification_report(y_test, y_pred, output_dict=True)
        report_path = f"{artifacts_dir}/classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(class_report, f, indent=4)
        mlflow.log_artifact(report_path)
        
        run_id = run.info.run_id
        run_id_path = Path(__file__).parent / "run_id.txt"
        with open(run_id_path, "w") as f:
            f.write(run_id)
        print(f"run_id saved to: {run_id_path}")
        
        print("\n" + "="*60)
        print("BASE MODEL TRAINING COMPLETED")
        print("="*60)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print("="*60)
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {mlflow.get_experiment(run.info.experiment_id).name}")
        print("="*60)


if __name__ == "__main__":
    train_base_model()
