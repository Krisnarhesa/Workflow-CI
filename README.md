# RandomForest Churn Prediction with MLflow CI/CD

## Project Overview

An end to end machine learning project demonstrating **advanced model training, hyperparameter optimization, and CI/CD automation** for customer churn prediction. The project implements **manual MLflow tracking** with **GitHub Actions workflow** for automated retraining and artifact management.

### Key Features:

- **Hyperparameter Tuning**: GridSearchCV optimization (8 combinations × 5-fold CV)
- **Imbalanced Data Handling**: Class-weighted Random Forest with threshold optimization
- **Manual MLflow Logging**: Custom metrics, parameters, and 5+ artifacts
- **CI/CD Pipeline**: GitHub Actions automated training & artifact storage
- **Docker Support**: Containerized model deployment to Docker Hub
- **DagsHub Integration**: MLflow tracking with cloud synchronization

---

## 📊 Model Performance

| Metric                | Base Model | Tuned Model  |
| --------------------- | ---------- | ------------ |
| **Accuracy**          | 74.32%     | 74.89%       |
| **Precision**         | 51.01%     | 51.72%       |
| **Recall**            | 75.20%     | 76.82%       |
| **F1-Score**          | 60.78%     | **61.82%** ✓ |
| **ROC-AUC**           | 82.64%     | **83.19%** ✓ |
| **Optimal Threshold** | 0.50       | 0.35         |

**Key Achievement**: F1-score improvement through threshold optimization targeting imbalanced data classification.

---

## Technical Implementation

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Hyperparameters** (Tuned):
  - n_estimators: 250
  - max_depth: 15
  - min_samples_split: 2
  - min_samples_leaf: 1
  - class_weight: balanced
- **Optimization**: GridSearchCV (5-fold CV, f1_weighted scoring)
- **Preprocessing**: StandardScaler normalization

### Data Pipeline

- **Dataset**: 7,010 samples, 34 features
- **Class Distribution**: 73.5% neg, 26.5% pos (imbalanced)
- **Train-Test Split**: 80-20 with stratification
- **Handling Imbalance**: Class weighting + threshold optimization

#### Data Source & Preprocessing

This project uses the **Telco Customer Churn dataset** from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The preprocessing pipeline is maintained in a separate repository:

**[📊 Preprocessing Repository](https://github.com/Krisnarhesa/Eksperimen_SML_Krisnarhesa)** - Customer Churn Data Preprocessing Pipeline

The preprocessing workflow includes:
- Drops unnecessary columns (e.g., `customerID`)
- Data type conversion (`TotalCharges`)
- Handling missing values and duplicates
- Feature engineering (`tenure_group` creation)
- One-hot encoding for categorical variables
- Binary encoding for target variable (`Churn`)
- Standard scaling for numerical features
- Output: `dataset_preprocessing.csv` ready for modeling

---

## Artifacts Generated

### Training Artifacts (5 outputs per run):

1. **confusion_matrix.png** - Prediction accuracy breakdown by class
2. **roc_curve.png** - ROC curve with AUC=0.8319
3. **feature_importance.png** - Top 15 most influential features
4. **classification_report.json** - Detailed precision/recall by class
5. **training_summary.json** - Complete training metadata

---

## CI/CD Workflow

### Automated Pipeline Features:

- **Trigger Points**: Push to main/develop, PR creation, manual dispatch
- **Training Execution**: MLflow project auto-runs with conda environment
- **Artifact Management**: Auto-commits to GitHub `MLProject/artifacts/`
- **Model Deployment**: Docker image build & push to Docker Hub
- **Tracking**: MLflow experiments logged to DagsHub

### Workflow Step Sequence:

```
1. Code checkout & Python 3.12 setup
2. MLflow project execution
3. Artifact generation
4. Change detection & commit
5. Docker image build & registry push
6. GitHub Actions artifact storage (30-day retention)
```

---

## 📁 Project Structure

```
machine-learning-models/
├── MLProject/
│   ├── modelling.py              # Training script with MLflow logging
│   ├── conda.yaml                # Environment specification
│   ├── MLProject                 # MLflow project config
│   ├── dataset_preprocessing.csv # Training data
│   └── artifacts/                # Generated outputs
├── .github/workflows/
│   └── mlflow-training.yml       # CI/CD automation
├── .gitignore
└── README.md
```

---

## Integration Points

| Component            | Purpose                          | Status        |
| -------------------- | -------------------------------- | ------------- |
| **DagsHub MLflow**   | Experiment tracking & monitoring | ✅ Configured |
| **GitHub Actions**   | Automated training & deployment  | ✅ Enabled    |
| **Docker Hub**       | Model containerization           | ✅ Ready      |
| **Artifact Storage** | Persistent result management     | ✅ Automatic  |

**MLflow Dashboard**: https://dagshub.com/Krisnarhesa/machine-learning-models.mlflow/

**Docker Hub Repository**: https://hub.docker.com/r/krisnarhesa/mlflow-churn-model

---

## 💡 Key Insights

### Problem: Class Imbalance

- Standard Random Forest (accuracy 79%) failed to effectively identify minority class
- Low F1-score (0.56) despite decent accuracy due to imbalance penalty

### Solution: Multi-faceted Approach

1. **Class Weighting**: `class_weight='balanced'` penalizes majority class bias
2. **Threshold Optimization**: Lowered decision threshold (0.50 → 0.35) to improve recall
3. **Evaluation Metric**: Switched focus from accuracy to F1-score + ROC-AUC
4. **Hyperparameter Tuning**: GridSearchCV balancing across parameter space

### Result

- ✓ 4.6% F1-score improvement (0.578 → 0.618)
- ✓ 42% recall increase (52.8% → 75.2%)
- ✓ Maintained 83.19% ROC-AUC discrimination

---

## 🎓 Learning Outcomes

### Demonstrated Skills:

- ML model development & optimization
- Imbalanced data classification techniques
- MLflow experiment tracking (manual logging)
- GitHub Actions CI/CD pipeline design
- Docker containerization workflow
- Model evaluation & metrics interpretation
