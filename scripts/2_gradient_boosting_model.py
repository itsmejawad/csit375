"""
Gradient Boosting Classifier for Malware Detection
==================================================

This script trains a Gradient Boosting model (XGBoost) to classify malware vs benign samples.

XGBoost (Extreme Gradient Boosting) is a powerful ensemble method that:
- Builds trees sequentially, each correcting errors of previous trees
- Uses gradient descent to minimize loss function
- Incorporates regularization to prevent overfitting
- Often achieves state-of-the-art performance
- Handles imbalanced data well
- Provides feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("GRADIENT BOOSTING (XGBoost) CLASSIFIER - MALWARE DETECTION")
print("="*80)

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] Loading preprocessed data...")

# Load training and test sets
train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

# Separate features (X) and target (y)
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Feature names: {list(X_train.columns)}")

# ============================================================================
# 2. BUILD GRADIENT BOOSTING MODEL
# ============================================================================
print("\n[2] Building XGBoost model...")

# Initialize XGBoost Classifier
# Parameters:
#   n_estimators: Number of boosting rounds (trees)
#   max_depth: Maximum depth of each tree (prevents overfitting)
#   learning_rate: Step size shrinkage to prevent overfitting
#   subsample: Fraction of samples to use for each tree
#   colsample_bytree: Fraction of features to use for each tree
#   gamma: Minimum loss reduction required for split
#   reg_alpha: L1 regularization (Lasso)
#   reg_lambda: L2 regularization (Ridge)
#   random_state: For reproducibility
#   n_jobs: Number of CPU cores (-1 = all)
#   eval_metric: Metric to optimize during training
xgb_model = XGBClassifier(
    n_estimators=200,           # Build 200 trees
    max_depth=6,                # Limit tree depth to 6
    learning_rate=0.1,          # Learning rate (eta)
    subsample=0.8,              # Use 80% of samples per tree
    colsample_bytree=0.8,       # Use 80% of features per tree
    gamma=0,                    # Minimum loss reduction
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    random_state=42,            # For reproducibility
    n_jobs=-1,                  # Use all CPU cores
    eval_metric='logloss',      # Optimization metric
    use_label_encoder=False     # Disable deprecated label encoder warning
)

print(f"Model configuration:")
print(f"  - Number of estimators: {xgb_model.n_estimators}")
print(f"  - Max depth: {xgb_model.max_depth}")
print(f"  - Learning rate: {xgb_model.learning_rate}")
print(f"  - Subsample: {xgb_model.subsample}")
print(f"  - Column sample by tree: {xgb_model.colsample_bytree}")
print(f"  - L1 regularization (alpha): {xgb_model.reg_alpha}")
print(f"  - L2 regularization (lambda): {xgb_model.reg_lambda}")

# ============================================================================
# 3. TRAIN THE MODEL
# ============================================================================
print("\n[3] Training XGBoost model...")

# Record training time
start_time = time.time()

# Train the model with evaluation set for monitoring
# eval_set allows us to track performance during training
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False  # Set to True to see training progress
)

# Calculate training time
training_time = time.time() - start_time

print(f"\n[OK] Training completed in {training_time:.2f} seconds")

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================
print("\n[4] Making predictions...")

# Predict on training set (to check for overfitting)
y_train_pred = xgb_model.predict(X_train)
y_train_proba = xgb_model.predict_proba(X_train)[:, 1]  # Probability of malware

# Predict on test set (unseen data)
y_test_pred = xgb_model.predict(X_test)
y_test_proba = xgb_model.predict_proba(X_test)[:, 1]  # Probability of malware

print("[OK] Predictions completed")

# ============================================================================
# 5. EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n[5] Evaluating model performance...")

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

# Display results
print("\n" + "="*80)
print("PERFORMANCE METRICS")
print("="*80)

print("\nTraining Set Performance:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print(f"  ROC-AUC:   {train_auc:.4f}")

print("\nTest Set Performance:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_auc:.4f}")

# Check for overfitting
print("\nOverfitting Analysis:")
accuracy_diff = train_accuracy - test_accuracy
print(f"  Accuracy difference: {accuracy_diff:.4f}")
if accuracy_diff < 0.05:
    print("  Status: [OK] Good - No significant overfitting")
elif accuracy_diff < 0.10:
    print("  Status: ⚠ Moderate - Some overfitting detected")
else:
    print("  Status: ✗ High - Significant overfitting")

# Detailed classification report
print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT (Test Set)")
print("="*80)
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malware']))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix (Test Set):")
print(f"                Predicted")
print(f"              Benign  Malware")
print(f"Actual Benign   {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"       Malware  {cm[1,0]:5d}   {cm[1,1]:5d}")

# ============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[6] Analyzing feature importance...")

# Get feature importances from XGBoost
# XGBoost provides multiple importance types:
# - 'weight': Number of times a feature appears in a tree
# - 'gain': Average gain when the feature is used for splitting
# - 'cover': Average coverage of the feature when used
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 7. SAVE MODEL AND RESULTS
# ============================================================================
print("\n[7] Saving model and results...")

# Save the trained model
model_path = '../models/xgboost_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"[OK] Model saved: {model_path}")

# Save feature importance
importance_path = '../results/xgb_feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"[OK] Feature importance saved: {importance_path}")

# Save performance metrics
metrics_dict = {
    'Model': 'XGBoost',
    'Training_Time_Sec': training_time,
    'Train_Accuracy': train_accuracy,
    'Train_Precision': train_precision,
    'Train_Recall': train_recall,
    'Train_F1': train_f1,
    'Train_AUC': train_auc,
    'Test_Accuracy': test_accuracy,
    'Test_Precision': test_precision,
    'Test_Recall': test_recall,
    'Test_F1': test_f1,
    'Test_AUC': test_auc,
    'Overfitting_Gap': accuracy_diff
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_path = '../results/xgb_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"[OK] Metrics saved: {metrics_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted_Label': y_test_pred,
    'Malware_Probability': y_test_proba
})
pred_path = '../results/xgb_predictions.csv'
predictions_df.to_csv(pred_path, index=False)
print(f"[OK] Predictions saved: {pred_path}")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
print("\n[8] Creating visualizations...")

# Visualization 1: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('XGBoost - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../visualizations/xgb_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: xgb_confusion_matrix.png")
plt.close()

# Visualization 2: Feature Importance Bar Plot
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values, color='darkorange', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('XGBoost - Top 15 Feature Importances', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../visualizations/xgb_feature_importance.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: xgb_feature_importance.png")
plt.close()

# Visualization 3: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('XGBoost - ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/xgb_roc_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: xgb_roc_curve.png")
plt.close()

# Visualization 4: Prediction Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='Benign', color='blue', edgecolor='black')
plt.hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='Malware', color='red', edgecolor='black')
plt.xlabel('Predicted Probability of Malware', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('XGBoost - Prediction Probability Distribution', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/xgb_probability_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: xgb_probability_distribution.png")
plt.close()

# Visualization 5: Training History (if available)
# Get evaluation results from training
results = xgb_model.evals_result()
if results:
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Train')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Test')
    plt.xlabel('Boosting Round', fontsize=12, fontweight='bold')
    plt.ylabel('Log Loss', fontsize=12, fontweight='bold')
    plt.title('XGBoost - Training History', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/xgb_training_history.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: xgb_training_history.png")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("XGBOOST MODEL - TRAINING COMPLETE")
print("="*80)
print(f"\n[OK] Model trained successfully in {training_time:.2f} seconds")
print(f"[OK] Test Accuracy: {test_accuracy*100:.2f}%")
print(f"[OK] Test F1-Score: {test_f1:.4f}")
print(f"[OK] ROC-AUC: {test_auc:.4f}")
print(f"\nTop 3 Most Important Features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
print("\n" + "="*80)
