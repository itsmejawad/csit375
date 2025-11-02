"""
Random Forest Classifier for Malware Detection
==============================================

This script trains a Random Forest model to classify malware vs benign samples
using process and memory management features.

Random Forest is an ensemble learning method that:
- Builds multiple decision trees during training
- Makes predictions by voting across all trees
- Handles non-linear relationships well
- Provides feature importance rankings
- Resistant to overfitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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
print("RANDOM FOREST CLASSIFIER - MALWARE DETECTION")
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
# 2. BUILD RANDOM FOREST MODEL
# ============================================================================
print("\n[2] Building Random Forest model...")

# Initialize Random Forest Classifier
# Parameters:
#   n_estimators: Number of trees in the forest (more = better but slower)
#   max_depth: Maximum depth of each tree (None = unlimited)
#   min_samples_split: Minimum samples required to split a node
#   min_samples_leaf: Minimum samples required at a leaf node
#   random_state: For reproducibility
#   n_jobs: Number of CPU cores to use (-1 = use all)
#   verbose: Show progress during training
rf_model = RandomForestClassifier(
    n_estimators=100,           # Build 100 decision trees
    max_depth=None,             # No limit on tree depth
    min_samples_split=2,        # Default splitting criterion
    min_samples_leaf=1,         # Minimum 1 sample per leaf
    random_state=42,            # For reproducible results
    n_jobs=-1,                  # Use all CPU cores
    verbose=1                   # Show training progress
)

print(f"Model configuration:")
print(f"  - Number of trees: {rf_model.n_estimators}")
print(f"  - Max depth: {rf_model.max_depth}")
print(f"  - Min samples split: {rf_model.min_samples_split}")
print(f"  - Random state: {rf_model.random_state}")

# ============================================================================
# 3. TRAIN THE MODEL
# ============================================================================
print("\n[3] Training Random Forest model...")

# Record training time
start_time = time.time()

# Train the model on training data
rf_model.fit(X_train, y_train)

# Calculate training time
training_time = time.time() - start_time

print(f"\n[OK] Training completed in {training_time:.2f} seconds")

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================
print("\n[4] Making predictions...")

# Predict on training set (to check for overfitting)
y_train_pred = rf_model.predict(X_train)
y_train_proba = rf_model.predict_proba(X_train)[:, 1]  # Probability of class 1 (malware)

# Predict on test set (unseen data)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability of class 1 (malware)

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

# Get feature importances from the model
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# 7. SAVE MODEL AND RESULTS
# ============================================================================
print("\n[7] Saving model and results...")

# Save the trained model
model_path = '../models/random_forest_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"[OK] Model saved: {model_path}")

# Save feature importance
importance_path = '../results/rf_feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)
print(f"[OK] Feature importance saved: {importance_path}")

# Save performance metrics
metrics_dict = {
    'Model': 'Random Forest',
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
metrics_path = '../results/rf_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"[OK] Metrics saved: {metrics_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'True_Label': y_test.values,
    'Predicted_Label': y_test_pred,
    'Malware_Probability': y_test_proba
})
pred_path = '../results/rf_predictions.csv'
predictions_df.to_csv(pred_path, index=False)
print(f"[OK] Predictions saved: {pred_path}")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
print("\n[8] Creating visualizations...")

# Visualization 1: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('Random Forest - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../visualizations/rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: rf_confusion_matrix.png")
plt.close()

# Visualization 2: Feature Importance Bar Plot
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values, color='forestgreen', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Random Forest - Top 15 Feature Importances', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../visualizations/rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: rf_feature_importance.png")
plt.close()

# Visualization 3: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Random Forest - ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/rf_roc_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: rf_roc_curve.png")
plt.close()

# Visualization 4: Prediction Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='Benign', color='blue', edgecolor='black')
plt.hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='Malware', color='red', edgecolor='black')
plt.xlabel('Predicted Probability of Malware', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Random Forest - Prediction Probability Distribution', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/rf_probability_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: rf_probability_distribution.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RANDOM FOREST MODEL - TRAINING COMPLETE")
print("="*80)
print(f"\n[OK] Model trained successfully in {training_time:.2f} seconds")
print(f"[OK] Test Accuracy: {test_accuracy*100:.2f}%")
print(f"[OK] Test F1-Score: {test_f1:.4f}")
print(f"[OK] ROC-AUC: {test_auc:.4f}")
print(f"\nTop 3 Most Important Features:")
for idx, row in feature_importance.head(3).iterrows():
    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
print("\n" + "="*80)
