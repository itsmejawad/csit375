import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import pickle
import time


# Loading Data
train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Build Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)

# Training Model
start_time = time.time()
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Making Predictions
y_train_pred = rf_model.predict(X_train)
y_train_proba = rf_model.predict_proba(X_train)[:, 1] # [:, 1] ":" = all rows, "1" = second column

y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluating Performance

# Training Metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

# Test Metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

# Printing Results
print("PERFORMANCE METRICS")
print("\nTraining Set:")
print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")
print(f"  ROC-AUC:   {train_auc:.4f}")

print("\nTest Set:")
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")
print(f"  ROC-AUC:   {test_auc:.4f}")

# Overfitting Check
accuracy_diff = train_accuracy - test_accuracy
print(f"\nOverfitting Gap: {accuracy_diff:.4f}")

# Confusion Matrix
print("\nCONFUSION MATRIX:")
cm = confusion_matrix(y_test, y_test_pred)
print(f"{'':15} Predicted")
print(f"{'':13} Benign  Malware")
print(f"Actual Benign  {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"       Malware {cm[1,0]:6d}  {cm[1,1]:6d}")

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))




# Save model and results
print("\nSaving Model and Results...")

with open('../models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# feature_importance.to_csv('../results/rf_feature_importance.csv', index=False)

# metrics_df = pd.DataFrame([{
#     'Model': 'Random Forest',
#     'Training_Time_Sec': training_time,
#     'Train_Accuracy': train_accuracy,
#     'Train_Precision': train_precision,
#     'Train_Recall': train_recall,
#     'Train_F1': train_f1,
#     'Train_AUC': train_auc,
#     'Test_Accuracy': test_accuracy,
#     'Test_Precision': test_precision,
#     'Test_Recall': test_recall,
#     'Test_F1': test_f1,
#     'Test_AUC': test_auc,
#     'Overfitting_Gap': accuracy_diff
# }])
# metrics_df.to_csv('../results/rf_metrics.csv', index=False)

# predictions_df = pd.DataFrame({
#     'True_Label': y_test.values,
#     'Predicted_Label': y_test_pred,
#     'Malware_Probability': y_test_proba
# })
# predictions_df.to_csv('../results/rf_predictions.csv', index=False)



# Create visualizations

# Visualization 1: Confusion Matrix
# Shows how many samples were correctly/incorrectly classified
# Diagonal = correct predictions, Off-diagonal = errors
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('Random Forest - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../visualizations/rf_confusion_matrix.png', dpi=300)
plt.close()

# Visualization 2: Feature Importance
# Shows which features were most important for the model's decisions
# Higher bars = more important features
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Random Forest - Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../visualizations/rf_feature_importance.png', dpi=300)
plt.close()

# # Visualization 3: ROC Curve
# # Measures the model's ability to distinguish between classes
# # AUC closer to 1.0 = better performance
# # Curve closer to top-left corner = better model
# fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
# plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Random Forest - ROC Curve')
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('../visualizations/rf_roc_curve.png', dpi=300)
# plt.close()

# print("\n" + "="*50)
# print("TRAINING COMPLETE")
# print("="*50)
# print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
# print(f"Test F1-Score: {test_f1:.4f}")
# print(f"ROC-AUC: {test_auc:.4f}")
# print(f"Training Time: {training_time:.2f}s")
# print("\n" + "="*50)
