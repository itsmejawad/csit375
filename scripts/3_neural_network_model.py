# Neural Network (Deep Learning) Model for Malware Detection

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import pickle
import time

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

print("Neural Network - Malware Detection")
print("="*50)
print(f"TensorFlow Version: {tf.__version__}")

# Load data
print("\nLoading data...")
train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

X_train = train_df.drop('target', axis=1).values
y_train = train_df['target'].values
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target'].values

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Build Neural Network
print("\nBuilding Neural Network...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dropout(0.2),

    Dense(1, activation='sigmoid')
])

# Compile model
print("\nCompiling model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

model.summary()

# Setup callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [reduce_lr]

# Train model
print("\nTraining model (100 epochs)...")
start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds")

# Make predictions
print("\nMaking predictions...")
y_train_proba = model.predict(X_train).flatten()
y_train_pred = (y_train_proba > 0.5).astype(int)

y_test_proba = model.predict(X_test).flatten()
y_test_pred = (y_test_proba > 0.5).astype(int)

# Evaluate performance
print("\nEvaluating model...")

# Training metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_auc = roc_auc_score(y_train, y_train_proba)

# Test metrics
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

# Print results
print("\n" + "="*50)
print("PERFORMANCE METRICS")
print("="*50)

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

# Overfitting check
accuracy_diff = train_accuracy - test_accuracy
print(f"\nOverfitting Gap: {accuracy_diff:.4f}")

# Confusion matrix
print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(y_test, y_test_pred)
print(f"\n{'':15} Predicted")
print(f"{'':13} Benign  Malware")
print(f"Actual Benign  {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"       Malware {cm[1,0]:6d}  {cm[1,1]:6d}")

# Classification report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malware']))

# Save model and results
print("\nSaving model and results...")

model.save('../models/neural_network_model.h5')

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('../results/nn_training_history.csv', index=False)

# Save metrics
metrics_df = pd.DataFrame([{
    'Model': 'Neural Network',
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
    'Overfitting_Gap': accuracy_diff,
    'Total_Epochs': len(history.history['loss'])
}])
metrics_df.to_csv('../results/nn_metrics.csv', index=False)

# Save predictions
predictions_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_test_pred,
    'Malware_Probability': y_test_proba
})
predictions_df.to_csv('../results/nn_predictions.csv', index=False)

# Create visualizations
print("\nGenerating visualizations...")

# Visualization 1: Confusion Matrix
# Shows how many samples were correctly/incorrectly classified
# Diagonal = correct predictions, Off-diagonal = errors
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('Neural Network - Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('../visualizations/nn_confusion_matrix.png', dpi=300)
plt.close()

# Visualization 2: Training History (4 metrics)
# Shows how the model improved during training across 100 epochs
# Accuracy/Precision/Recall increasing = model learning
# Train and Test lines close together = no overfitting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Test')
axes[0, 0].set_title('Model Accuracy')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Test')
axes[0, 1].set_title('Model Loss')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train')
axes[1, 0].plot(history.history['val_precision'], label='Test')
axes[1, 0].set_title('Model Precision')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train')
axes[1, 1].plot(history.history['val_recall'], label='Test')
axes[1, 1].set_title('Model Recall')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/nn_training_history.png', dpi=300)
plt.close()

# Visualization 3: ROC Curve
# Measures the model's ability to distinguish between classes
# AUC closer to 1.0 = better performance
# Curve closer to top-left corner = better model
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Neural Network - ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/nn_roc_curve.png', dpi=300)
plt.close()

print("\n" + "="*50)
print("TRAINING COMPLETE")
print("="*50)
print(f"\nTest Accuracy: {test_accuracy*100:.2f}%")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"ROC-AUC: {test_auc:.4f}")
print(f"Training Time: {training_time:.2f}s")
print(f"Total Epochs: {len(history.history['loss'])}")
print("\n" + "="*50)
