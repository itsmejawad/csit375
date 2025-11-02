"""
Neural Network (Deep Learning) for Malware Detection
===================================================

This script trains a Deep Neural Network using TensorFlow/Keras to classify malware.

Neural Networks:
- Learn complex non-linear patterns through multiple layers
- Can discover hidden feature interactions automatically
- Uses backpropagation and gradient descent for learning
- Requires more data and training time than traditional ML
- Can achieve very high accuracy with proper tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*80)
print("DEEP NEURAL NETWORK - MALWARE DETECTION")
print("="*80)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {len(gpus)} GPU(s) detected")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("GPU Available: No (using CPU)")

# ============================================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================================
print("\n[1] Loading preprocessed data...")

# Load training and test sets
train_df = pd.read_csv('../data/train_data.csv')
test_df = pd.read_csv('../data/test_data.csv')

# Separate features (X) and target (y)
X_train = train_df.drop('target', axis=1).values  # Convert to numpy array
y_train = train_df['target'].values
X_test = test_df.drop('target', axis=1).values
y_test = test_df['target'].values

# Get feature names for later use
feature_names = train_df.drop('target', axis=1).columns.tolist()

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Data type: {type(X_train)}")

# Create validation split from training data
# This will be used for early stopping
validation_split = 0.2  # Use 20% of training data for validation

# ============================================================================
# 2. BUILD NEURAL NETWORK ARCHITECTURE
# ============================================================================
print("\n[2] Building Deep Neural Network...")

# Network Architecture:
# Input Layer -> Hidden Layer 1 -> Dropout -> Hidden Layer 2 -> Dropout ->
# Hidden Layer 3 -> Dropout -> Output Layer

model = Sequential([
    # Input layer (automatically inferred from first Dense layer)
    # First hidden layer: 128 neurons with ReLU activation
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), name='hidden_layer_1'),
    BatchNormalization(),  # Normalize activations for stable training
    Dropout(0.3),          # Drop 30% of neurons to prevent overfitting

    # Second hidden layer: 64 neurons with ReLU activation
    Dense(64, activation='relu', name='hidden_layer_2'),
    BatchNormalization(),
    Dropout(0.3),

    # Third hidden layer: 32 neurons with ReLU activation
    Dense(32, activation='relu', name='hidden_layer_3'),
    BatchNormalization(),
    Dropout(0.2),

    # Fourth hidden layer: 16 neurons with ReLU activation
    Dense(16, activation='relu', name='hidden_layer_4'),
    Dropout(0.2),

    # Output layer: 1 neuron with sigmoid activation for binary classification
    # Sigmoid outputs probability between 0 and 1
    Dense(1, activation='sigmoid', name='output_layer')
])

# Display model architecture
print("\nModel Architecture:")
model.summary()

# ============================================================================
# 3. COMPILE THE MODEL
# ============================================================================
print("\n[3] Compiling the model...")

# Compile the model with:
# - Optimizer: Adam (adaptive learning rate)
# - Loss function: Binary crossentropy (for binary classification)
# - Metrics: Accuracy, Precision, Recall, AUC
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Learning rate = 0.001
    loss='binary_crossentropy',           # Binary classification loss
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

print("[OK] Model compiled successfully")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Loss function: Binary Crossentropy")
print(f"  Metrics: Accuracy, Precision, Recall, AUC")

# ============================================================================
# 4. SETUP TRAINING CALLBACKS
# ============================================================================
print("\n[4] Setting up training callbacks...")

# Callback 1: Learning Rate Reduction
# Reduces learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # Monitor validation loss
    factor=0.5,           # Reduce LR by half
    patience=5,           # Wait 5 epochs before reducing
    min_lr=1e-7,          # Minimum learning rate
    verbose=1
)

# Note: Early stopping removed to train for full 100 epochs
callbacks = [reduce_lr]

print("[OK] Callbacks configured:")
print("  - Learning Rate Reduction (factor=0.5, patience=5)")
print("  - Training for full 100 epochs (early stopping disabled)")

# ============================================================================
# 5. TRAIN THE MODEL
# ============================================================================
print("\n[5] Training Neural Network...")
print("This may take a few minutes...\n")

# Record training time
start_time = time.time()

# Train the model
# Validation split: 20% of training data used for validation
# Batch size: 32 samples per gradient update
# Epochs: Maximum 100 (early stopping may end training sooner)
history = model.fit(
    X_train, y_train,
    validation_split=validation_split,  # Use 20% for validation
    epochs=100,                         # Maximum epochs
    batch_size=32,                      # Batch size
    callbacks=callbacks,                # Use callbacks
    verbose=1                           # Show progress bar
)

# Calculate training time
training_time = time.time() - start_time

print(f"\n[OK] Training completed in {training_time:.2f} seconds")
print(f"  Total epochs trained: {len(history.history['loss'])}")

# ============================================================================
# 6. MAKE PREDICTIONS
# ============================================================================
print("\n[6] Making predictions...")

# Predict on training set (to check for overfitting)
y_train_proba = model.predict(X_train, verbose=0).flatten()
y_train_pred = (y_train_proba > 0.5).astype(int)  # Convert probabilities to binary

# Predict on test set (unseen data)
y_test_proba = model.predict(X_test, verbose=0).flatten()
y_test_pred = (y_test_proba > 0.5).astype(int)  # Threshold at 0.5

print("[OK] Predictions completed")

# ============================================================================
# 7. EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n[7] Evaluating model performance...")

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
# 8. SAVE MODEL AND RESULTS
# ============================================================================
print("\n[8] Saving model and results...")

# Save the trained model (Keras format)
model_path = '../models/neural_network_model.h5'
model.save(model_path)
print(f"[OK] Model saved: {model_path}")

# Save training history
history_df = pd.DataFrame(history.history)
history_path = '../results/nn_training_history.csv'
history_df.to_csv(history_path, index=False)
print(f"[OK] Training history saved: {history_path}")

# Save performance metrics
metrics_dict = {
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
}

metrics_df = pd.DataFrame([metrics_dict])
metrics_path = '../results/nn_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"[OK] Metrics saved: {metrics_path}")

# Save predictions
predictions_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_test_pred,
    'Malware_Probability': y_test_proba
})
pred_path = '../results/nn_predictions.csv'
predictions_df.to_csv(pred_path, index=False)
print(f"[OK] Predictions saved: {pred_path}")

# ============================================================================
# 9. CREATE VISUALIZATIONS
# ============================================================================
print("\n[9] Creating visualizations...")

# Visualization 1: Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=True,
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'])
plt.title('Neural Network - Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('../visualizations/nn_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: nn_confusion_matrix.png")
plt.close()

# Visualization 2: Training History - Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Model Loss During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

# Visualization 3: Training History - Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Model Accuracy During Training', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/nn_training_history.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: nn_training_history.png")
plt.close()

# Visualization 4: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('Neural Network - ROC Curve', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/nn_roc_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: nn_roc_curve.png")
plt.close()

# Visualization 5: Prediction Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='Benign', color='blue', edgecolor='black')
plt.hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='Malware', color='red', edgecolor='black')
plt.xlabel('Predicted Probability of Malware', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Neural Network - Prediction Probability Distribution', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../visualizations/nn_probability_distribution.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: nn_probability_distribution.png")
plt.close()

# Visualization 6: All Metrics History
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].set_title('Loss', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 1].set_title('Accuracy', fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
axes[1, 0].set_title('Precision', fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
axes[1, 1].set_title('Recall', fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.suptitle('Neural Network - All Training Metrics', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('../visualizations/nn_all_metrics.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: nn_all_metrics.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("NEURAL NETWORK MODEL - TRAINING COMPLETE")
print("="*80)
print(f"\n[OK] Model trained successfully in {training_time:.2f} seconds")
print(f"[OK] Total epochs: {len(history.history['loss'])}")
print(f"[OK] Test Accuracy: {test_accuracy*100:.2f}%")
print(f"[OK] Test F1-Score: {test_f1:.4f}")
print(f"[OK] ROC-AUC: {test_auc:.4f}")
print(f"\nModel Architecture:")
print(f"  - Input layer: {X_train.shape[1]} features")
print(f"  - Hidden layers: 128 -> 64 -> 32 -> 16 neurons")
print(f"  - Output layer: 1 neuron (sigmoid)")
print(f"  - Total parameters: {model.count_params():,}")
print("\n" + "="*80)
