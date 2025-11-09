import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# StandardScaler: Used to scale the features to have a mean of 0 and a standard deviation of 1
# LabelEncoder: Used to encode the target variable into a numerical value for the model to understand
from sklearn.preprocessing import StandardScaler, LabelEncoder

# train_test_split: Used to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# pickle: Used to save the model and the scaler
import pickle

df = pd.read_csv('../data/malware_dataset.csv') # Load the dataset

print(f"Dataset shape: {df.shape}") # Print the shape of the dataset

# Print the basic info of the dataset
print(f"\nDataset Info:") 
print(df.info())

print(f"\nMissing values:\n{df.isnull().sum()}") # Print the missing values in the dataset


# Print the class distribution in the dataset
print(f"\nClass distribution:")
print(df['classification'].value_counts())

# Separate features and target
X = df.drop(['hash', 'millisecond', 'classification'], axis=1)
y = df['classification']

# Encode labels (benign = 0, malware = 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"Original features: {X.shape[1]}")

# Remove constant features (features with zero variance)
constant_features = []
for col in X.columns:
    if X[col].nunique() == 1:
        constant_features.append(col)

X = X.drop(columns=constant_features)

print(f"Features after removal: {X.shape[1]}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_df['target'] = y # Add target column



# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df.drop('target', axis=1),
    X_scaled_df['target'],
    test_size=0.2,
    random_state=1,
    stratify=X_scaled_df['target']
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Save datasets
train_df = X_train.copy()
train_df['target'] = y_train
train_df.to_csv('../data/train_data.csv', index=False)

test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('../data/test_data.csv', index=False)

# Save scaler and encoder
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('../models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\nSaved preprocessed data and models")

# Visualizations

# 1. Class distribution
plt.figure(figsize=(8, 6))
classes = ['Benign', 'Malware']
counts = pd.Series(y).value_counts().sort_index()
plt.bar(classes, counts, color=['green', 'red'])
plt.title('Class Distribution')
plt.ylabel('Count')
plt.savefig('../visualizations/1_class_distribution.png')
plt.close()


# 2. Correlation heatmap
plt.figure(figsize=(12, 10))
corr = X_scaled_df.drop('target', axis=1).corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('../visualizations/3_correlation_heatmap.png')
plt.close()

# 3. Feature variance
plt.figure(figsize=(12, 8))
variance = X.var().sort_values(ascending=False)
plt.bar(range(len(variance)), variance.values)
plt.title('Feature Variance')
plt.xlabel('Feature Index')
plt.ylabel('Variance')
plt.savefig('../visualizations/6_feature_variance.png')
plt.close()


print("\nPreprocessing complete!")
