import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# StandardScaler: Used to scale the Features to have a mean of 0 and a standard deviation of 1
# LabelEncoder: Used to encode the target variable into a numerical value for the model to understand
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Train-Test Split: Used to split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Pickle: Used to save the model and the Scaler
import pickle

df = pd.read_csv('../data/malware_dataset.csv') # Loading Dataset

print(f"Dataset shape: {df.shape}") # Printing Shape of the Dataset

# Printing Basic Info of the Dataset
print(f"\nDataset Info:") 
print(df.info())

print(f"\nMissing values:\n{df.isnull().sum()}") # Printing Missing Values in the Dataset


# Printing Class Distribution in the Dataset
print(f"\nClass distribution:")
print(df['classification'].value_counts())

# Separating Features and Target
X = df.drop(['hash', 'millisecond', 'classification'], axis=1)
y = df['classification']

# Encoding Labels (benign = 0, malware = 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"Original features: {X.shape[1]}")

# Removing Constant Features (Features with Zero Variance)
constant_features = []
for col in X.columns:
    if X[col].nunique() == 1:
        constant_features.append(col)

X = X.drop(columns=constant_features)

print(f"Features after removal: {X.shape[1]}")

# Scaling Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_df['target'] = y # Add target column



# Train-Test Split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df.drop('target', axis=1),
    X_scaled_df['target'],
    test_size=0.2,
    random_state=1,
    stratify=X_scaled_df['target']
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Saving Datasets
train_df = X_train.copy()
train_df['target'] = y_train
train_df.to_csv('../data/train_data.csv', index=False)

test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('../data/test_data.csv', index=False)

# Saving Scaler and Encoder
with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('../models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("\nSaved Preprocessed Data and Models")

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


