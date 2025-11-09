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






# Load the dataset
df = pd.read_csv('Malware dataset.csv')

# Print the shape of the dataset
print(f"Dataset shape: {df.shape}")

# Print the basic info of the dataset
print(f"\nDataset Info:")
print(df.info())

# Print the missing values in the dataset
print(f"\nMissing values:\n{df.isnull().sum()}")

# Print the class distribution in the dataset
print(f"\nClass distribution:")
print(df['classification'].value_counts())

# Separate features and target
X = df.drop(['hash', 'millisecond', 'classification'], axis=1)
y = df['classification']

# Encode labels (benign = 0, malware = 1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"\nOriginal features: {X.shape[1]}")

# Remove constant features (features with zero variance)
constant_features = []
for col in X.columns:
    if X[col].nunique() == 1:
        constant_features.append(col)

print(f"Removing {len(constant_features)} constant features...")
X = X.drop(columns=constant_features)
print(f"Features after removal: {X.shape[1]}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Add target column
X_scaled_df['target'] = y

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df.drop('target', axis=1),
    X_scaled_df['target'],
    test_size=0.2,
    random_state=42,
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
print("\nGenerating visualizations...")

# 1. Class distribution
plt.figure(figsize=(8, 6))
classes = ['Benign', 'Malware']
counts = pd.Series(y).value_counts().sort_index()
plt.bar(classes, counts, color=['green', 'red'])
plt.title('Class Distribution')
plt.ylabel('Count')
plt.savefig('../visualizations/1_class_distribution.png')
plt.close()

# 2. Pie chart
plt.figure(figsize=(8, 6))
plt.pie(counts, labels=classes, autopct='%1.1f%%', colors=['green', 'red'])
plt.title('Class Distribution')
plt.savefig('../visualizations/2_class_distribution_pie.png')
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(12, 10))
corr = X_scaled_df.drop('target', axis=1).corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('../visualizations/3_correlation_heatmap.png')
plt.close()

# 4. Feature distributions
fig, axes = plt.subplots(5, 5, figsize=(20, 16))
features = X.columns[:21]
for idx, feature in enumerate(features):
    row = idx // 5
    col = idx % 5
    axes[row, col].hist(X[feature], bins=50, edgecolor='black')
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')
plt.tight_layout()
plt.savefig('../visualizations/4_feature_distributions.png')
plt.close()

# 5. Boxplots by class
fig, axes = plt.subplots(5, 5, figsize=(20, 16))
for idx, feature in enumerate(features):
    row = idx // 5
    col = idx % 5
    data = pd.DataFrame({feature: X[feature], 'class': y})
    data.boxplot(column=feature, by='class', ax=axes[row, col])
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel('Class (0=Benign, 1=Malware)')
plt.tight_layout()
plt.savefig('../visualizations/5_boxplots_by_class.png')
plt.close()

# 6. Feature variance
plt.figure(figsize=(12, 8))
variance = X.var().sort_values(ascending=False)
plt.bar(range(len(variance)), variance.values)
plt.title('Feature Variance')
plt.xlabel('Feature Index')
plt.ylabel('Variance')
plt.savefig('../visualizations/6_feature_variance.png')
plt.close()

# 7. Pairplot of top features
top_features = variance.head(4).index.tolist()
pairplot_data = X[top_features].copy()
pairplot_data['class'] = y
sns.pairplot(pairplot_data, hue='class', palette={0: 'green', 1: 'red'})
plt.savefig('../visualizations/7_pairplot_top_features.png')
plt.close()

print("\nPreprocessing complete!")
