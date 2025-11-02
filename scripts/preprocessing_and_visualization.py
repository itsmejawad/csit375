import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("MALWARE DATASET PREPROCESSING AND VISUALIZATION")
print("="*80)

# 1. LOAD DATA
print("\n[1] Loading dataset...")
df = pd.read_csv('Malware dataset.csv')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 2. EXPLORATORY DATA ANALYSIS
print("\n" + "="*80)
print("[2] EXPLORATORY DATA ANALYSIS")
print("="*80)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")

print("\nClass Distribution:")
print(df['classification'].value_counts())
print("\nClass Percentage:")
print(df['classification'].value_counts(normalize=True) * 100)

print("\nUnique samples (by hash):")
print(f"Total unique samples: {df['hash'].nunique()}")

# 3. DATA PREPROCESSING
print("\n" + "="*80)
print("[3] DATA PREPROCESSING")
print("="*80)

# Create a copy for preprocessing
df_processed = df.copy()

# Drop unnecessary columns (hash and millisecond - not useful for ML)
print("\nDropping 'hash' and 'millisecond' columns...")
df_processed = df_processed.drop(['hash', 'millisecond'], axis=1)
print(f"New shape: {df_processed.shape}")

# Encode target variable
print("\nEncoding target variable 'classification'...")
le = LabelEncoder()
df_processed['classification_encoded'] = le.fit_transform(df_processed['classification'])
print(f"Classes: {le.classes_}")
print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separate features and target
X = df_processed.drop(['classification', 'classification_encoded'], axis=1)
y = df_processed['classification_encoded']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check for duplicates
print(f"\nDuplicate rows: {X.duplicated().sum()}")

# Check for constant features
print("\nChecking for constant features...")
constant_features = [col for col in X.columns if X[col].nunique() <= 1]
if constant_features:
    print(f"Constant features found: {constant_features}")
    X = X.drop(constant_features, axis=1)
    print(f"Dropped {len(constant_features)} constant features")
else:
    print("No constant features found")

# Check for high correlation features (multicollinearity)
print("\nChecking correlation matrix...")
corr_matrix = X.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
print(f"Highly correlated features (>0.95): {len(high_corr_features)}")
if high_corr_features:
    print(f"Features: {high_corr_features}")

# Feature Scaling
print("\nApplying StandardScaler to features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Scaling completed!")
print(f"\nScaled data statistics:")
print(X_scaled_df.describe())

# 4. TRAIN-TEST SPLIT
print("\n" + "="*80)
print("[4] TRAIN-TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTraining class distribution:")
print(pd.Series(y_train).value_counts())
print(f"\nTest class distribution:")
print(pd.Series(y_test).value_counts())

# 5. SAVE PREPROCESSED DATA
print("\n" + "="*80)
print("[5] SAVING PREPROCESSED DATA")
print("="*80)

# Save full preprocessed dataset
X_scaled_df['target'] = y.values
X_scaled_df.to_csv('preprocessed_malware_data.csv', index=False)
print("Saved: preprocessed_malware_data.csv")

# Save train/test splits
train_df = X_train.copy()
train_df['target'] = y_train.values
train_df.to_csv('train_data.csv', index=False)
print("Saved: train_data.csv")

test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv('test_data.csv', index=False)
print("Saved: test_data.csv")

# Save original features (unscaled) for reference
X_original = df_processed.drop(['classification', 'classification_encoded'], axis=1)
if constant_features:
    X_original = X_original.drop(constant_features, axis=1)
X_original['target'] = y.values
X_original.to_csv('unscaled_malware_data.csv', index=False)
print("Saved: unscaled_malware_data.csv")

# Save scaler and encoder for future use
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved: scaler.pkl")

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("Saved: label_encoder.pkl")

# 6. VISUALIZATIONS
print("\n" + "="*80)
print("[6] CREATING VISUALIZATIONS")
print("="*80)

# Create visualizations directory
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')
    print("Created 'visualizations' directory")

# Visualization 1: Class Distribution
plt.figure(figsize=(10, 6))
class_counts = df['classification'].value_counts()
colors = ['#ff6b6b', '#4ecdc4']
plt.bar(class_counts.index, class_counts.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Class Distribution in Malware Dataset', fontsize=16, fontweight='bold')
plt.xlabel('Classification', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(fontsize=11)
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 100, str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/1_class_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/1_class_distribution.png")
plt.close()

# Visualization 2: Class Distribution Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=90, explode=(0.05, 0.05),
        textprops={'fontsize': 12, 'fontweight': 'bold'})
plt.title('Class Distribution (Percentage)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/2_class_distribution_pie.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/2_class_distribution_pie.png")
plt.close()

# Visualization 3: Correlation Heatmap (top features)
plt.figure(figsize=(14, 12))
correlation = X_original.drop('target', axis=1).corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=False, cmap='coolwarm',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/3_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/3_correlation_heatmap.png")
plt.close()

# Visualization 4: Distribution of top features by class
top_features = X.columns[:8]  # First 8 features
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    for class_label in df['classification'].unique():
        data = df[df['classification'] == class_label][feature]
        axes[idx].hist(data, alpha=0.6, label=class_label, bins=30, edgecolor='black')
    axes[idx].set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visualizations/4_feature_distributions.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/4_feature_distributions.png")
plt.close()

# Visualization 5: Box plots for top features
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    df.boxplot(column=feature, by='classification', ax=axes[idx],
               patch_artist=True, grid=False)
    axes[idx].set_title(f'{feature} by Class', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Classification', fontsize=9)
    axes[idx].set_ylabel(feature, fontsize=9)
    axes[idx].get_figure().suptitle('')

plt.suptitle('Box Plots: Feature Values by Class', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('visualizations/5_boxplots_by_class.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/5_boxplots_by_class.png")
plt.close()

# Visualization 6: Feature importance based on variance
feature_variance = X.var().sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 6))
plt.barh(range(len(feature_variance)), feature_variance.values, color='steelblue', edgecolor='black')
plt.yticks(range(len(feature_variance)), feature_variance.index)
plt.xlabel('Variance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 15 Features by Variance', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('visualizations/6_feature_variance.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/6_feature_variance.png")
plt.close()

# Visualization 7: Pairplot for top correlated features with target
# Select top 5 most varying features
top_varying_features = X.var().sort_values(ascending=False).head(5).index.tolist()
plot_df = df[top_varying_features + ['classification']].sample(n=min(5000, len(df)), random_state=42)
pairplot = sns.pairplot(plot_df, hue='classification', palette=colors,
                        plot_kws={'alpha': 0.6, 's': 20}, diag_kind='kde')
pairplot.fig.suptitle('Pairplot of Top 5 Varying Features', y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/7_pairplot_top_features.png', dpi=300, bbox_inches='tight')
print("Saved: visualizations/7_pairplot_top_features.png")
plt.close()

# Visualization 8: Missing values heatmap (if any)
if df.isnull().sum().sum() > 0:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visualizations/8_missing_values.png', dpi=300, bbox_inches='tight')
    print("Saved: visualizations/8_missing_values.png")
    plt.close()
else:
    print("No missing values to visualize")

print("\n" + "="*80)
print("PREPROCESSING AND VISUALIZATION COMPLETE!")
print("="*80)
print("\nOutput Files:")
print("- preprocessed_malware_data.csv (scaled features + target)")
print("- unscaled_malware_data.csv (original features + target)")
print("- train_data.csv (80% training set)")
print("- test_data.csv (20% test set)")
print("- scaler.pkl (StandardScaler for future use)")
print("- label_encoder.pkl (LabelEncoder for future use)")
print("\nVisualizations saved in 'visualizations/' directory")
print("\nYou can now use the preprocessed data for:")
print("1. Random Forest")
print("2. XGBoost/Gradient Boosting")
print("3. Neural Networks (Deep Learning)")
print("="*80)
