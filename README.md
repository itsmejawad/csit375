# Malware Detection using Machine Learning

A machine learning project that classifies malware vs benign samples using three different models: Random Forest, XGBoost, and Deep Neural Network.

## Dataset

- **Total Samples**: 100,000 (50,000 malware, 50,000 benign)
- **Features**: 21 behavioral features from Linux processes
- **Source**: Process monitoring metrics (CPU time, memory, context switches, etc.)

## Models & Performance

| Model | Test Accuracy | Training Time | Errors |
|-------|--------------|---------------|---------|
| Random Forest | 100.00% | 1.38s | 0 |
| XGBoost | 100.00% | 1.79s | 0 |
| Neural Network | 99.985% | 468.86s | 3 |

## Project Structure

```
project/
├── data/                      # Datasets (train/test splits)
├── models/                    # Trained models (.pkl, .h5)
├── scripts/                   # Training scripts
├── results/                   # Metrics, predictions, feature importance
├── visualizations/            # Charts and plots
└── docs/                      # Documentation
```

## Quick Start

1. Install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
```

2. Run preprocessing:
```bash
python scripts/preprocessing_and_visualization.py
```

3. Train models:
```bash
python scripts/1_random_forest_model.py
python scripts/2_gradient_boosting_model.py
python scripts/3_neural_network_model.py
```

## Results

All three models achieved excellent performance with minimal overfitting. Random Forest and XGBoost achieved perfect 100% accuracy, while the Neural Network achieved 99.985% accuracy with only 3 misclassifications out of 20,000 test samples.

See `docs/MODEL_COMPARISON_REPORT.md` for detailed analysis.

## Files

- `docs/MODEL_COMPARISON_REPORT.md` - Comprehensive model comparison
- `docs/FEATURE_EXPLANATIONS.md` - Feature descriptions
- `docs/PREPROCESSING_SUMMARY.md` - Data preprocessing details
