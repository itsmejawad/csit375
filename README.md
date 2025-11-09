# Malware Detection using Machine Learning

A machine learning project that classifies malware vs benign samples using three different models: Random Forest, XGBoost, and Deep Neural Network.

## Dataset

- **Total Samples**: 100,000 (50,000 malware, 50,000 benign)
- **Features**: 21 behavioral features from Linux processes
- **Source**: Process monitoring metrics (CPU time, memory, context switches, etc.)

## Models & Performance

| Model          | Test Accuracy | Training Time | Errors |
| -------------- | ------------- | ------------- | ------ |
| Random Forest  | 100.00%       | 1.38s         | 0      |
| XGBoost        | 100.00%       | 1.79s         | 0      |
| Neural Network | 99.985%       | 468.86s       | 3      |
