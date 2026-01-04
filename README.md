# Federated Learning for CSI (Channel State Information) Prediction

A federated learning implementation using Flower framework with LightGBM and Optuna hyperparameter tuning.

## 📊 Performance
- **Best Model:** LightGBM (single model, no ensemble)
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Score:** **12.18668 MAE** on Kaggle competition

## 🚀 Quick Start

### Prerequisites
```bash
pip install flwr flwr[simulation] lightgbm optuna pandas scikit-learn
```

### To Run the Code
```bash
python simulate.py
```
```bash
python predict.py
```
