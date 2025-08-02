# Electricity_Prediction_models
Prediction of Electricity Load Prediction Output based on three models Artificial Neural Network (ANN), Random Forest (RFR), Support Vector Machine (SVM)

The Goal of the study is Prepare, Train and Evaluate three different Machine learning models on Load Prediction and analyse the results.

# Electricity Load & Price Forecasting

This project benchmarks machine learning models for predicting electricity system load using historical weather and calendar data.

- **Dataset:** The raw data files can be obtained directly from ISO New England (www.iso-ne.com)

- **Training Testing Split:** Training data used to train these model is from 2004 to 2008. While Validation Set is from 2008 to 2009.

## Features

- **Data Preparation:** Cleans and engineers features from raw CSV and Excel datasets.
- **EDA Notebook:** Visualizes distributions, correlations, and outliers.

- **Feature Engineering:** Adds lag features, workday/holiday flags, and aggregates.
- **Model Pipeline:** Supports Random Forest, SVM, and ANN (Keras) models.
- **Evaluation:** Outputs metrics (MAE, MSE, R2, MAPE) for validation and test sets.
- **Comparison:** Easily compare model performance and export results to CSV.

## Workflow
1. **Prepare Data:**  
   Run `run.py` to preprocess data and split into train/validation/test sets.
2. **Fit Pipeline:**  
   Scales and transforms features using scikit-learn pipelines.
3. **Train Models:**  
   Trains Random Forest, SVM, and ANN models.
4. **Evaluate & Compare:**  
   Outputs metrics for each model and saves results to `models_scores.csv`.

## Usage
Installs Required Python Packages

```bash
pip install requirements.txt
```

command to whole ML Workflow
```bash
python run.py
```

## File Structure

**Summary of contents:**
- `datasets/`: Raw data files used for training and feature engineering.
- `notebooks/`: Interactive analysis and visualization for each model.
- `utils/`: Python modules for reusable code (data prep, modeling).
- `run.py`: Entry point for automated training and evaluation.
- `models_scores.csv`: Results file for model comparison.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project description
