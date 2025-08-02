from utils.models import my_ann_model, my_svm_model, my_rf_model
from utils.models import compare_models
from utils.utils import data_preparation, get_data_pipeline, train_test_split
import datetime as dt
import warnings
import os

# -- RUN.py --
# Main script for ML workflow: prepares data, splits into train/validation/test, transforms features, trains and compares ANN/SVM/RF models, and saves evaluation metrics.
# Use this script to benchmark model performance on electricity load forecasting after updating data

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
warnings.filterwarnings('ignore')  # Suppress Python warnings (including Keras and protobuf)

# 1. Data preparation
print("=> Preparing data...")
data = data_preparation()

# 2. Train-test split parameters
print("=> Splitting data into train, validation, and test sets...")
train_date_cutoff = dt.datetime(2008, 1, 1)
validation_date_cutoff = dt.datetime(2009, 1, 1)

((X_train, X_valid, X_test), (y_train, y_valid, y_test), (dates_train, dates_valid, dates_test)) = train_test_split(
    data=data, target_name='SYSLoad',
    train_date_cutoff=train_date_cutoff,
    validation_date_cutoff=validation_date_cutoff
)

print(f"    -- Train date cutoff: {train_date_cutoff.year} Samples: {X_train.shape[0]}\n    -- Validation date cutoff: {validation_date_cutoff.year} Samples: {X_valid.shape[0]}\n    -- Testing Samples: {X_test.shape[0]}")

# 3. Data transformation pipeline
print("=> Fitting Data Pipeline...")
# 3.1 fit pipeline
pipeline = get_data_pipeline() # sklearn Pipeline object

# 3.2 Transform data
print("=> Transforming Data...")
X_train = pipeline.fit_transform(X_train)
X_valid = pipeline.transform(X_valid)
X_test = pipeline.transform(X_test)


# 4. Model Intantiation
# 4.1 models
input_dim = X_train.shape[1]
ann = my_ann_model(input_dim=input_dim)
svm = my_svm_model()
rf = my_rf_model()

# model training params
# -- model params excepts
# -- EPOCHS: int, BATCH_SIZE: int, early_stopping: bool

models_dict = {
    'ANN': {'model_instance': ann, 'model_params':{'EPOCHS': 50, 'BATCH_SIZE': 150, 'early_stopping':True}},
    'SVM': {'model_instance': svm, 'model_params':None},
    'RF': {'model_instance': rf, 'model_params':None}
}

# 5. Model Comparison
# Compare models
print("=> Comparing models...")
score_metrics = compare_models(models_dict, 
                                X_train, y_train, 
                                X_valid, y_valid, 
                                X_test, y_test)

# Storing the results in .csv
result_file_path = 'models_scores.csv'
score_metrics.to_csv(result_file_path)

print(f'Results stored: {result_file_path}')
print("Model comparison completed successfully.")