import gzip
import shutil
import joblib
import os

def identify_model_type(model):
    '''
        Identifies the type of model passed in.
    '''

    from keras.models import Model, Sequential
    from sklearn.base import BaseEstimator

    if isinstance(model, (Model, Sequential)):
        return 1 # Keras Model
    elif isinstance(model, BaseEstimator):
        return 0 # Scikit-learn Model
    else:
        return 2 

def save_compressed_model(model, filename):
    '''
        Saves the model to a file and compresses it using gzip.
        The filename should not have a .gz extension, it will be added automatically.
    '''
    # Add .gz extension for compressed file
    if not filename.endswith('.gz'):
        gz_filename = filename + '.gz'
    else:
        gz_filename = filename

    if identify_model_type(model) == 1:
        model.save(filename)
        with open(filename, 'rb') as f_in:
            with gzip.open(gz_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif identify_model_type(model) == 0:
        # Save to .pkl first, then compress
        joblib.dump(model, filename)
        with open(filename, 'rb') as f_in:
            with gzip.open(gz_filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError("Unsupported model type for compression")
    
def load_compressed_model(filename):
    from keras.models import load_model
    with gzip.open(filename, 'rb') as f_in:
        if filename.endswith('.h5'):
            return load_model(f_in)
        else:
            return joblib.load(f_in)

def get_dataset_file_path(file_name):
    # Get current working directory
    cwd = os.getcwd()
    # Construct relative path to the dataset
    relative_path = os.path.join(cwd, "datasets", file_name)
    # Convert to absolute path
    absolute_path = os.path.abspath(relative_path)

    return absolute_path

def data_preparation():
    '''
    Prepares the features DataFrame for the electricity load forecasting model.
    This function loads the dataset, processes the features, and returns a DataFrame
    '''
    # Import necessary libraries
    import pandas as pd
    import numpy as np

    # load data
    print(" 1. Loading data...")
    data_file_name = 'Book3.csv'
    holidays_file_name = 'holidays.xls'
    
    data = pd.read_csv(get_dataset_file_path(data_file_name))
    holidays = pd.read_excel(get_dataset_file_path(holidays_file_name))['Date'].values

    # Prepare features
    print(" 2. Preparing features...")
    data['Date'] = pd.to_datetime(data['Date'])
    dayofweek = data.Date.dt.weekday
    isworkday = np.isin(dayofweek, [0,1,2,3,4]) & ~np.isin(data['Date'], holidays)
    prevdaysamehour = np.hstack(((np.ones(24)*-1), (data['SYSLoad'][0:-24])))
    prevweeksamehour = np.hstack(((np.ones(168)*-1), (data['SYSLoad'][0:-168])))
    import scipy.signal
    prev24houravg = scipy.signal.lfilter(np.ones(24) / 24, 1, data['SYSLoad'])

    # Build DataFrame for pipeline
    print(" 3. Building features DataFrame...")
    features_df = pd.DataFrame({
        'Date': data['Date'],
        'DryBulb': data['DryBulb'],
        'DewPnt': data['DewPnt'],
        'Hour': data['Hour'],
        'DayOfWeek': dayofweek,
        'IsWorkday': isworkday,
        'PrevWeekSameHour': prevweeksamehour,
        'PrevDaySameHour': prevdaysamehour,
        'Prev24HourAvg': prev24houravg,
        'SYSLoad': data['SYSLoad']
    })

    # Remove first 168 rows with nulls
    print(" 4. Cleaning features DataFrame...")
    features_df = features_df.iloc[168:].reset_index(drop=True)

    # Convert all 'dtypes' to int32
    print(" 5. Converting dtypes...")
    features_df = features_df.astype({
        'DryBulb': 'int32',
        'DewPnt': 'int32',
        'Hour': 'int32',
        'DayOfWeek': 'int32',
        'IsWorkday': 'int32',
        'PrevWeekSameHour': 'float32',
        'PrevDaySameHour': 'float32',
        'Prev24HourAvg': 'float32',
        'SYSLoad': 'float32'
    })

    return features_df

def get_data_pipeline():
    '''
        Creates a preprocessing pipeline for the electricity load forecasting model.
        returns a sklearn pipeline object
    '''

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    # Prepare features DataFrame
    scale_cols = ['DryBulb', 'DewPnt', 'PrevWeekSameHour', 'PrevDaySameHour', 'Prev24HourAvg']
    passthrough_cols = ['IsWorkday', 'Hour', 'DayOfWeek']

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), scale_cols),
            ('cat', 'passthrough', passthrough_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocess', preprocessor)
    ])

    return pipeline

def train_test_split(data, 
                     target_name, 
                     train_date_cutoff,
                     validation_date_cutoff):
    '''
        Splits the features DataFrame into training and testing sets.
        returns X_train, X_test, y_train, y_test
    '''
    import datetime as dt
    import pandas as pd
    import numpy as np

    X = data.drop(columns=['Date', target_name])
    y = data[target_name].astype('float32')

    # specify dates
    dates = data['Date']

    # Train-test split Indices
    nTrain = dates < train_date_cutoff
    nValid = (np.logical_and(dates >= train_date_cutoff, dates < validation_date_cutoff))
    nTest = (dates >= validation_date_cutoff)

    # X features
    X_train = X.loc[nTrain]
    X_valid = X.loc[nValid]
    X_test = X.loc[nTest]

    # y target
    y_train = y.loc[nTrain]
    y_valid = y.loc[nValid]
    y_test = y.loc[nTest]

    # Dates
    dates_train = dates.loc[nTrain]
    dates_valid = dates.loc[nValid]
    dates_test = dates.loc[nTest]


    return ((X_train, X_valid, X_test), 
            (y_train, y_valid, y_test),
            (dates_train, dates_valid, dates_test))