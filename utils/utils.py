import gzip
import shutil
from keras.models import load_model
from keras.models import Model, Sequential
from sklearn.base import BaseEstimator
import joblib

def identify_model_type(model):
    if isinstance(model, (Model, Sequential)):
        return 1 # Keras Model
    elif isinstance(model, BaseEstimator):
        return 0 # Scikit-learn Model
    else:
        return 2 

def save_compressed_model(model, filename):
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
    with gzip.open(filename, 'rb') as f_in:
        if filename.endswith('.h5'):
            return load_model(f_in)
        else:
            return joblib.load(f_in)