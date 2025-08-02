from utils.utils import identify_model_type

def my_ann_model(input_dim, learning_rate=1e-3):
    from keras.optimizers import SGD
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization
    from keras.optimizers import Adam, RMSprop, SGD

    model2 = Sequential()
    model2.add(Dense(256, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.3))
    model2.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.2))
    model2.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
    model2.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model2.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model2

def my_svm_model():
    from sklearn.svm import SVR

    svm_model = SVR(kernel='rbf')
    return svm_model

def my_rf_model():
    from sklearn.ensemble import RandomForestRegressor

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    return rf_model

def evaluate_model(y_true, y_pred, prefix=None):
    '''
        Evaluates the model performance using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
        returns a dictionary with evaluation metrics
    '''
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    evaluation_results = {
        f'{prefix}MAE': mae,
        f'{prefix}MSE': mse,
        f'{prefix}R2': r2,
        f'{prefix}MAPE': mape,
    }
    return evaluation_results

def fit_evaluate_model(model, model_params, X_train, y_train, X_valid, y_valid, X_test, y_test):
    import pandas as pd
    if identify_model_type(model) == 1:  # Keras Model
        EPOCHS = model_params.get('EPOCHS')
        BATCH_SIZE = model_params.get('BATCH_SIZE')

        if model_params.get('early_stopping') == True:
            from keras.callbacks import EarlyStopping
            # Define the early stopping callback
            early_stop = EarlyStopping(
                monitor='val_loss',       # Metric to monitor
                patience=10,               # Number of epochs to wait before stopping
                restore_best_weights=True # Restore weights from best epoch
            )
        else:
            early_stop = None

        # Fit the model
        model.fit(X_train, y_train, 
                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_valid, y_valid),
                  verbose=0,
                  callbacks=[early_stop] if early_stop else None)

    elif identify_model_type(model) == 0:  # Scikit-learn Model
        model.fit(X_train, y_train)
    else:
        raise ValueError("Unsupported model type for evaluation")
    
    # Evaluate on validation set
    valid_preds = model.predict(X_valid)
    valid_score_metrics = evaluate_model(y_valid, valid_preds, prefix='valid_')

    # Evaluate on test set
    test_preds = model.predict(X_test)
    test_score_metrics = evaluate_model(y_test, test_preds, prefix='test_')

    # Combine dictionaries
    score_metrics = valid_score_metrics | test_score_metrics

    return score_metrics

def compare_models(models_dict, X_train, y_train, X_valid, y_valid, X_test, y_test):
    '''
        Compares multiple models by training and evaluating them.
        models_dict: Dictionary of model names and their corresponding model instances and parameters.
        returns a DataFrame with evaluation metrics for each model

        model_dict format:
        {
            'model_name': (model_instance, model_params),
        }
    '''
    import pandas as pd

    models_metrics = pd.DataFrame()
    for i, (model_name, values) in enumerate(models_dict.items()):
        print(f"    -- Training and evaluating {model_name}...")
        model, model_params = values['model_instance'], values['model_params']
        models_metrics.loc[i, 'model'] = model_name
        models_metrics.loc[i, 'model_params'] = str(model_params) if model_params else None

        # Fit and evaluate the model
        scores = fit_evaluate_model(model, model_params, 
                                           X_train, y_train, 
                                           X_valid, y_valid, 
                                           X_test, y_test)
        
        # Adding model metrics
        for key, values in scores.items():
            models_metrics.loc[i, key] = values

    return models_metrics

if __name__ == "__main__":
    print("This module is not meant to be run directly. Import it in your main script.")