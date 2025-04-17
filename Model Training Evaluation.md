# Model Training and Evaluation for Stock Index Price Prediction

This document outlines the comprehensive approach to training and evaluating our LSTM-based model for stock index price prediction. It covers training methodologies, hyperparameter tuning, evaluation metrics, and validation strategies to ensure robust and reliable predictions.

## 1. Model Training Methodology

### Batch Training Process

```python
def train_lstm_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    # Define model architecture
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history
```

### Loss Functions

1. **Mean Squared Error (MSE)**: Standard loss function for regression problems
   ```python
   model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
   ```

2. **Mean Absolute Error (MAE)**: Less sensitive to outliers
   ```python
   model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_error')
   ```

3. **Huber Loss**: Combines benefits of MSE and MAE
   ```python
   model.compile(optimizer=Adam(learning_rate=0.001), loss=tf.keras.losses.Huber(delta=1.0))
   ```

4. **Custom Directional Loss**: Penalizes incorrect direction predictions more heavily
   ```python
   def directional_loss(y_true, y_pred):
       # Calculate returns
       y_true_returns = (y_true[1:] - y_true[:-1]) / y_true[:-1]
       y_pred_returns = (y_pred[1:] - y_pred[:-1]) / y_pred[:-1]
       
       # Calculate direction match
       direction_match = tf.sign(y_true_returns) * tf.sign(y_pred_returns)
       
       # Calculate MSE
       mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
       
       # Penalize incorrect directions more heavily
       direction_penalty = tf.maximum(0.0, -direction_match)
       
       return mse + 0.5 * direction_penalty
   
   model.compile(optimizer=Adam(learning_rate=0.001), loss=directional_loss)
   ```

### Regularization Techniques

1. **Dropout**: Prevents overfitting by randomly deactivating neurons
   ```python
   model.add(Dropout(0.2))  # 20% dropout rate
   ```

2. **L1/L2 Regularization**: Adds penalty for large weights
   ```python
   from tensorflow.keras.regularizers import l1, l2, l1_l2
   
   model.add(LSTM(128, return_sequences=True, 
                 kernel_regularizer=l2(0.01),
                 recurrent_regularizer=l2(0.01),
                 bias_regularizer=l2(0.01)))
   ```

3. **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving
   ```python
   early_stopping = EarlyStopping(
       monitor='val_loss',
       patience=10,
       restore_best_weights=True
   )
   ```

## 2. Hyperparameter Tuning

### Grid Search

```python
def grid_search_lstm(X_train, y_train, X_val, y_val):
    # Define hyperparameter grid
    param_grid = {
        'lstm_units_1': [64, 128, 256],
        'lstm_units_2': [32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64]
    }
    
    best_val_loss = float('inf')
    best_params = None
    
    # Perform grid search
    for lstm_units_1 in param_grid['lstm_units_1']:
        for lstm_units_2 in param_grid['lstm_units_2']:
            for dropout_rate in param_grid['dropout_rate']:
                for learning_rate in param_grid['learning_rate']:
                    for batch_size in param_grid['batch_size']:
                        # Define model
                        model = Sequential()
                        model.add(LSTM(lstm_units_1, return_sequences=True, 
                                      input_shape=(X_train.shape[1], X_train.shape[2])))
                        model.add(Dropout(dropout_rate))
                        model.add(LSTM(lstm_units_2, return_sequences=False))
                        model.add(Dropout(dropout_rate))
                        model.add(Dense(32, activation='relu'))
                        model.add(Dense(1))
                        
                        # Compile model
                        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                                     loss='mean_squared_error')
                        
                        # Train model
                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, 
                                                      restore_best_weights=True)
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=50,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping],
                            verbose=0
                        )
                        
                        # Evaluate model
                        val_loss = min(history.history['val_loss'])
                        
                        # Update best parameters
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_params = {
                                'lstm_units_1': lstm_units_1,
                                'lstm_units_2': lstm_units_2,
                                'dropout_rate': dropout_rate,
                                'learning_rate': learning_rate,
                                'batch_size': batch_size
                            }
    
    return best_params, best_val_loss
```

### Bayesian Optimization

```python
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

def bayesian_optimization_lstm(X_train, y_train, X_val, y_val):
    # Define the search space
    dim_lstm_units_1 = Integer(32, 256, name='lstm_units_1')
    dim_lstm_units_2 = Integer(16, 128, name='lstm_units_2')
    dim_dropout_rate = Real(0.1, 0.5, name='dropout_rate')
    dim_learning_rate = Real(0.0001, 0.01, "log-uniform", name='learning_rate')
    dim_batch_size = Integer(8, 64, name='batch_size')
    
    dimensions = [dim_lstm_units_1, dim_lstm_units_2, dim_dropout_rate, 
                 dim_learning_rate, dim_batch_size]
    
    # Define the objective function
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        # Build and train model with given parameters
        model = Sequential()
        model.add(LSTM(params['lstm_units_1'], return_sequences=True, 
                      input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(params['dropout_rate']))
        model.add(LSTM(params['lstm_units_2'], return_sequences=False))
        model.add(Dropout(params['dropout_rate']))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), 
                     loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, 
                                      restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=params['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Return the validation loss (to be minimized)
        return min(history.history['val_loss'])
    
    # Run Bayesian optimization
    result = gp_minimize(objective, dimensions, n_calls=50, random_state=42)
    
    # Get best parameters
    best_params = {
        'lstm_units_1': result.x[0],
        'lstm_units_2': result.x[1],
        'dropout_rate': result.x[2],
        'learning_rate': result.x[3],
        'batch_size': result.x[4]
    }
    
    return best_params, result.fun
```

## 3. Evaluation Metrics

### Regression Metrics

```python
def evaluate_model(model, X_test, y_test, scaler_y=None):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform if scaler was used
    if scaler_y is not None:
        y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inv = scaler_y.inverse_transform(y_pred).flatten()
    else:
        y_test_inv = y_test
        y_pred_inv = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    # Calculate MAPE
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    
    # Calculate R-squared
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    # Calculate directional accuracy
    y_test_direction = np.sign(np.diff(y_test_inv))
    y_pred_direction = np.sign(np.diff(y_pred_inv))
    directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }
```

### Trading Strategy Metrics

```python
def evaluate_trading_strategy(y_test, y_pred, initial_capital=10000):
    # Calculate returns
    y_test_returns = np.diff(y_test) / y_test[:-1]
    
    # Generate trading signals (1 for buy, -1 for sell, 0 for hold)
    signals = np.sign(np.diff(y_pred.flatten()))
    
    # Calculate strategy returns (position * market return)
    strategy_returns = signals[:-1] * y_test_returns[1:]
    
    # Calculate cumulative returns
    cumulative_market_returns = np.cumprod(1 + y_test_returns) - 1
    cumulative_strategy_returns = np.cumprod(1 + strategy_returns) - 1
    
    # Calculate final portfolio value
    final_market_value = initial_capital * (1 + cumulative_market_returns[-1])
    final_strategy_value = initial_capital * (1 + cumulative_strategy_returns[-1])
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)  # Annualized
    
    # Calculate maximum drawdown
    peak = np.maximum.accumulate(cumulative_strategy_returns)
    drawdown = (peak - cumulative_strategy_returns) / (peak + 1e-10)
    max_drawdown = np.max(drawdown)
    
    # Calculate win rate
    win_rate = np.sum(strategy_returns > 0) / len(strategy_returns) * 100
    
    return {
        'Final_Market_Value': final_market_value,
        'Final_Strategy_Value': final_strategy_value,
        'Strategy_Return': (final_strategy_value / initial_capital - 1) * 100,
        'Market_Return': (final_market_value / initial_capital - 1) * 100,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown * 100,
        'Win_Rate': win_rate
    }
```

## 4. Validation Strategies

### Walk-Forward Validation

```python
def walk_forward_validation(data, feature_columns, target_column, sequence_length=50, 
                           train_size=252*2, test_size=63, step=21):
    """
    Implements walk-forward validation for time series forecasting.
    
    Args:
        data: DataFrame containing features and target
        feature_columns: List of feature column names
        target_column: Target column name
        sequence_length: Length of input sequences
        train_size: Size of initial training set (e.g., 2 years of daily data)
        test_size: Size of test set for each iteration (e.g., 3 months)
        step: Number of steps to move forward after each iteration (e.g., 1 month)
    
    Returns:
        List of evaluation metrics for each iteration
    """
    results = []
    
    # Prepare data
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Initialize scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Walk forward through the dataset
    for i in range(0, len(data) - train_size - test_size + 1, step):
        # Define train/test indices for this iteration
        train_start = i
        train_end = i + train_size
        test_start = train_end
        test_end = test_start + test_size
        
        # Extract train/test data
        X_train_raw = X[train_start:train_end]
        y_train_raw = y[train_start:train_end]
        X_test_raw = X[test_start:test_end]
        y_test_raw = y[test_start:test_end]
        
        # Scale data
        X_train_scaled = scaler_X.fit_transform(X_train_raw)
        y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        X_test_scaled = scaler_X.transform(X_test_raw)
        y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
        X_test, y_test = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train_scaled.shape[1])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test_scaled.shape[1])
        
        # Train model
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Split training data for validation
        train_size_val = int(0.8 * len(X_train))
        X_train_final, X_val = X_train[:train_size_val], X_train[train_size_val:]
        y_train_final, y_val = y_train[:train_size_val], y_train[train_size_val:]
        
        model.fit(
            X_train_final, y_train_final,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, scaler_y)
        
        # Add period information
        metrics['Train_Start'] = data.index[train_start]
        metrics['Train_End'] = data.index[train_end-1]
        metrics['Test_Start'] = data.index[test_start]
        metrics['Test_End'] = data.index[min(test_end-1, len(data)-1)]
        
        results.append(metrics)
    
    return results
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(X, y, n_splits=5):
    """
    Implements time series cross-validation.
    
    Args:
        X: Input features (3D array for LSTM: [samples, time steps, features])
        y: Target values
        n_splits: Number of splits for cross-validation
    
    Returns:
        Mean and std of evaluation metrics across folds
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for train_index, test_index in tsc
(Content truncated due to size limit. Use line ranges to read in chunks)
