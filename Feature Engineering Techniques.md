# Feature Engineering Techniques for Stock Index Price Prediction

Feature engineering is a critical component in developing effective machine learning models for stock index price prediction. This document details specific techniques to transform raw financial data into meaningful features that can improve the predictive power of our LSTM-based model.

## 1. Technical Indicator Engineering

### Trend Indicators
- **Moving Averages Crossovers**: 
  - Create binary signals when short-term MA crosses long-term MA
  - Example: Generate a feature that equals 1 when 5-day SMA crosses above 20-day SMA, -1 when it crosses below, and 0 otherwise
  ```python
  def ma_crossover_signal(data, short_window=5, long_window=20):
      signals = pd.DataFrame(index=data.index)
      signals['signal'] = 0.0
      
      # Create short and long moving averages
      signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
      signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
      
      # Create signals
      signals['signal'][short_window:] = np.where(
          signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0)
      
      # Generate trading orders
      signals['positions'] = signals['signal'].diff()
      
      return signals
  ```

- **MACD Derived Features**:
  - MACD Histogram: Difference between MACD line and signal line
  - MACD Divergence: Difference between price trend and MACD trend
  - MACD Slope: Rate of change of MACD line

### Volatility Features
- **Bollinger Band Width**: Measure of volatility
  ```python
  def bollinger_band_width(data, window=20, num_std=2):
      rolling_mean = data['Close'].rolling(window=window).mean()
      rolling_std = data['Close'].rolling(window=window).std()
      
      upper_band = rolling_mean + (rolling_std * num_std)
      lower_band = rolling_mean - (rolling_std * num_std)
      
      # Calculate bandwidth
      bandwidth = (upper_band - lower_band) / rolling_mean
      return bandwidth
  ```

- **Volatility Ratio**: Ratio of short-term to long-term volatility
  ```python
  def volatility_ratio(data, short_window=5, long_window=20):
      short_vol = data['Close'].rolling(window=short_window).std()
      long_vol = data['Close'].rolling(window=long_window).std()
      
      vol_ratio = short_vol / long_vol
      return vol_ratio
  ```

### Momentum Features
- **RSI Extremes**: Binary indicators for overbought/oversold conditions
  ```python
  def rsi_signals(data, window=14, overbought=70, oversold=30):
      delta = data['Close'].diff()
      gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
      
      rs = gain / loss
      rsi = 100 - (100 / (1 + rs))
      
      # Create signals
      overbought_signal = np.where(rsi > overbought, 1, 0)
      oversold_signal = np.where(rsi < oversold, 1, 0)
      
      return pd.DataFrame({
          'RSI': rsi,
          'Overbought': overbought_signal,
          'Oversold': oversold_signal
      }, index=data.index)
  ```

- **Stochastic Oscillator Crossovers**: Signals when %K crosses %D
  ```python
  def stochastic_crossover(data, k_period=14, d_period=3):
      # Calculate %K
      low_min = data['Low'].rolling(window=k_period).min()
      high_max = data['High'].rolling(window=k_period).max()
      
      k = 100 * ((data['Close'] - low_min) / (high_max - low_min))
      
      # Calculate %D
      d = k.rolling(window=d_period).mean()
      
      # Create crossover signal
      crossover = np.where(k > d, 1, np.where(k < d, -1, 0))
      
      return pd.DataFrame({
          '%K': k,
          '%D': d,
          'Crossover': crossover
      }, index=data.index)
  ```

## 2. Price Pattern Features

### Candlestick Patterns
- **Doji**: Days with small body (open and close are nearly equal)
  ```python
  def detect_doji(data, threshold=0.1):
      body_size = abs(data['Close'] - data['Open'])
      avg_body = body_size.rolling(window=20).mean()
      
      is_doji = body_size < (threshold * avg_body)
      return is_doji
  ```

- **Hammer/Hanging Man**: Long lower shadow, small body at the top
- **Engulfing Patterns**: Current candle completely engulfs previous candle

### Support and Resistance
- **Pivot Points**: Identify potential support/resistance levels
  ```python
  def pivot_points(data):
      pivot = (data['High'] + data['Low'] + data['Close']) / 3
      
      support1 = (2 * pivot) - data['High']
      support2 = pivot - (data['High'] - data['Low'])
      
      resistance1 = (2 * pivot) - data['Low']
      resistance2 = pivot + (data['High'] - data['Low'])
      
      return pd.DataFrame({
          'Pivot': pivot,
          'S1': support1,
          'S2': support2,
          'R1': resistance1,
          'R2': resistance2
      }, index=data.index)
  ```

- **Price Distance from Support/Resistance**: Normalized distance from identified levels

## 3. Time-Based Features

### Temporal Decomposition
- **Day of Week**: Encode trading day (Monday through Friday)
  ```python
  def add_time_features(data):
      # Ensure index is datetime
      if not isinstance(data.index, pd.DatetimeIndex):
          data.index = pd.to_datetime(data.index)
      
      # Extract time features
      data['day_of_week'] = data.index.dayofweek
      data['month'] = data.index.month
      data['quarter'] = data.index.quarter
      
      # One-hot encode if needed
      data = pd.get_dummies(data, columns=['day_of_week', 'month', 'quarter'], prefix=['dow', 'm', 'q'])
      
      return data
  ```

- **Month/Quarter Effects**: Capture seasonal patterns in market behavior
- **Pre/Post Holiday Effects**: Binary indicators for trading days before/after holidays

### Cyclical Encoding
- **Sine/Cosine Transformation**: For cyclical features like day of week, month
  ```python
  def cyclical_encoding(data, col, period):
      data[f'{col}_sin'] = np.sin(2 * np.pi * data[col] / period)
      data[f'{col}_cos'] = np.cos(2 * np.pi * data[col] / period)
      return data
  ```

## 4. Volume-Based Features

### Volume Indicators
- **Volume Relative to Moving Average**: Ratio of current volume to n-day average
  ```python
  def relative_volume(data, window=20):
      avg_volume = data['Volume'].rolling(window=window).mean()
      rel_vol = data['Volume'] / avg_volume
      return rel_vol
  ```

- **On-Balance Volume (OBV) Momentum**: Rate of change of OBV
  ```python
  def obv_momentum(data, obv_window=1, momentum_window=5):
      # Calculate OBV
      obv = np.zeros(len(data))
      for i in range(1, len(data)):
          if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
              obv[i] = obv[i-1] + data['Volume'].iloc[i]
          elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
              obv[i] = obv[i-1] - data['Volume'].iloc[i]
          else:
              obv[i] = obv[i-1]
      
      # Convert to Series
      obv_series = pd.Series(obv, index=data.index)
      
      # Calculate momentum
      obv_momentum = obv_series.diff(obv_window).rolling(window=momentum_window).mean()
      
      return obv_momentum
  ```

- **Volume Price Trend (VPT)**: Cumulative volume that considers price changes

## 5. Derivative Features

### Statistical Derivatives
- **Return Acceleration**: Second derivative of price (rate of change of returns)
  ```python
  def return_acceleration(data, window=1):
      # First derivative: returns
      returns = data['Close'].pct_change(window)
      
      # Second derivative: acceleration
      acceleration = returns.diff(window)
      
      return acceleration
  ```

- **Volatility of Volatility**: Standard deviation of rolling standard deviation
  ```python
  def vol_of_vol(data, vol_window=20, vov_window=20):
      # Calculate volatility
      volatility = data['Close'].pct_change().rolling(window=vol_window).std()
      
      # Calculate volatility of volatility
      vol_of_vol = volatility.rolling(window=vov_window).std()
      
      return vol_of_vol
  ```

### Ratio Features
- **Price to Moving Average Ratio**: Normalized price relative to its trend
  ```python
  def price_to_ma_ratio(data, window=20):
      ma = data['Close'].rolling(window=window).mean()
      ratio = data['Close'] / ma
      return ratio
  ```

- **High-Low Range Ratio**: Daily trading range relative to historical range

## 6. Advanced Feature Engineering

### Fourier Transforms
- **Fast Fourier Transform (FFT)**: Decompose time series into frequency components
  ```python
  def fft_features(data, num_components=3):
      # Get the close prices
      close_prices = data['Close'].values
      
      # Apply FFT
      fft_result = np.fft.fft(close_prices)
      
      # Get the power spectrum
      power = np.abs(fft_result) ** 2
      
      # Get indices of top frequency components
      top_indices = np.argsort(power)[::-1][:num_components+1][1:]  # Skip the DC component
      
      # Create features for each component
      fft_features = pd.DataFrame(index=data.index)
      
      for i, idx in enumerate(top_indices):
          # Extract amplitude and phase
          amplitude = np.abs(fft_result[idx])
          phase = np.angle(fft_result[idx])
          
          # Create sinusoidal feature
          period = len(close_prices) / idx
          time_idx = np.arange(len(close_prices))
          
          fft_features[f'fft_comp_{i+1}'] = amplitude * np.sin(2 * np.pi * time_idx / period + phase)
      
      return fft_features
  ```

### Wavelet Transforms
- **Discrete Wavelet Transform (DWT)**: Multi-resolution analysis of time series
  ```python
  def wavelet_features(data, wavelet='db1', level=3):
      import pywt
      
      # Get the close prices
      close_prices = data['Close'].values
      
      # Apply wavelet transform
      coeffs = pywt.wavedec(close_prices, wavelet, level=level)
      
      # Create features from coefficients
      wavelet_features = pd.DataFrame(index=data.index)
      
      for i, coeff in enumerate(coeffs):
          # Pad or truncate to match length
          if len(coeff) < len(close_prices):
              coeff = np.pad(coeff, (0, len(close_prices) - len(coeff)), 'constant')
          elif len(coeff) > len(close_prices):
              coeff = coeff[:len(close_prices)]
          
          wavelet_features[f'wavelet_level_{i}'] = coeff
      
      return wavelet_features
  ```

### Feature Interactions
- **Multiplicative Interactions**: Products of important features
  ```python
  def create_interaction_features(data, features):
      interactions = pd.DataFrame(index=data.index)
      
      for i in range(len(features)):
          for j in range(i+1, len(features)):
              feat1 = features[i]
              feat2 = features[j]
              interactions[f'{feat1}_{feat2}_interaction'] = data[feat1] * data[feat2]
      
      return interactions
  ```

- **Ratio Interactions**: Ratios between related features

## 7. Feature Selection and Dimensionality Reduction

### Correlation-Based Selection
- **Feature Correlation Matrix**: Identify and remove highly correlated features
  ```python
  def correlation_selection(data, threshold=0.8):
      # Calculate correlation matrix
      corr_matrix = data.corr().abs()
      
      # Create upper triangle mask
      upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
      
      # Find features with correlation greater than threshold
      to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
      
      # Drop highly correlated features
      reduced_data = data.drop(columns=to_drop)
      
      return reduced_data, to_drop
  ```

### Feature Importance
- **Random Forest Feature Importance**: Use tree-based models to rank features
  ```python
  def rf_feature_importance(X, y, n_estimators=100):
      from sklearn.ensemble import RandomForestRegressor
      
      # Train a Random Forest model
      rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
      rf.fit(X, y)
      
      # Get feature importances
      importances = rf.feature_importances_
      
      # Create a DataFrame with feature names and importances
      feature_importance = pd.DataFrame({
          'Feature': X.columns,
          'Importance': importances
      }).sort_values('Importance', ascending=False)
      
      return feature_importance
  ```

- **Recursive Feature Elimination (RFE)**: Iteratively remove least important features

These feature engineering techniques transform raw financial data into a rich set of features that capture various aspects of market behavior. By implementing these techniques, we can significantly improve the predictive power of our LSTM model for stock index price prediction.
