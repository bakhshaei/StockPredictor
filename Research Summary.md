# Research Summary: ML Approaches for Stock Index Price Prediction

## Traditional Time Series Models
- **Moving Average (MA)** techniques: Simple MA and Exponential MA for smoothing short-term fluctuations
- **ARIMA (AutoRegressive Integrated Moving Average)**: Combines autoregression, differencing, and moving average components
- **Support Vector Machines (SVM)**: Effective for classification and regression in stock prediction
- **Random Forest**: Tree-based ensemble technique for predicting stock prices

## Deep Learning Approaches
- **LSTM (Long Short-Term Memory)**: Most effective for stock prediction with reported 93% accuracy
  - Capable of handling longer input sequences compared to other RNNs
  - Ability to distinguish between short-term and long-term factors
  - Implemented using libraries like Keras, TensorFlow, and Sklearn
- **RNN (Recurrent Neural Networks)**: Base architecture for sequence data but suffers from vanishing gradient problem
- **CNN (Convolutional Neural Networks)**: Used in combination with other models for feature extraction
- **CNN-sliding window model**: Combined with LSTM and RNN for improved performance

## Ensemble Methods
- **Fusion of machine learning techniques**: Combining multiple models for improved prediction accuracy
- **Deep learning for event-driven stock prediction**: Incorporating news and events data
- **Hybrid models**: Combining technical analysis with deep learning approaches

## Feature Selection Techniques
- **Technical Analysis Indicators**:
  - Simple Moving Average (SMA)
  - Convergence Divergence Moving Average (MACD)
  - Relative Strength Index (RSI)
- **Price data features**: Opening price, closing price, highest price, lowest price, trading volume
- **Data preparation steps**:
  1. Check and handle data defects (empty data, deviations)
  2. Calculate technical indicators
  3. Aggregate historical price data with technical indicators
  4. Split data into training and testing sets (typically 80/20 split)

## Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Measures differences between predicted and true values
- **MAPE (Mean Absolute Percentage Error)**: Measures difference relative to true values
