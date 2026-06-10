import pandas as pd
import numpy as np
import os

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    RSI is a bounded momentum indicator between 0 and 100.
    identify overbought (RSI > 70) and oversold (RSI < 30) conditions
    """
    delta = series.diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def engineer_features(input_path='../STOCK-MARKET-NEWS-PREDICTION/data/processed/cleaned_panel_data.csv', 
                      output_path='../STOCK-MARKET-NEWS-PREDICTION/data/processed/features_panel_data.csv'):
    
   
    print("STARTING PHASE 3: Feature Engineering...")
    
    # 1. Load Data
    print("Loading cleaned panel data...")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # CRITICAL: Re-enforce sorting. If rows got shuffled, rolling math will be corrupted.
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # 2. Base Returns
    print("Calculating logarithmic returns...")
    df['Log_Return'] = np.log(df['Close'] / df.groupby('Ticker')['Close'].shift(1))

    # 3. Institutional Target Formulation (Cross-Sectional Rank)
    print("Formulating Cross-Sectional Rank Target...")
    df['Target_Fwd_Return'] = df.groupby('Ticker')['Log_Return'].shift(-1)
    
    # Rank the forward returns across all stocks on a specific date, scaled -0.5 to +0.5
    df['Target'] = df.groupby('Date')['Target_Fwd_Return'].transform(lambda x: x.rank(pct=True)) - 0.5

    # 4. Alpha Features (The Predictors)
    print("Engineering Alpha Signals (Momentum, Volatility, Mean-Reversion)...")
    
    # FEATURE 1: Price to Simple Moving Average (Trend/Momentum)
    # Are we trading above or below the 20-day trend?
    df['SMA_20'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    df['Price_to_SMA_20'] = (df['Close'] / df['SMA_20']) - 1 
    
    # FEATURE 2: Rolling Volatility (Risk)
    # ML models need to know if the asset is currently calm or highly erratic
    df['Volatility_14d'] = df.groupby('Ticker')['Log_Return'].transform(lambda x: x.rolling(14).std())
    
    # FEATURE 3: Relative Strength Index (Mean Reversion)
    # Is the stock overbought (>70) or oversold (<30)?
    df['RSI_14'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi(x, 14))
    
    # FEATURE 4: Autocorrelation Lags (Historical Memory)
    # We feed the model the last 3 days of returns so it can detect immediate micro-trends
    df['Return_Lag1'] = df.groupby('Ticker')['Log_Return'].shift(1)
    df['Return_Lag2'] = df.groupby('Ticker')['Log_Return'].shift(2)
    df['Return_Lag3'] = df.groupby('Ticker')['Log_Return'].shift(3)

    # 5. Handle the "Burn-In" Period
    print("\nHandling 'Burn-in' missing values...")
    # A 20-day SMA requires 20 days of history. The first 19 days of every stock will be NaN.
    # The Target requires tomorrow's price. The last day of every stock will be NaN.
    # We must drop these incomplete rows before feeding them to Scikit-Learn.
    initial_rows = len(df)
    
    # Drop intermediate columns we don't need the ML model to see
    df = df.drop(columns=['SMA_20', 'Target_Fwd_Return']) 
    
    df = df.dropna()
    final_rows = len(df)
    
    print(f"Dropped {initial_rows - final_rows} incomplete rows.")
    print(f"Final ML Matrix Size: {final_rows} rows.")
    
    # 6. Save the final dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"SUCCESS: Feature matrix saved to {output_path}")
    print("==================================================")

if __name__ == "__main__":
    engineer_features()