import pandas as pd 
import os 

def clean_raw_data(input_path = '../Stock-MARKET-NEWS-PREDICTION/data/raw/raw_panel_data.csv' , output_path = '../Stock-MARKET-NEWS-PREDICTION/data/processed/cleaned_panel_data.csv'):
    print("Loading data for preprocessing")

    #Read raw data from csv 
    df  = pd.read_csv(input_path)

    #1 Normalize Dates 
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.normalize()
    
    #2 Enforce Sorting
    df = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # 3. Handle Gaps with Grouped Forward-Fill
    price_columns = ['Open', 'High', 'Low', 'Close']
    df[price_columns] = df.groupby('Ticker')[price_columns].ffill()
    
    if 'Volume' in df.columns:
        df['Volume'] = df.groupby('Ticker')['Volume'].transform(lambda x: x.fillna(0))
        
    df = df.dropna(subset=['Close'])
    
    # Ensure the target directory exists
    os.makedirs('../data/processed', exist_ok=True)
    
    # Save the cleaned data ready for feature engineering
    df.to_csv(output_path, index=False)
    print(f"Cleaned panel data saved to {output_path}")

if __name__ == "__main__":
    clean_raw_data()

    