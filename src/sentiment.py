import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import warnings

warnings.filterwarnings('ignore')
nltk.download('vader_lexicon', quiet=True)

def build_macro_sentiment_signal(input_csv='../STOCK-MARKET-NEWS-PREDICTION/data/raw/sp500_headlines_2008_2024.csv', 
                                 output_csv='../STOCK-MARKET-NEWS-PREDICTION/data/processed/macro_sentiment.csv',start_date = '2020-01-31',end_date = '2024-12-30'):
    
    print(f"STARTING MACRO NLP PROCESSING ({start_date} to {end_date})")

# 1. Load the Raw S&P 500 News Dataset
    try:
        print("Loading raw macro news database...")
        news_df = pd.read_csv(input_csv, usecols=['Date', 'Title'])
    except FileNotFoundError:
        print(f"ERROR: Could not find {input_csv}.")
        return

    # Standardize column names
    news_df = news_df.rename(columns={'date': 'Date', 'title': 'Title'})
    
    # Clean the Dates
    news_df['Date'] = pd.to_datetime(news_df['Date'], errors='coerce').dt.tz_localize(None).dt.normalize()
    news_df = news_df.dropna(subset=['Date', 'Title'])

    
    # 2. ENFORCE TEMPORAL BOUNDS (THE FIX)
    
    initial_rows = len(news_df)
    
    # Create a boolean mask to slice only the exact 5-year window we need
    mask = (news_df['Date'] >= start_date) & (news_df['Date'] <= end_date)
    news_df = news_df.loc[mask]
    
    print(f"Filtered dataset from {initial_rows} down to {len(news_df)} headlines for our time window.")

    # 3. Run the NLP Engine
    print("Initializing VADER NLP Engine...")
    sia = SentimentIntensityAnalyzer()
    
    print("Scoring headlines... (This will be much faster now)")
    news_df['Macro_Sentiment'] = news_df['Title'].apply(
        lambda text: sia.polarity_scores(str(text))['compound']
    )

    # 4. Aggregate into a Daily Macro Indicator
    print("Aggregating scores into a daily macro time-series...")
    daily_macro = news_df.groupby('Date').agg(
        Macro_Sentiment=('Macro_Sentiment', 'mean'),
        Macro_News_Volume=('Title', 'count') 
    ).reset_index()

    # Create a 3-day rolling average to smooth out the noise
    daily_macro['Macro_Sentiment_3d'] = daily_macro['Macro_Sentiment'].rolling(3).mean()
    
    # Drop the NaN rows created by the 3-day rolling average at the very beginning of 2020
    daily_macro = daily_macro.dropna()

    # 5. Save the Processed Signal
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    daily_macro.to_csv(output_csv, index=False)
    
    print(f"SUCCESS: Generated Macro Market Sentiment for {len(daily_macro)} trading days.")


if __name__ == "__main__":
    build_macro_sentiment_signal()