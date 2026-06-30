import sys
import os

# Add the project root to path so we can import from core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data import fetch_stock_data, add_technical_features

def test_pipeline():
    ticker = "AAPL"
    print(f"Testing Phase 1 with ticker: {ticker}...")
    
    try:
        # 1. Fetch raw data
        raw_df = fetch_stock_data(ticker, period="1mo")
        print(f"✓ Data download successful. Rows fetched: {len(raw_df)}")
        print("Raw Data Sample:")
        print(raw_df.head(3))
        
        # 2. Add features
        feature_df = add_technical_features(raw_df)
        print(f"\n✓ Technical features added. Rows after dropping NaNs: {len(feature_df)}")
        print("Feature Data Sample (showing engineered columns):")
        print(feature_df[['ds', 'y', 'sma_5', 'sma_20', 'returns', 'lag_1']].head(3))
        
        print("\nPhase 1 verification PASSED successfully!")
        
    except Exception as e:
        print(f"\n✗ Phase 1 verification FAILED: {str(e)}")

if __name__ == "__main__":
    test_pipeline()
