import sys
import os
import time

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import get_forecast
from core.cache import get_model_path

def test_phase2():
    ticker = "AAPL"
    model_path = get_model_path(ticker)
    
    # 1. Clear existing cache if present to force a clean training run
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Cleared existing cached model for {ticker}.")

    print(f"\n=== Running First Call (Training Run) for {ticker} ===")
    start_time = time.time()
    forecast_df_1 = get_forecast(ticker, days_to_predict=5)
    training_time = time.time() - start_time
    print(f"✓ Training Run complete. Took {training_time:.2f} seconds.")
    
    # 2. Run second time to verify cache speed
    print(f"\n=== Running Second Call (Cached Run) for {ticker} ===")
    start_time = time.time()
    forecast_df_2 = get_forecast(ticker, days_to_predict=5)
    cached_time = time.time() - start_time
    print(f"✓ Cached Run complete. Took {cached_time:.4f} seconds.")
    
    # 3. Print verification and results
    assert len(forecast_df_1) == 5, "Forecast row count mismatch"
    assert os.path.exists(model_path), "Model cache file was not created"
    
    speed_improvement = (training_time / cached_time) if cached_time > 0 else 0
    print(f"\n✓ Speed Improvement: Cache was {speed_improvement:.1f}x faster!")
    
    print("\nForecast Sample Output (Prophet vs XGBoost vs Hybrid):")
    print(forecast_df_1.to_string(index=False))
    
    print("\nPhase 2 verification PASSED successfully!")

if __name__ == "__main__":
    test_phase2()
