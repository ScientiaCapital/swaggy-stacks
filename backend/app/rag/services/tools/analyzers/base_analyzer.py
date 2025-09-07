"""
Base analyzer class for pattern recognition modules
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class BaseAnalyzer(ABC):
    """Base class for all pattern analyzers"""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def analyze(self, data: List[Dict], lookback_period: int, min_strength: float, **kwargs) -> Dict[str, Any]:
        """Analyze patterns in the given data"""
        pass
    
    def _prepare_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Convert data list to pandas DataFrame with proper types"""
        try:
            df = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_cols = ['close']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in data")
            
            # Convert to numeric
            numeric_cols = ['close', 'high', 'low', 'open', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to prepare data: {str(e)}")
    
    def _find_swing_points(self, data: pd.Series, lookback: int = 5) -> Dict[str, List]:
        """Find swing highs and lows in price data"""
        highs = []
        lows = []
        
        for i in range(lookback, len(data) - lookback):
            # Check for swing high
            if all(data.iloc[i] >= data.iloc[i-j] for j in range(1, lookback + 1)) and \
               all(data.iloc[i] >= data.iloc[i+j] for j in range(1, lookback + 1)):
                highs.append({'index': i, 'price': data.iloc[i]})
            
            # Check for swing low
            if all(data.iloc[i] <= data.iloc[i-j] for j in range(1, lookback + 1)) and \
               all(data.iloc[i] <= data.iloc[i+j] for j in range(1, lookback + 1)):
                lows.append({'index': i, 'price': data.iloc[i]})
        
        return {'highs': highs, 'lows': lows}
    
    def _calculate_trend(self, prices: pd.Series, period: int = 20) -> str:
        """Determine overall trend direction"""
        if len(prices) < period:
            return "unknown"
        
        recent_prices = prices.tail(period)
        slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if slope > 0.001:
            return "bullish"
        elif slope < -0.001:
            return "bearish"
        else:
            return "sideways"
    
    def _validate_data(self, data: List[Dict], min_length: int = 10) -> bool:
        """Validate input data meets minimum requirements"""
        if not data or len(data) < min_length:
            return False
        
        # Check for required fields
        required_fields = ['close']
        for item in data[:3]:  # Check first few items
            if not all(field in item for field in required_fields):
                return False
        
        return True