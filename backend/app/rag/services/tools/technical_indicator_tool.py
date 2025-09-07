"""
Technical Indicator Tool for calculating trading indicators
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from .base_tool import AgentTool, ToolResult, ToolParameter

logger = logging.getLogger(__name__)


class TechnicalIndicatorTool(AgentTool):
    """Tool for calculating technical indicators"""
    
    def __init__(self):
        super().__init__(
            name="technical_indicators",
            description="Calculate technical indicators like RSI, MACD, Bollinger Bands, etc."
        )
        self.category = "technical_analysis"
    
    def get_parameters(self) -> List[ToolParameter]:
        """Get tool parameter definitions"""
        return [
            ToolParameter(
                name="indicator",
                type="str",
                description="Indicator to calculate: 'rsi', 'macd', 'bollinger', 'sma', 'ema'",
                required=True
            ),
            ToolParameter(
                name="data",
                type="list",
                description="Price data as list of dictionaries with 'close', 'high', 'low' values",
                required=True
            ),
            ToolParameter(
                name="period",
                type="int",
                description="Period for calculation (e.g., 14 for RSI, 20 for Bollinger)",
                required=False,
                default=14
            ),
            ToolParameter(
                name="fast_period",
                type="int",
                description="Fast period for MACD",
                required=False,
                default=12
            ),
            ToolParameter(
                name="slow_period",
                type="int",
                description="Slow period for MACD",
                required=False,
                default=26
            ),
            ToolParameter(
                name="signal_period",
                type="int",
                description="Signal line period for MACD",
                required=False,
                default=9
            ),
            ToolParameter(
                name="std_dev",
                type="float",
                description="Standard deviations for Bollinger Bands",
                required=False,
                default=2.0
            )
        ]
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute technical indicator calculation"""
        try:
            indicator = parameters["indicator"].lower()
            data = parameters["data"]
            
            # Convert data to pandas DataFrame
            df = self._prepare_data(data)
            
            if indicator == "rsi":
                return await self._calculate_rsi(df, parameters)
            elif indicator == "macd":
                return await self._calculate_macd(df, parameters)
            elif indicator == "bollinger":
                return await self._calculate_bollinger_bands(df, parameters)
            elif indicator == "sma":
                return await self._calculate_sma(df, parameters)
            elif indicator == "ema":
                return await self._calculate_ema(df, parameters)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown indicator: {indicator}. Supported: 'rsi', 'macd', 'bollinger', 'sma', 'ema'"
                )
                
        except Exception as e:
            logger.error(f"Technical indicator calculation error: {e}")
            return ToolResult(success=False, data=None, error=f"Calculation error: {str(e)}")
    
    def _prepare_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare data for calculation"""
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
            
            # Sort by timestamp if available
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to prepare data: {str(e)}")
    
    async def _calculate_rsi(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> ToolResult:
        """Calculate Relative Strength Index (RSI)"""
        try:
            period = parameters.get("period", 14)
            
            # Calculate price changes
            delta = df['close'].diff()
            
            # Calculate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gains = gains.rolling(window=period, min_periods=1).mean()
            avg_losses = losses.rolling(window=period, min_periods=1).mean()
            
            # Calculate RS and RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # Prepare result
            result_data = {
                "values": rsi.dropna().tolist(),
                "period": period,
                "latest_value": float(rsi.iloc[-1]) if not rsi.empty else None,
                "signal": self._interpret_rsi(float(rsi.iloc[-1]) if not rsi.empty else None)
            }
            
            metadata = {
                "indicator": "RSI",
                "period": period,
                "data_points": len(result_data["values"]),
                "calculated_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"RSI calculation failed: {str(e)}")
    
    async def _calculate_macd(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> ToolResult:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            fast_period = parameters.get("fast_period", 12)
            slow_period = parameters.get("slow_period", 26)
            signal_period = parameters.get("signal_period", 9)
            
            # Calculate EMAs
            ema_fast = df['close'].ewm(span=fast_period).mean()
            ema_slow = df['close'].ewm(span=slow_period).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Prepare result
            result_data = {
                "macd_line": macd_line.dropna().tolist(),
                "signal_line": signal_line.dropna().tolist(),
                "histogram": histogram.dropna().tolist(),
                "latest_macd": float(macd_line.iloc[-1]) if not macd_line.empty else None,
                "latest_signal": float(signal_line.iloc[-1]) if not signal_line.empty else None,
                "latest_histogram": float(histogram.iloc[-1]) if not histogram.empty else None,
                "signal": self._interpret_macd(
                    float(macd_line.iloc[-1]) if not macd_line.empty else None,
                    float(signal_line.iloc[-1]) if not signal_line.empty else None,
                    float(histogram.iloc[-1]) if not histogram.empty else None
                )
            }
            
            metadata = {
                "indicator": "MACD",
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "data_points": len(result_data["macd_line"]),
                "calculated_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"MACD calculation failed: {str(e)}")
    
    async def _calculate_bollinger_bands(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> ToolResult:
        """Calculate Bollinger Bands"""
        try:
            period = parameters.get("period", 20)
            std_dev = parameters.get("std_dev", 2.0)
            
            # Calculate moving average
            sma = df['close'].rolling(window=period).mean()
            
            # Calculate standard deviation
            std = df['close'].rolling(window=period).std()
            
            # Calculate bands
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Calculate position relative to bands
            latest_price = float(df['close'].iloc[-1]) if not df['close'].empty else None
            latest_upper = float(upper_band.iloc[-1]) if not upper_band.empty else None
            latest_lower = float(lower_band.iloc[-1]) if not lower_band.empty else None
            latest_middle = float(sma.iloc[-1]) if not sma.empty else None
            
            # Prepare result
            result_data = {
                "upper_band": upper_band.dropna().tolist(),
                "middle_band": sma.dropna().tolist(),
                "lower_band": lower_band.dropna().tolist(),
                "latest_upper": latest_upper,
                "latest_middle": latest_middle,
                "latest_lower": latest_lower,
                "latest_price": latest_price,
                "signal": self._interpret_bollinger(latest_price, latest_upper, latest_lower, latest_middle)
            }
            
            metadata = {
                "indicator": "Bollinger Bands",
                "period": period,
                "std_dev": std_dev,
                "data_points": len(result_data["upper_band"]),
                "calculated_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"Bollinger Bands calculation failed: {str(e)}")
    
    async def _calculate_sma(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> ToolResult:
        """Calculate Simple Moving Average"""
        try:
            period = parameters.get("period", 20)
            
            sma = df['close'].rolling(window=period).mean()
            
            result_data = {
                "values": sma.dropna().tolist(),
                "period": period,
                "latest_value": float(sma.iloc[-1]) if not sma.empty else None,
                "latest_price": float(df['close'].iloc[-1]) if not df['close'].empty else None
            }
            
            metadata = {
                "indicator": "SMA",
                "period": period,
                "data_points": len(result_data["values"]),
                "calculated_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"SMA calculation failed: {str(e)}")
    
    async def _calculate_ema(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> ToolResult:
        """Calculate Exponential Moving Average"""
        try:
            period = parameters.get("period", 20)
            
            ema = df['close'].ewm(span=period).mean()
            
            result_data = {
                "values": ema.dropna().tolist(),
                "period": period,
                "latest_value": float(ema.iloc[-1]) if not ema.empty else None,
                "latest_price": float(df['close'].iloc[-1]) if not df['close'].empty else None
            }
            
            metadata = {
                "indicator": "EMA",
                "period": period,
                "data_points": len(result_data["values"]),
                "calculated_at": datetime.now().isoformat()
            }
            
            return ToolResult(success=True, data=result_data, metadata=metadata)
            
        except Exception as e:
            raise ValueError(f"EMA calculation failed: {str(e)}")
    
    def _interpret_rsi(self, rsi_value: Optional[float]) -> str:
        """Interpret RSI signal"""
        if rsi_value is None:
            return "unknown"
        elif rsi_value > 70:
            return "overbought"
        elif rsi_value < 30:
            return "oversold"
        elif rsi_value > 50:
            return "bullish"
        else:
            return "bearish"
    
    def _interpret_macd(self, macd: Optional[float], signal: Optional[float], histogram: Optional[float]) -> str:
        """Interpret MACD signal"""
        if macd is None or signal is None or histogram is None:
            return "unknown"
        
        if macd > signal and histogram > 0:
            return "bullish"
        elif macd < signal and histogram < 0:
            return "bearish"
        elif macd > signal:
            return "potential_bullish"
        else:
            return "potential_bearish"
    
    def _interpret_bollinger(self, price: Optional[float], upper: Optional[float], 
                           lower: Optional[float], middle: Optional[float]) -> str:
        """Interpret Bollinger Bands signal"""
        if not all([price, upper, lower, middle]):
            return "unknown"
        
        if price > upper:
            return "overbought"
        elif price < lower:
            return "oversold"
        elif price > middle:
            return "above_middle"
        else:
            return "below_middle"