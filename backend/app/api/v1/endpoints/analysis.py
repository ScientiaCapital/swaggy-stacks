"""
Analysis endpoints for trading system with modern indicators and ML predictions
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.indicators.indicator_factory import IndicatorFactory, IndicatorType
from app.ml.llm_predictors import get_llm_predictor
from app.core.database import get_db
from app.services.backtest_service import BacktestService

router = APIRouter()

# Initialize services
indicator_factory = IndicatorFactory()
backtest_service = BacktestService()


# Pydantic models for request/response
class MarketDataRequest(BaseModel):
    """Market data input for indicator calculations"""

    symbol: str = Field(..., description="Trading symbol (e.g., AAPL)")
    prices: List[float] = Field(..., description="Historical price data")
    volumes: Optional[List[int]] = Field(None, description="Volume data")
    timestamps: Optional[List[str]] = Field(None, description="Timestamp data")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "prices": [150.0, 151.5, 149.8, 152.1, 153.0],
                "volumes": [1000000, 1200000, 800000, 1500000, 1100000],
                "timestamps": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
            }
        }


class IndicatorCalculationRequest(BaseModel):
    """Request for calculating modern indicators"""

    market_data: MarketDataRequest
    indicator_type: str = Field(
        default="modern", description="Type: traditional, modern, or both"
    )
    specific_indicators: Optional[List[str]] = Field(
        None, description="Specific indicators to calculate"
    )
    use_cache: bool = Field(default=True, description="Use cached results if available")


class LLMPredictionRequest(BaseModel):
    """Request for LLM-based predictions"""

    market_data: MarketDataRequest
    horizon_days: int = Field(
        default=5, ge=1, le=30, description="Prediction horizon in days"
    )
    models: Optional[List[str]] = Field(None, description="Specific LLM models to use")


class PerformanceQueryRequest(BaseModel):
    """Request for performance metrics query"""

    indicator_name: Optional[str] = Field(None, description="Specific indicator name")
    indicator_type: Optional[str] = Field(
        None, description="Filter by type: TRADITIONAL, MODERN, LLM"
    )
    market_condition: Optional[str] = Field(None, description="Market condition filter")
    lookback_days: Optional[int] = Field(
        default=90, description="Lookback period in days"
    )


class OptimizationRequest(BaseModel):
    """Request for parameter optimization"""

    indicator_name: str = Field(..., description="Indicator to optimize")
    parameter_space: Dict[str, Dict[str, Any]] = Field(
        ..., description="Parameter search space"
    )
    optimization_metric: str = Field(
        default="sharpe_ratio", description="Optimization target metric"
    )
    max_iterations: int = Field(
        default=50, ge=1, le=200, description="Maximum optimization iterations"
    )
    market_condition: Optional[str] = Field(
        None, description="Optimize for specific market condition"
    )


@router.get("/")
async def get_analysis():
    """Get analysis overview"""
    return {
        "status": "Modern indicator analysis API ready",
        "version": "2.0.0",
        "features": [
            "Modern technical indicators (30+ indicators)",
            "LLM ensemble predictions",
            "Performance tracking",
            "Parameter optimization",
            "Real-time calculations",
        ],
        "endpoints": [
            "/indicators/modern",
            "/indicators/llm-predict",
            "/indicators/performance",
            "/indicators/optimize",
            "/indicators/top-performers",
        ],
    }


@router.post("/indicators/modern")
async def calculate_modern_indicators(
    request: IndicatorCalculationRequest, db: Session = Depends(get_db)
):
    """Calculate modern technical indicators"""
    try:
        # Convert request to DataFrame
        market_data = request.market_data
        data_dict = {"close": market_data.prices}

        if market_data.volumes:
            data_dict["volume"] = market_data.volumes[: len(market_data.prices)]

        if market_data.timestamps:
            data_dict["timestamp"] = pd.to_datetime(
                market_data.timestamps[: len(market_data.prices)]
            )
            df = pd.DataFrame(data_dict).set_index("timestamp")
        else:
            df = pd.DataFrame(data_dict)

        # Generate OHLC if not provided
        if "open" not in df.columns:
            df["open"] = df["close"].shift(1).fillna(df["close"])
        if "high" not in df.columns:
            df["high"] = df[["open", "close"]].max(axis=1)
        if "low" not in df.columns:
            df["low"] = df[["open", "close"]].min(axis=1)

        # Calculate indicators
        indicator_type = (
            IndicatorType.MODERN
            if request.indicator_type == "modern"
            else IndicatorType.BOTH
        )

        indicators = indicator_factory.calculate_all(
            df,
            indicator_type=indicator_type,
            use_cache=request.use_cache,
            force_refresh=False,
        )

        # Filter specific indicators if requested
        if request.specific_indicators:
            indicators = {
                k: v for k, v in indicators.items() if k in request.specific_indicators
            }

        return {
            "status": "success",
            "symbol": market_data.symbol,
            "indicator_type": request.indicator_type,
            "indicators_calculated": len(indicators),
            "indicators": indicators,
            "calculation_time": datetime.now().isoformat(),
            "data_points": len(df),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating indicators: {str(e)}"
        )


@router.post("/indicators/llm-predict")
async def get_llm_predictions(
    request: LLMPredictionRequest, background_tasks: BackgroundTasks
):
    """Get LLM ensemble predictions"""
    try:
        # Convert request to DataFrame
        market_data = request.market_data
        data_dict = {"close": market_data.prices}

        if market_data.volumes:
            data_dict["volume"] = market_data.volumes[: len(market_data.prices)]

        df = pd.DataFrame(data_dict)

        # Get LLM predictor
        llm_predictor = get_llm_predictor()

        if not llm_predictor:
            raise HTTPException(
                status_code=503, detail="LLM prediction service unavailable"
            )

        # Generate predictions
        prediction_result = await llm_predictor.ensemble_predict(
            symbol=market_data.symbol, market_data=df, horizon_days=request.horizon_days
        )

        # Track prediction performance in background
        if prediction_result:
            background_tasks.add_task(
                track_prediction_performance,
                market_data.symbol,
                prediction_result,
                request.horizon_days,
            )

        return {
            "status": "success",
            "symbol": market_data.symbol,
            "horizon_days": request.horizon_days,
            "prediction_time": datetime.now().isoformat(),
            "ensemble_prediction": prediction_result.get("ensemble_prediction"),
            "confidence": prediction_result.get("confidence", 0.0),
            "individual_models": prediction_result.get("model_predictions", {}),
            "reasoning": prediction_result.get("reasoning", ""),
            "risk_assessment": prediction_result.get("risk_assessment", {}),
            "market_context": prediction_result.get("market_context", {}),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating LLM predictions: {str(e)}"
        )


@router.get("/indicators/performance/{indicator_name}")
async def get_indicator_performance(
    indicator_name: str,
    market_condition: Optional[str] = Query(
        None, description="Market condition filter"
    ),
    lookback_days: Optional[int] = Query(90, description="Lookback period"),
):
    """Get performance metrics for a specific indicator"""
    try:
        report = await backtest_service.get_indicator_performance_report(
            indicator_name=indicator_name,
            market_condition=market_condition,
            lookback_days=lookback_days,
        )

        return report

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving indicator performance: {str(e)}"
        )


@router.post("/indicators/performance/report")
async def get_performance_report(request: PerformanceQueryRequest):
    """Get comprehensive performance report"""
    try:
        report = await backtest_service.get_indicator_performance_report(
            indicator_name=request.indicator_name,
            indicator_type=request.indicator_type,
            market_condition=request.market_condition,
            lookback_days=request.lookback_days,
        )

        return report

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating performance report: {str(e)}"
        )


@router.get("/indicators/top-performers")
async def get_top_performers(
    metric: str = Query(default="sharpe_ratio", description="Ranking metric"),
    top_n: int = Query(default=10, ge=1, le=50, description="Number of top performers"),
    min_signals: Optional[int] = Query(None, description="Minimum signals threshold"),
):
    """Get top performing indicators"""
    try:
        result = await backtest_service.get_top_performing_indicators(
            metric=metric, top_n=top_n, min_signals=min_signals
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving top performers: {str(e)}"
        )


@router.post("/indicators/optimize")
async def optimize_parameters(
    request: OptimizationRequest, background_tasks: BackgroundTasks
):
    """Optimize indicator parameters"""
    try:
        # Start optimization in background
        background_tasks.add_task(
            run_parameter_optimization,
            request.indicator_name,
            request.parameter_space,
            request.optimization_metric,
            request.max_iterations,
            request.market_condition,
        )

        return {
            "status": "optimization_started",
            "indicator_name": request.indicator_name,
            "optimization_metric": request.optimization_metric,
            "max_iterations": request.max_iterations,
            "message": "Parameter optimization started in background. Use /indicators/optimization-status to track progress.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error starting optimization: {str(e)}"
        )


@router.get("/indicators/market-conditions")
async def get_market_condition_analysis(
    market_condition: Optional[str] = Query(
        None, description="Specific market condition"
    )
):
    """Get indicator performance by market conditions"""
    try:
        analysis = await backtest_service.get_indicator_market_condition_analysis(
            market_condition=market_condition
        )

        return analysis

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing market conditions: {str(e)}"
        )


@router.post("/indicators/track-signal")
async def track_signal_performance(
    indicator_name: str,
    indicator_type: str,
    signal_data: Dict[str, Any],
    market_conditions: Dict[str, Any],
    actual_outcome: Optional[Dict[str, Any]] = None,
):
    """Track individual indicator signal performance"""
    try:
        result = await backtest_service.track_indicator_performance(
            indicator_name=indicator_name,
            indicator_type=indicator_type,
            signal_data=signal_data,
            market_conditions=market_conditions,
            actual_outcome=actual_outcome,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error tracking signal: {str(e)}")


@router.put("/indicators/signal-outcome/{signal_id}")
async def update_signal_outcome(signal_id: str, actual_outcome: Dict[str, Any]):
    """Update actual outcome for a tracked signal"""
    try:
        result = await backtest_service.update_indicator_signal_outcome(
            signal_id=signal_id, actual_outcome=actual_outcome
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating signal outcome: {str(e)}"
        )


# Background task functions
async def track_prediction_performance(
    symbol: str, prediction_result: Dict[str, Any], horizon_days: int
):
    """Background task to track LLM prediction performance"""
    try:
        # Record prediction for later outcome tracking
        await backtest_service.track_indicator_performance(
            indicator_name="llm_ensemble",
            indicator_type="LLM",
            signal_data={
                "signal_direction": prediction_result.get("ensemble_prediction"),
                "confidence": prediction_result.get("confidence", 0.0),
                "individual_predictions": prediction_result.get(
                    "model_predictions", {}
                ),
            },
            market_conditions=prediction_result.get("market_context", {}),
            actual_outcome=None,  # Will be updated later
        )

    except Exception as e:
        print(f"Error tracking prediction performance: {e}")


async def run_parameter_optimization(
    indicator_name: str,
    parameter_space: Dict[str, Dict[str, Any]],
    optimization_metric: str,
    max_iterations: int,
    market_condition: Optional[str],
):
    """Background task for parameter optimization"""
    try:
        # This would integrate with the optimization system we'll build in Task 12
        # For now, just log the optimization request
        print(
            f"Starting optimization for {indicator_name} with {max_iterations} iterations"
        )

        # TODO: Implement actual optimization logic in Task 12

    except Exception as e:
        print(f"Error in parameter optimization: {e}")


# WebSocket endpoint for real-time updates (if needed)
@router.websocket("/indicators/realtime")
async def realtime_indicators(websocket):
    """WebSocket endpoint for real-time indicator updates"""
    await websocket.accept()

    try:
        while True:
            # Accept market data from client
            await websocket.receive_json()

            # Calculate indicators in real-time
            # This would integrate with live data feeds

            await websocket.send_json(
                {
                    "status": "indicators_updated",
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
