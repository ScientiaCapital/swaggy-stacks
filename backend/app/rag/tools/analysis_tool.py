"""
Analysis Tools for LangChain Integration

Provides comprehensive market and fundamental analysis as LangChain Tools
"""

import json
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from langchain.agents import Tool

logger = logging.getLogger(__name__)


class AnalysisTool:
    """Comprehensive analysis tools for market intelligence"""

    def get_tools(self) -> List[Tool]:
        """Get all analysis tools"""
        return [
            Tool(
                name="technical_summary",
                description="Get comprehensive technical analysis summary for a symbol",
                func=self._technical_summary,
            ),
            Tool(
                name="fundamental_analysis",
                description="Get fundamental analysis metrics for a symbol",
                func=self._fundamental_analysis,
            ),
            Tool(
                name="market_regime_detection",
                description="Detect current market regime (bull/bear/sideways)",
                func=self._market_regime_detection,
            ),
            Tool(
                name="correlation_analysis",
                description="Analyze correlations between symbols (format: 'SYMBOL1,SYMBOL2,SYMBOL3')",
                func=self._correlation_analysis,
            ),
            Tool(
                name="sector_analysis",
                description="Analyze sector performance and rotation",
                func=self._sector_analysis,
            ),
            Tool(
                name="volatility_analysis",
                description="Comprehensive volatility analysis for a symbol",
                func=self._volatility_analysis,
            ),
            Tool(
                name="momentum_analysis",
                description="Multi-timeframe momentum analysis",
                func=self._momentum_analysis,
            ),
            Tool(
                name="comparative_analysis",
                description="Compare multiple symbols (format: 'SYMBOL1,SYMBOL2,SYMBOL3')",
                func=self._comparative_analysis,
            ),
        ]

    def _technical_summary(self, symbol: str) -> str:
        """Comprehensive technical analysis summary"""
        try:
            symbol = symbol.strip().upper()

            # Get historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return f"No data available for technical analysis of {symbol}"

            closes = hist["Close"]
            volumes = hist["Volume"]
            highs = hist["High"]
            lows = hist["Low"]

            current_price = float(closes.iloc[-1])

            # Calculate key technical indicators
            # Moving averages
            ma20 = (
                closes.rolling(20).mean().iloc[-1]
                if len(closes) >= 20
                else current_price
            )
            ma50 = (
                closes.rolling(50).mean().iloc[-1]
                if len(closes) >= 50
                else current_price
            )
            ma200 = (
                closes.rolling(200).mean().iloc[-1]
                if len(closes) >= 200
                else current_price
            )

            # RSI
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 else 50

            # Volatility
            returns = closes.pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility %

            # Volume analysis
            avg_volume = volumes.mean()
            recent_volume = volumes.iloc[-1]
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

            # Price position analysis
            year_high = highs.max()
            year_low = lows.min()
            price_position = (
                (current_price - year_low) / (year_high - year_low)
                if year_high != year_low
                else 0.5
            )

            # Trend analysis
            trend_strength = self._calculate_trend_strength(closes)

            return json.dumps(
                {
                    "symbol": symbol,
                    "current_price": round(current_price, 2),
                    "technical_indicators": {
                        "moving_averages": {
                            "ma20": round(ma20, 2),
                            "ma50": round(ma50, 2),
                            "ma200": round(ma200, 2),
                            "price_vs_ma20": round((current_price / ma20 - 1) * 100, 2),
                            "price_vs_ma50": round((current_price / ma50 - 1) * 100, 2),
                            "ma_alignment": self._assess_ma_alignment(
                                current_price, ma20, ma50, ma200
                            ),
                        },
                        "momentum": {
                            "rsi": round(current_rsi, 1),
                            "rsi_condition": (
                                "overbought"
                                if current_rsi > 70
                                else "oversold" if current_rsi < 30 else "neutral"
                            ),
                        },
                        "volatility": {
                            "annualized_volatility": round(volatility, 1),
                            "volatility_rank": (
                                "high"
                                if volatility > 30
                                else "medium" if volatility > 15 else "low"
                            ),
                        },
                    },
                    "price_analysis": {
                        "year_high": round(year_high, 2),
                        "year_low": round(year_low, 2),
                        "price_position": round(price_position * 100, 1),
                        "distance_from_high": round(
                            (year_high - current_price) / current_price * 100, 1
                        ),
                        "distance_from_low": round(
                            (current_price - year_low) / current_price * 100, 1
                        ),
                    },
                    "volume_analysis": {
                        "average_volume": int(avg_volume),
                        "recent_volume": int(recent_volume),
                        "volume_ratio": round(volume_ratio, 2),
                        "volume_trend": (
                            "above_average"
                            if volume_ratio > 1.2
                            else "below_average" if volume_ratio < 0.8 else "normal"
                        ),
                    },
                    "trend_analysis": trend_strength,
                    "overall_signal": self._generate_overall_signal(
                        current_rsi, trend_strength, current_price, ma20, ma50
                    ),
                }
            )

        except Exception as e:
            logger.error(f"Error in technical summary: {e}")
            return f"Error in technical summary: {str(e)}"

    def _fundamental_analysis(self, symbol: str) -> str:
        """Fundamental analysis metrics"""
        try:
            symbol = symbol.strip().upper()

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "symbol" not in info:
                return f"No fundamental data available for {symbol}"

            # Key financial metrics
            market_cap = info.get("marketCap", 0)
            pe_ratio = info.get("forwardPE", info.get("trailingPE", 0))
            pb_ratio = info.get("priceToBook", 0)
            roe = info.get("returnOnEquity", 0)
            debt_to_equity = info.get("debtToEquity", 0)
            revenue_growth = info.get("revenueGrowth", 0)
            earnings_growth = info.get("earningsGrowth", 0)

            # Dividend metrics
            dividend_rate = info.get("dividendRate", 0)
            dividend_yield = (
                info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
            )
            payout_ratio = (
                info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else 0
            )

            # Profitability metrics
            profit_margins = (
                info.get("profitMargins", 0) * 100 if info.get("profitMargins") else 0
            )
            operating_margins = (
                info.get("operatingMargins", 0) * 100
                if info.get("operatingMargins")
                else 0
            )

            return json.dumps(
                {
                    "symbol": symbol,
                    "company_name": info.get("longName", symbol),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                    "valuation_metrics": {
                        "market_cap": market_cap,
                        "market_cap_category": self._categorize_market_cap(market_cap),
                        "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
                        "pb_ratio": round(pb_ratio, 2) if pb_ratio else None,
                        "valuation_assessment": self._assess_valuation(
                            pe_ratio, pb_ratio
                        ),
                    },
                    "profitability_metrics": {
                        "roe_percent": round(roe * 100, 1) if roe else None,
                        "profit_margins": round(profit_margins, 1),
                        "operating_margins": round(operating_margins, 1),
                        "profitability_score": self._score_profitability(
                            roe, profit_margins, operating_margins
                        ),
                    },
                    "growth_metrics": {
                        "revenue_growth": (
                            round(revenue_growth * 100, 1) if revenue_growth else None
                        ),
                        "earnings_growth": (
                            round(earnings_growth * 100, 1) if earnings_growth else None
                        ),
                        "growth_score": self._score_growth(
                            revenue_growth, earnings_growth
                        ),
                    },
                    "dividend_metrics": {
                        "dividend_rate": dividend_rate,
                        "dividend_yield": round(dividend_yield, 2),
                        "payout_ratio": round(payout_ratio, 1),
                        "dividend_sustainability": self._assess_dividend_sustainability(
                            dividend_yield, payout_ratio
                        ),
                    },
                    "financial_health": {
                        "debt_to_equity": (
                            round(debt_to_equity, 2) if debt_to_equity else None
                        ),
                        "financial_strength": self._assess_financial_strength(
                            debt_to_equity, roe
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
            return f"Error in fundamental analysis: {str(e)}"

    def _market_regime_detection(self, symbol: str = "SPY") -> str:
        """Detect current market regime"""
        try:
            symbol = symbol.strip().upper() or "SPY"

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")

            if hist.empty:
                return f"No data available for market regime detection using {symbol}"

            closes = hist["Close"]
            volumes = hist["Volume"]

            # Multiple timeframe analysis
            short_trend = self._analyze_regime_trend(closes.tail(50))  # ~2 months
            medium_trend = self._analyze_regime_trend(closes.tail(150))  # ~6 months
            long_trend = self._analyze_regime_trend(closes.tail(252))  # ~1 year

            # Volatility regime
            returns = closes.pct_change().dropna()
            recent_vol = returns.tail(20).std() * np.sqrt(252)
            historical_vol = returns.std() * np.sqrt(252)
            vol_regime = (
                "high"
                if recent_vol > historical_vol * 1.2
                else "low" if recent_vol < historical_vol * 0.8 else "normal"
            )

            # Volume regime
            recent_volume = volumes.tail(20).mean()
            historical_volume = volumes.mean()
            volume_regime = (
                "high"
                if recent_volume > historical_volume * 1.2
                else "low" if recent_volume < historical_volume * 0.8 else "normal"
            )

            # Overall regime assessment
            regime_signals = [
                short_trend["regime"],
                medium_trend["regime"],
                long_trend["regime"],
            ]
            dominant_regime = max(set(regime_signals), key=regime_signals.count)

            # Regime strength
            regime_strength = regime_signals.count(dominant_regime) / len(
                regime_signals
            )

            return json.dumps(
                {
                    "market_proxy": symbol,
                    "regime_analysis": {
                        "current_regime": dominant_regime,
                        "regime_strength": round(regime_strength, 2),
                        "timeframe_breakdown": {
                            "short_term": short_trend,
                            "medium_term": medium_trend,
                            "long_term": long_trend,
                        },
                    },
                    "regime_characteristics": {
                        "volatility_regime": vol_regime,
                        "volume_regime": volume_regime,
                        "trend_consistency": regime_strength > 0.66,
                    },
                    "trading_implications": {
                        "strategy_recommendation": self._get_regime_strategy(
                            dominant_regime, vol_regime
                        ),
                        "risk_level": (
                            "high"
                            if vol_regime == "high"
                            else "medium" if vol_regime == "normal" else "low"
                        ),
                        "position_sizing": (
                            "reduced" if vol_regime == "high" else "normal"
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in market regime detection: {e}")
            return f"Error in market regime detection: {str(e)}"

    def _correlation_analysis(self, symbols_str: str) -> str:
        """Analyze correlations between multiple symbols"""
        try:
            symbols = [s.strip().upper() for s in symbols_str.split(",")]

            if len(symbols) < 2:
                return "Need at least 2 symbols for correlation analysis"

            # Get data for all symbols
            correlation_data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="6mo")
                if not hist.empty:
                    correlation_data[symbol] = hist["Close"].pct_change().dropna()

            if len(correlation_data) < 2:
                return "Insufficient data for correlation analysis"

            # Create correlation matrix
            import pandas as pd

            df = pd.DataFrame(correlation_data)
            correlation_matrix = df.corr()

            # Analysis
            correlations = {}
            high_correlations = []
            low_correlations = []

            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    sym1, sym2 = symbols[i], symbols[j]
                    if (
                        sym1 in correlation_matrix.index
                        and sym2 in correlation_matrix.columns
                    ):
                        corr = correlation_matrix.loc[sym1, sym2]
                        correlations[f"{sym1}_{sym2}"] = round(corr, 3)

                        if abs(corr) > 0.7:
                            high_correlations.append((sym1, sym2, corr))
                        elif abs(corr) < 0.3:
                            low_correlations.append((sym1, sym2, corr))

            return json.dumps(
                {
                    "symbols": symbols,
                    "correlation_matrix": correlation_matrix.round(3).to_dict(),
                    "correlation_summary": {
                        "average_correlation": round(
                            np.mean(list(correlations.values())), 3
                        ),
                        "highest_correlation": (
                            max(correlations.values()) if correlations else 0
                        ),
                        "lowest_correlation": (
                            min(correlations.values()) if correlations else 0
                        ),
                        "high_correlations": [
                            (s1, s2, round(c, 3)) for s1, s2, c in high_correlations
                        ],
                        "low_correlations": [
                            (s1, s2, round(c, 3)) for s1, s2, c in low_correlations
                        ],
                    },
                    "diversification_analysis": {
                        "diversification_benefit": len(high_correlations) == 0,
                        "concentration_risk": len(high_correlations)
                        > len(symbols) // 2,
                        "recommendation": (
                            "good_diversification"
                            if len(high_correlations) == 0
                            else "consider_rebalancing"
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return f"Error in correlation analysis: {str(e)}"

    def _sector_analysis(self, _: str = "") -> str:
        """Analyze sector performance and rotation"""
        try:
            # Major sector ETFs for analysis
            sector_etfs = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Industrials": "XLI",
                "Materials": "XLB",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
            }

            sector_performance = {}

            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="3mo")
                    if not hist.empty:
                        returns_3m = (
                            hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1
                        ) * 100
                        recent_vol = (
                            hist["Close"].pct_change().std() * np.sqrt(252) * 100
                        )

                        sector_performance[sector] = {
                            "etf": etf,
                            "returns_3m": round(returns_3m, 2),
                            "volatility": round(recent_vol, 1),
                            "risk_adjusted_return": round(
                                returns_3m / recent_vol if recent_vol > 0 else 0, 2
                            ),
                        }
                except Exception:
                    continue

            if not sector_performance:
                return "Unable to retrieve sector performance data"

            # Rank sectors
            best_performers = sorted(
                sector_performance.items(),
                key=lambda x: x[1]["returns_3m"],
                reverse=True,
            )[:3]
            worst_performers = sorted(
                sector_performance.items(), key=lambda x: x[1]["returns_3m"]
            )[:3]

            # Risk-adjusted performance
            risk_adjusted = sorted(
                sector_performance.items(),
                key=lambda x: x[1]["risk_adjusted_return"],
                reverse=True,
            )[:3]

            return json.dumps(
                {
                    "sector_performance": sector_performance,
                    "sector_rankings": {
                        "best_performers_3m": [
                            (sector, data["returns_3m"])
                            for sector, data in best_performers
                        ],
                        "worst_performers_3m": [
                            (sector, data["returns_3m"])
                            for sector, data in worst_performers
                        ],
                        "best_risk_adjusted": [
                            (sector, data["risk_adjusted_return"])
                            for sector, data in risk_adjusted
                        ],
                    },
                    "rotation_analysis": {
                        "leading_sectors": [sector for sector, _ in best_performers],
                        "lagging_sectors": [sector for sector, _ in worst_performers],
                        "market_style": self._analyze_market_style(sector_performance),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in sector analysis: {e}")
            return f"Error in sector analysis: {str(e)}"

    def _volatility_analysis(self, symbol: str) -> str:
        """Comprehensive volatility analysis"""
        try:
            symbol = symbol.strip().upper()

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return f"No data available for volatility analysis of {symbol}"

            closes = hist["Close"]
            returns = closes.pct_change().dropna()

            # Multiple timeframe volatility
            vol_1w = returns.tail(5).std() * np.sqrt(252) * 100
            vol_1m = returns.tail(21).std() * np.sqrt(252) * 100
            vol_3m = returns.tail(63).std() * np.sqrt(252) * 100
            vol_1y = returns.std() * np.sqrt(252) * 100

            # Volatility percentiles
            rolling_vol = returns.rolling(21).std() * np.sqrt(252) * 100
            current_vol_percentile = (rolling_vol < vol_1m).mean() * 100

            # Volatility clustering
            vol_clustering = self._detect_volatility_clustering(returns)

            return json.dumps(
                {
                    "symbol": symbol,
                    "volatility_metrics": {
                        "current_1week": round(vol_1w, 1),
                        "current_1month": round(vol_1m, 1),
                        "current_3month": round(vol_3m, 1),
                        "annual_average": round(vol_1y, 1),
                    },
                    "volatility_analysis": {
                        "volatility_trend": (
                            "increasing" if vol_1m > vol_3m else "decreasing"
                        ),
                        "volatility_rank": (
                            "high"
                            if vol_1m > 30
                            else "medium" if vol_1m > 15 else "low"
                        ),
                        "volatility_percentile": round(current_vol_percentile, 1),
                        "relative_volatility": (
                            "above_average" if vol_1m > vol_1y else "below_average"
                        ),
                    },
                    "volatility_patterns": {
                        "clustering_detected": vol_clustering["has_clustering"],
                        "regime_changes": vol_clustering["regime_changes"],
                        "stability_score": vol_clustering["stability_score"],
                    },
                    "trading_implications": {
                        "option_premium": (
                            "expensive"
                            if current_vol_percentile > 70
                            else "cheap" if current_vol_percentile < 30 else "fair"
                        ),
                        "position_sizing": (
                            "reduce"
                            if vol_1m > 40
                            else "increase" if vol_1m < 10 else "normal"
                        ),
                        "strategy_recommendation": self._get_volatility_strategy(
                            vol_1m, current_vol_percentile
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in volatility analysis: {e}")
            return f"Error in volatility analysis: {str(e)}"

    def _momentum_analysis(self, symbol: str) -> str:
        """Multi-timeframe momentum analysis"""
        try:
            symbol = symbol.strip().upper()

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")

            if hist.empty:
                return f"No data available for momentum analysis of {symbol}"

            closes = hist["Close"]
            current_price = closes.iloc[-1]

            # Multiple timeframe returns
            returns = {
                "1_week": (
                    (current_price / closes.iloc[-5] - 1) * 100
                    if len(closes) >= 5
                    else 0
                ),
                "1_month": (
                    (current_price / closes.iloc[-21] - 1) * 100
                    if len(closes) >= 21
                    else 0
                ),
                "3_month": (
                    (current_price / closes.iloc[-63] - 1) * 100
                    if len(closes) >= 63
                    else 0
                ),
                "6_month": (
                    (current_price / closes.iloc[-126] - 1) * 100
                    if len(closes) >= 126
                    else 0
                ),
                "1_year": (
                    (current_price / closes.iloc[0] - 1) * 100
                    if len(closes) >= 252
                    else 0
                ),
            }

            # Momentum scoring
            momentum_scores = {}
            for period, return_pct in returns.items():
                if abs(return_pct) > 20:
                    score = "strong"
                elif abs(return_pct) > 10:
                    score = "moderate"
                elif abs(return_pct) > 5:
                    score = "weak"
                else:
                    score = "neutral"

                momentum_scores[period] = {
                    "return_percent": round(return_pct, 2),
                    "direction": "positive" if return_pct > 0 else "negative",
                    "strength": score,
                }

            # Momentum consistency
            positive_periods = sum(1 for r in returns.values() if r > 0)
            momentum_consistency = positive_periods / len(returns)

            # Rate of change analysis
            roc_short = returns["1_month"] - returns["1_week"]
            roc_long = returns["6_month"] - returns["3_month"]

            return json.dumps(
                {
                    "symbol": symbol,
                    "current_price": round(current_price, 2),
                    "momentum_analysis": momentum_scores,
                    "momentum_summary": {
                        "overall_direction": (
                            "bullish"
                            if momentum_consistency > 0.6
                            else "bearish" if momentum_consistency < 0.4 else "mixed"
                        ),
                        "consistency_score": round(momentum_consistency, 2),
                        "strongest_timeframe": max(
                            returns.items(), key=lambda x: abs(x[1])
                        )[0],
                        "acceleration": {
                            "short_term": (
                                "accelerating"
                                if roc_short > 2
                                else "decelerating" if roc_short < -2 else "stable"
                            ),
                            "long_term": (
                                "accelerating"
                                if roc_long > 5
                                else "decelerating" if roc_long < -5 else "stable"
                            ),
                        },
                    },
                    "trading_signals": {
                        "trend_following": (
                            "buy"
                            if momentum_consistency > 0.7
                            else "sell" if momentum_consistency < 0.3 else "hold"
                        ),
                        "mean_reversion": (
                            "sell"
                            if returns["1_week"] > 10
                            else "buy" if returns["1_week"] < -10 else "hold"
                        ),
                        "momentum_quality": (
                            "high"
                            if momentum_consistency > 0.8
                            and abs(returns["3_month"]) > 10
                            else "low"
                        ),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in momentum analysis: {e}")
            return f"Error in momentum analysis: {str(e)}"

    def _comparative_analysis(self, symbols_str: str) -> str:
        """Compare multiple symbols across various metrics"""
        try:
            symbols = [s.strip().upper() for s in symbols_str.split(",")]

            if len(symbols) < 2:
                return "Need at least 2 symbols for comparative analysis"

            comparison_data = {}

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="6mo")
                    info = ticker.info

                    if hist.empty:
                        continue

                    closes = hist["Close"]
                    volumes = hist["Volume"]
                    returns = closes.pct_change().dropna()

                    current_price = closes.iloc[-1]

                    comparison_data[symbol] = {
                        "current_price": round(current_price, 2),
                        "returns": {
                            "1_month": (
                                round((current_price / closes.iloc[-21] - 1) * 100, 2)
                                if len(closes) >= 21
                                else 0
                            ),
                            "3_month": (
                                round((current_price / closes.iloc[-63] - 1) * 100, 2)
                                if len(closes) >= 63
                                else 0
                            ),
                            "6_month": round(
                                (current_price / closes.iloc[0] - 1) * 100, 2
                            ),
                        },
                        "volatility": round(returns.std() * np.sqrt(252) * 100, 1),
                        "sharpe_ratio": round(
                            (returns.mean() * 252) / (returns.std() * np.sqrt(252)), 2
                        ),
                        "max_drawdown": self._calculate_max_drawdown(closes),
                        "average_volume": int(volumes.mean()),
                        "market_cap": info.get("marketCap", 0),
                        "pe_ratio": info.get("forwardPE", info.get("trailingPE", 0))
                        or 0,
                        "sector": info.get("sector", "Unknown"),
                    }

                except Exception as e:
                    logger.warning(f"Error analyzing {symbol}: {e}")
                    continue

            if len(comparison_data) < 2:
                return "Insufficient data for comparative analysis"

            # Rankings
            rankings = {
                "best_1m_return": sorted(
                    comparison_data.items(),
                    key=lambda x: x[1]["returns"]["1_month"],
                    reverse=True,
                ),
                "best_6m_return": sorted(
                    comparison_data.items(),
                    key=lambda x: x[1]["returns"]["6_month"],
                    reverse=True,
                ),
                "best_sharpe": sorted(
                    comparison_data.items(),
                    key=lambda x: x[1]["sharpe_ratio"],
                    reverse=True,
                ),
                "lowest_volatility": sorted(
                    comparison_data.items(), key=lambda x: x[1]["volatility"]
                ),
                "smallest_drawdown": sorted(
                    comparison_data.items(), key=lambda x: x[1]["max_drawdown"]
                ),
            }

            return json.dumps(
                {
                    "symbols": symbols,
                    "comparative_metrics": comparison_data,
                    "rankings": {
                        "performance_leaders": {
                            "1_month": rankings["best_1m_return"][0][0],
                            "6_month": rankings["best_6m_return"][0][0],
                        },
                        "risk_metrics": {
                            "best_risk_adjusted": rankings["best_sharpe"][0][0],
                            "most_stable": rankings["lowest_volatility"][0][0],
                            "smallest_drawdown": rankings["smallest_drawdown"][0][0],
                        },
                    },
                    "investment_recommendation": {
                        "growth_pick": rankings["best_6m_return"][0][0],
                        "value_pick": min(
                            comparison_data.items(),
                            key=lambda x: (
                                x[1]["pe_ratio"] if x[1]["pe_ratio"] > 0 else 999
                            ),
                        )[0],
                        "defensive_pick": rankings["lowest_volatility"][0][0],
                        "overall_winner": rankings["best_sharpe"][0][0],
                    },
                }
            )

        except Exception as e:
            logger.error(f"Error in comparative analysis: {e}")
            return f"Error in comparative analysis: {str(e)}"

    # Helper methods
    def _calculate_trend_strength(self, closes: pd.Series) -> dict:
        """Calculate trend strength metrics"""
        if len(closes) < 20:
            return {"strength": "unknown", "direction": "unknown", "consistency": 0}

        # Calculate trend using linear regression
        x = np.arange(len(closes.tail(50)))
        y = closes.tail(50).values

        try:
            from scipy import stats

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Calculate trend characteristics
            trend_strength = abs(slope * len(x) / y[0]) * 100 if y[0] != 0 else 0

            return {
                "strength": (
                    "strong"
                    if trend_strength > 2
                    else "moderate" if trend_strength > 1 else "weak"
                ),
                "direction": "up" if slope > 0 else "down",
                "consistency": round(r_value**2, 3),
                "slope_percent": round(trend_strength, 2),
            }
        except Exception:
            return {"strength": "unknown", "direction": "unknown", "consistency": 0}

    def _assess_ma_alignment(
        self, price: float, ma20: float, ma50: float, ma200: float
    ) -> str:
        """Assess moving average alignment"""
        if price > ma20 > ma50 > ma200:
            return "bullish_alignment"
        elif price < ma20 < ma50 < ma200:
            return "bearish_alignment"
        elif price > ma20 and ma20 > ma50:
            return "short_term_bullish"
        elif price < ma20 and ma20 < ma50:
            return "short_term_bearish"
        else:
            return "mixed"

    def _generate_overall_signal(
        self, rsi: float, trend: dict, price: float, ma20: float, ma50: float
    ) -> dict:
        """Generate overall trading signal"""
        signals = []

        # Trend signal
        if trend["strength"] == "strong" and trend["direction"] == "up":
            signals.append("buy")
        elif trend["strength"] == "strong" and trend["direction"] == "down":
            signals.append("sell")

        # MA signal
        if price > ma20 > ma50:
            signals.append("buy")
        elif price < ma20 < ma50:
            signals.append("sell")

        # RSI signal
        if rsi < 30:
            signals.append("buy")
        elif rsi > 70:
            signals.append("sell")

        # Consensus
        buy_signals = signals.count("buy")
        sell_signals = signals.count("sell")

        if buy_signals > sell_signals:
            return {
                "signal": "BUY",
                "strength": buy_signals,
                "confidence": buy_signals / len(signals),
            }
        elif sell_signals > buy_signals:
            return {
                "signal": "SELL",
                "strength": sell_signals,
                "confidence": sell_signals / len(signals),
            }
        else:
            return {"signal": "HOLD", "strength": 0, "confidence": 0.5}

    def _categorize_market_cap(self, market_cap: int) -> str:
        """Categorize market cap"""
        if market_cap >= 200_000_000_000:
            return "mega_cap"
        elif market_cap >= 10_000_000_000:
            return "large_cap"
        elif market_cap >= 2_000_000_000:
            return "mid_cap"
        elif market_cap >= 300_000_000:
            return "small_cap"
        else:
            return "micro_cap"

    def _assess_valuation(self, pe: float, pb: float) -> str:
        """Assess valuation metrics"""
        if not pe or not pb:
            return "insufficient_data"

        if pe < 15 and pb < 1.5:
            return "undervalued"
        elif pe > 25 or pb > 3:
            return "overvalued"
        else:
            return "fairly_valued"

    def _score_profitability(
        self, roe: float, profit_margin: float, operating_margin: float
    ) -> str:
        """Score profitability metrics"""
        score = 0

        if roe and roe > 0.15:
            score += 1
        if profit_margin > 10:
            score += 1
        if operating_margin > 15:
            score += 1

        return (
            "excellent"
            if score == 3
            else "good" if score == 2 else "average" if score == 1 else "poor"
        )

    def _score_growth(self, revenue_growth: float, earnings_growth: float) -> str:
        """Score growth metrics"""
        if not revenue_growth or not earnings_growth:
            return "insufficient_data"

        if revenue_growth > 0.1 and earnings_growth > 0.1:
            return "high_growth"
        elif revenue_growth > 0.05 or earnings_growth > 0.05:
            return "moderate_growth"
        else:
            return "low_growth"

    def _assess_dividend_sustainability(
        self, dividend_yield: float, payout_ratio: float
    ) -> str:
        """Assess dividend sustainability"""
        if dividend_yield == 0:
            return "no_dividend"

        if payout_ratio > 100:
            return "unsustainable"
        elif payout_ratio > 80:
            return "risky"
        elif payout_ratio > 60:
            return "moderate"
        else:
            return "sustainable"

    def _assess_financial_strength(self, debt_to_equity: float, roe: float) -> str:
        """Assess financial strength"""
        score = 0

        if debt_to_equity is not None and debt_to_equity < 0.5:
            score += 1
        if roe and roe > 0.12:
            score += 1

        return "strong" if score == 2 else "moderate" if score == 1 else "weak"

    def _analyze_regime_trend(self, prices: pd.Series) -> dict:
        """Analyze trend for regime detection"""
        if len(prices) < 10:
            return {"regime": "unknown", "confidence": 0}

        returns = prices.pct_change().dropna()

        # Calculate trend metrics
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        positive_days = (returns > 0).mean()

        # Regime classification
        if total_return > 10 and positive_days > 0.55:
            regime = "bull"
            confidence = min(total_return / 20, 1.0)
        elif total_return < -10 and positive_days < 0.45:
            regime = "bear"
            confidence = min(abs(total_return) / 20, 1.0)
        else:
            regime = "sideways"
            confidence = 1 - min(abs(total_return) / 10, 1.0)

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "return_percent": round(total_return, 2),
            "volatility": round(volatility, 1),
            "positive_day_ratio": round(positive_days, 2),
        }

    def _get_regime_strategy(self, regime: str, vol_regime: str) -> str:
        """Get strategy recommendation based on regime"""
        if regime == "bull":
            return "momentum" if vol_regime == "low" else "selective_growth"
        elif regime == "bear":
            return "defensive" if vol_regime == "high" else "value_hunting"
        else:
            return "range_trading" if vol_regime == "low" else "volatility_trading"

    def _analyze_market_style(self, sector_performance: dict) -> str:
        """Analyze market style based on sector performance"""
        growth_sectors = ["Technology", "Consumer Discretionary", "Healthcare"]
        value_sectors = ["Financials", "Energy", "Materials"]
        defensive_sectors = ["Utilities", "Consumer Staples", "Real Estate"]

        growth_avg = (
            np.mean(
                [
                    sector_performance[s]["returns_3m"]
                    for s in growth_sectors
                    if s in sector_performance
                ]
            )
            if any(s in sector_performance for s in growth_sectors)
            else 0
        )

        value_avg = (
            np.mean(
                [
                    sector_performance[s]["returns_3m"]
                    for s in value_sectors
                    if s in sector_performance
                ]
            )
            if any(s in sector_performance for s in value_sectors)
            else 0
        )

        defensive_avg = (
            np.mean(
                [
                    sector_performance[s]["returns_3m"]
                    for s in defensive_sectors
                    if s in sector_performance
                ]
            )
            if any(s in sector_performance for s in defensive_sectors)
            else 0
        )

        if growth_avg > value_avg and growth_avg > defensive_avg:
            return "growth_favored"
        elif value_avg > growth_avg and value_avg > defensive_avg:
            return "value_favored"
        elif defensive_avg > growth_avg and defensive_avg > value_avg:
            return "defensive_favored"
        else:
            return "mixed"

    def _detect_volatility_clustering(self, returns: pd.Series) -> dict:
        """Detect volatility clustering patterns"""
        if len(returns) < 50:
            return {
                "has_clustering": False,
                "regime_changes": 0,
                "stability_score": 0.5,
            }

        # Calculate rolling volatility
        rolling_vol = returns.rolling(20).std()
        vol_changes = rolling_vol.diff().abs()

        # Detect regime changes (simplified)
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        regime_changes = (
            (rolling_vol > vol_mean + vol_std) | (rolling_vol < vol_mean - vol_std)
        ).sum()

        # Stability score
        stability_score = (
            1 - (vol_changes.mean() / rolling_vol.mean())
            if rolling_vol.mean() > 0
            else 0.5
        )

        return {
            "has_clustering": regime_changes > len(returns) * 0.1,
            "regime_changes": int(regime_changes),
            "stability_score": round(stability_score, 2),
        }

    def _get_volatility_strategy(self, vol_level: float, vol_percentile: float) -> str:
        """Get strategy recommendation based on volatility"""
        if vol_percentile > 80:
            return "sell_volatility"  # High vol - sell options
        elif vol_percentile < 20:
            return "buy_volatility"  # Low vol - buy options
        elif vol_level > 30:
            return "defensive"  # High absolute vol
        else:
            return "neutral"

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return round(drawdown.min() * 100, 2)


# Global analysis tool instance
_analysis_tool: Optional[AnalysisTool] = None


def get_analysis_tool() -> AnalysisTool:
    """Get the global analysis tool instance"""
    global _analysis_tool

    if _analysis_tool is None:
        _analysis_tool = AnalysisTool()

    return _analysis_tool
