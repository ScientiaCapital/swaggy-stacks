# Trading Algorithms and Strategies

## Core Trading System Components

### Consolidated Markov System (`app/analysis/consolidated_markov_system.py`)
**Comprehensive Markov chain analysis engine consolidated from multiple modules:**

#### MarkovCore Class
- **Multi-state regime detection**: Advanced statistical analysis of market states
- **Volatility integration**: Market volatility considerations in state transitions
- **Volume analysis**: Trading volume impact on state probability
- **Technical indicators**: RSI, MACD, Bollinger Bands integration

#### EnhancedMarkovSystem Class
- **Advanced features**: Extends MarkovCore with enhanced analysis capabilities  
- **Risk-adjusted signals**: Position sizing based on market regime confidence
- **Dynamic lookback periods**: Adaptive historical analysis windows
- **Performance tracking**: Strategy performance metrics and optimization

#### EnhancedDataHandler Class
- **Market data processing**: Clean, normalize, and prepare market data
- **Technical indicator calculation**: Automated technical analysis
- **Data validation**: Ensure data quality and consistency
- **Historical data management**: Efficient storage and retrieval

### Plugin-Based Strategy System (`app/rag/agents/consolidated_strategy_agent.py`)
**Unified strategy agent system supporting multiple trading methodologies:**

#### Available Strategy Plugins
1. **Markov Strategy**: Statistical regime analysis and transition probabilities
2. **Wyckoff Strategy**: Market accumulation and distribution phase analysis  
3. **Fibonacci Strategy**: Retracement and extension level analysis
4. **Elliott Wave Strategy**: Wave pattern recognition and counting

#### ConsolidatedStrategyAgent Features
- **Multi-strategy consensus**: Combine signals from multiple strategies
- **Weighted decision making**: Confidence-based strategy weighting
- **Plugin architecture**: Easy addition of new strategies
- **Performance tracking**: Individual strategy performance metrics

### Risk Management System

#### Position Sizing (PositionSizer class)
- **Kelly Criterion**: Optimal position sizing based on win probability and risk-reward ratio
- **Volatility-based sizing**: Position size adjusted for market volatility
- **Portfolio heat**: Total portfolio risk exposure management
- **Maximum position limits**: Hard limits on individual position sizes

#### Risk Controls
- **Daily loss limits**: Automatic trading halt on excessive losses
- **Portfolio exposure**: Maximum percentage of portfolio at risk
- **Correlation analysis**: Avoid over-concentration in correlated positions
- **Drawdown management**: Dynamic position reduction during adverse periods

### TradingManager Singleton (`app/trading/trading_manager.py`)
**Centralized trading operations management:**

#### Core Functionality
- **Order execution**: Interface with Alpaca API for trade execution
- **Portfolio management**: Real-time portfolio state tracking  
- **Risk validation**: Pre-trade risk checks and position validation
- **Session management**: Trading session lifecycle management
- **Health monitoring**: System health checks and error handling

#### Integration Features
- **Strategy coordination**: Interface with consolidated strategy agents
- **Market data integration**: Real-time market data processing
- **Background task coordination**: Celery task management
- **Performance tracking**: Trade execution metrics and analysis

## Technical Indicators Integration

### Built-in Indicators
- **RSI (Relative Strength Index)**: Momentum oscillator for overbought/oversold conditions
- **MACD**: Moving Average Convergence Divergence for trend analysis
- **Bollinger Bands**: Volatility-based support and resistance levels
- **Moving Averages**: Simple and Exponential moving averages
- **Volume indicators**: Volume-based confirmation signals

### Advanced Analysis
- **Regime detection**: Statistical analysis of market behavior phases
- **Volatility clustering**: Identification of high/low volatility periods
- **Mean reversion**: Statistical identification of mean-reverting patterns
- **Trend following**: Momentum-based trend identification algorithms