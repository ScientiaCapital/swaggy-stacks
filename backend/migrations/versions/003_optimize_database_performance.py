"""Optimize database performance for trading system

Revision ID: 003
Revises: 002
Create Date: 2025-09-14 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade():
    """Add performance optimizations for trading system tables"""

    # ===== INDICATOR PERFORMANCE OPTIMIZATIONS =====

    # Composite index for time-series queries with indicator filtering
    op.create_index(
        'ix_indicator_performance_time_series_optimized',
        'indicator_performance',
        ['indicator_name', 'indicator_type', 'period_start'],
        postgresql_using='btree',
        postgresql_ops={'period_start': 'DESC'}
    )

    # Partial index for high-performing indicators (win_rate > 60%)
    op.create_index(
        'ix_indicator_performance_high_performers',
        'indicator_performance',
        ['indicator_name', 'win_rate', 'total_return'],
        postgresql_where=sa.text('win_rate > 0.6'),
        postgresql_using='btree'
    )

    # Composite index for backtest correlation queries
    op.create_index(
        'ix_indicator_performance_backtest_correlation',
        'indicator_performance',
        ['backtest_run_id', 'indicator_name', 'market_condition'],
        postgresql_using='btree'
    )

    # Add check constraints for data integrity
    op.create_check_constraint(
        'ck_indicator_performance_win_rate',
        'indicator_performance',
        'win_rate >= 0 AND win_rate <= 1'
    )

    op.create_check_constraint(
        'ck_indicator_performance_signal_counts',
        'indicator_performance',
        'correct_signals <= total_signals AND total_signals >= 0 AND correct_signals >= 0'
    )

    op.create_check_constraint(
        'ck_indicator_performance_period_order',
        'indicator_performance',
        'period_end > period_start'
    )

    # ===== ML PREDICTIONS OPTIMIZATIONS =====

    # Composite index for symbol-time-confidence queries
    op.create_index(
        'ix_ml_predictions_symbol_time_confidence',
        'ml_predictions',
        ['symbol', 'prediction_time', 'confidence_score'],
        postgresql_using='btree',
        postgresql_ops={'prediction_time': 'DESC', 'confidence_score': 'DESC'}
    )

    # Partial index for pending feedback processing
    op.create_index(
        'ix_ml_predictions_pending_feedback',
        'ml_predictions',
        ['model_version_id', 'prediction_time', 'symbol'],
        postgresql_where=sa.text('feedback_processed = false'),
        postgresql_using='btree'
    )

    # Partial index for successful predictions analysis
    op.create_index(
        'ix_ml_predictions_successful',
        'ml_predictions',
        ['model_version_id', 'confidence_score', 'predicted_return'],
        postgresql_where=sa.text('was_correct = true'),
        postgresql_using='btree'
    )

    # GIN index for fast JSON queries on technical indicators
    op.create_index(
        'ix_ml_predictions_technical_indicators_gin',
        'ml_predictions',
        ['technical_indicators'],
        postgresql_using='gin'
    )

    # Add check constraints
    op.create_check_constraint(
        'ck_ml_predictions_confidence_score',
        'ml_predictions',
        'confidence_score >= 0 AND confidence_score <= 1'
    )

    op.create_check_constraint(
        'ck_ml_predictions_horizon',
        'ml_predictions',
        'prediction_horizon > 0 AND prediction_horizon <= 365'
    )

    op.create_check_constraint(
        'ck_ml_predictions_direction',
        'ml_predictions',
        "predicted_direction IN ('UP', 'DOWN', 'NEUTRAL')"
    )

    # ===== SIGNAL HISTORY OPTIMIZATIONS =====

    # Composite index for trading signal queries
    op.create_index(
        'ix_signal_history_trading_signals',
        'signal_history',
        ['symbol', 'signal_time', 'executed'],
        postgresql_using='btree',
        postgresql_ops={'signal_time': 'DESC'}
    )

    # Partial index for unexecuted signals
    op.create_index(
        'ix_signal_history_pending_execution',
        'signal_history',
        ['signal_time', 'signal_strength', 'symbol'],
        postgresql_where=sa.text('executed = false'),
        postgresql_using='btree'
    )

    # Composite index for strategy performance analysis
    op.create_index(
        'ix_signal_history_strategy_analysis',
        'signal_history',
        ['source_strategy', 'market_trend', 'signal_strength'],
        postgresql_using='btree'
    )

    # Add check constraints
    op.create_check_constraint(
        'ck_signal_history_strength',
        'signal_history',
        'signal_strength >= 0 AND signal_strength <= 100'
    )

    op.create_check_constraint(
        'ck_signal_history_prices',
        'signal_history',
        'entry_price > 0 AND stop_loss > 0 AND take_profit > 0'
    )

    # ===== BACKTEST TRADES OPTIMIZATIONS =====

    # Composite index for backtest performance queries
    op.create_index(
        'ix_backtest_trades_performance',
        'backtest_trades',
        ['backtest_run_id', 'entry_time', 'status'],
        postgresql_using='btree',
        postgresql_ops={'entry_time': 'DESC'}
    )

    # Partial index for open positions
    op.create_index(
        'ix_backtest_trades_open_positions',
        'backtest_trades',
        ['symbol', 'entry_time', 'position_size_usd'],
        postgresql_where=sa.text("status = 'OPEN'"),
        postgresql_using='btree'
    )

    # Composite index for P&L analysis
    op.create_index(
        'ix_backtest_trades_pnl_analysis',
        'backtest_trades',
        ['symbol', 'pnl_percentage', 'entry_time'],
        postgresql_using='btree',
        postgresql_ops={'pnl_percentage': 'DESC', 'entry_time': 'DESC'}
    )

    # Add check constraints
    op.create_check_constraint(
        'ck_backtest_trades_quantity',
        'backtest_trades',
        'quantity > 0'
    )

    op.create_check_constraint(
        'ck_backtest_trades_prices',
        'backtest_trades',
        'entry_price > 0 AND (exit_price IS NULL OR exit_price > 0)'
    )

    op.create_check_constraint(
        'ck_backtest_trades_time_order',
        'backtest_trades',
        'exit_time IS NULL OR exit_time > entry_time'
    )

    # ===== PATTERN DETECTIONS OPTIMIZATIONS =====

    # Composite index for pattern analysis queries
    op.create_index(
        'ix_pattern_detections_analysis',
        'pattern_detections',
        ['symbol', 'pattern_name', 'detection_time', 'confidence'],
        postgresql_using='btree',
        postgresql_ops={'detection_time': 'DESC', 'confidence': 'DESC'}
    )

    # Partial index for high-confidence patterns
    op.create_index(
        'ix_pattern_detections_high_confidence',
        'pattern_detections',
        ['pattern_name', 'pattern_type', 'outcome'],
        postgresql_where=sa.text('confidence >= 80'),
        postgresql_using='btree'
    )

    # Add check constraints
    op.create_check_constraint(
        'ck_pattern_detections_confidence',
        'pattern_detections',
        'confidence >= 0 AND confidence <= 100'
    )

    op.create_check_constraint(
        'ck_pattern_detections_prices',
        'pattern_detections',
        'price_at_detection > 0 AND (outcome_price IS NULL OR outcome_price > 0)'
    )

    # ===== BACKTEST RUNS OPTIMIZATIONS =====

    # Composite index for run status and performance
    op.create_index(
        'ix_backtest_runs_performance',
        'backtest_runs',
        ['status', 'sharpe_ratio', 'total_pnl'],
        postgresql_using='btree',
        postgresql_ops={'sharpe_ratio': 'DESC', 'total_pnl': 'DESC'}
    )

    # Partial index for completed runs
    op.create_index(
        'ix_backtest_runs_completed',
        'backtest_runs',
        ['strategy_id', 'completed_at', 'win_rate'],
        postgresql_where=sa.text("status = 'COMPLETED'"),
        postgresql_using='btree'
    )

    # Add check constraints
    op.create_check_constraint(
        'ck_backtest_runs_dates',
        'backtest_runs',
        'end_date > start_date'
    )

    op.create_check_constraint(
        'ck_backtest_runs_capital',
        'backtest_runs',
        'initial_capital > 0'
    )

    op.create_check_constraint(
        'ck_backtest_runs_trades',
        'backtest_runs',
        'total_trades = winning_trades + losing_trades'
    )

    # ===== FOREIGN KEY OPTIMIZATIONS =====

    # Update foreign keys with cascade options for better data management
    # Note: This would typically be done by dropping and recreating the constraints
    # For safety, we'll add new optimized foreign keys where beneficial

    # ===== PERFORMANCE VIEWS =====

    # Create materialized view for aggregated indicator performance
    op.execute("""
        CREATE MATERIALIZED VIEW mv_indicator_performance_summary AS
        SELECT
            indicator_name,
            indicator_type,
            market_condition,
            COUNT(*) as total_periods,
            AVG(win_rate) as avg_win_rate,
            AVG(total_return) as avg_return,
            AVG(sharpe_ratio) as avg_sharpe,
            AVG(max_drawdown) as avg_max_drawdown,
            STDDEV(win_rate) as win_rate_volatility,
            MAX(period_end) as last_updated
        FROM indicator_performance
        WHERE total_signals >= 10  -- Only include statistically significant results
        GROUP BY indicator_name, indicator_type, market_condition
        ORDER BY avg_win_rate DESC, avg_return DESC;
    """)

    # Create index on the materialized view
    op.create_index(
        'ix_mv_indicator_performance_summary_lookup',
        'mv_indicator_performance_summary',
        ['indicator_name', 'indicator_type', 'avg_win_rate'],
        postgresql_ops={'avg_win_rate': 'DESC'}
    )

    # Create view for real-time P&L calculations
    op.execute("""
        CREATE VIEW v_realtime_pnl AS
        SELECT
            bt.backtest_run_id,
            bt.symbol,
            bt.side,
            bt.entry_time,
            bt.entry_price,
            bt.exit_price,
            bt.quantity,
            bt.pnl,
            bt.pnl_percentage,
            bt.status,
            br.name as backtest_name,
            br.strategy_id,
            CASE
                WHEN bt.status = 'OPEN' THEN 'Active Position'
                WHEN bt.pnl > 0 THEN 'Profitable'
                ELSE 'Loss'
            END as position_status,
            CASE
                WHEN bt.status = 'OPEN' THEN EXTRACT(EPOCH FROM NOW() - bt.entry_time)/3600
                ELSE bt.duration_minutes/60.0
            END as hours_held
        FROM backtest_trades bt
        JOIN backtest_runs br ON bt.backtest_run_id = br.id
        WHERE bt.entry_time >= NOW() - INTERVAL '30 days'
        ORDER BY bt.entry_time DESC;
    """)


def downgrade():
    """Remove performance optimizations"""

    # Drop views
    op.execute("DROP VIEW IF EXISTS v_realtime_pnl;")
    op.execute("DROP MATERIALIZED VIEW IF EXISTS mv_indicator_performance_summary;")

    # Drop indexes (in reverse order of creation)
    op.drop_index('ix_mv_indicator_performance_summary_lookup')
    op.drop_index('ix_backtest_runs_completed')
    op.drop_index('ix_backtest_runs_performance')
    op.drop_index('ix_pattern_detections_high_confidence')
    op.drop_index('ix_pattern_detections_analysis')
    op.drop_index('ix_backtest_trades_pnl_analysis')
    op.drop_index('ix_backtest_trades_open_positions')
    op.drop_index('ix_backtest_trades_performance')
    op.drop_index('ix_signal_history_strategy_analysis')
    op.drop_index('ix_signal_history_pending_execution')
    op.drop_index('ix_signal_history_trading_signals')
    op.drop_index('ix_ml_predictions_technical_indicators_gin')
    op.drop_index('ix_ml_predictions_successful')
    op.drop_index('ix_ml_predictions_pending_feedback')
    op.drop_index('ix_ml_predictions_symbol_time_confidence')
    op.drop_index('ix_indicator_performance_backtest_correlation')
    op.drop_index('ix_indicator_performance_high_performers')
    op.drop_index('ix_indicator_performance_time_series_optimized')

    # Drop check constraints
    constraint_table_pairs = [
        ('ck_backtest_runs_trades', 'backtest_runs'),
        ('ck_backtest_runs_capital', 'backtest_runs'),
        ('ck_backtest_runs_dates', 'backtest_runs'),
        ('ck_pattern_detections_prices', 'pattern_detections'),
        ('ck_pattern_detections_confidence', 'pattern_detections'),
        ('ck_backtest_trades_time_order', 'backtest_trades'),
        ('ck_backtest_trades_prices', 'backtest_trades'),
        ('ck_backtest_trades_quantity', 'backtest_trades'),
        ('ck_signal_history_prices', 'signal_history'),
        ('ck_signal_history_strength', 'signal_history'),
        ('ck_ml_predictions_direction', 'ml_predictions'),
        ('ck_ml_predictions_horizon', 'ml_predictions'),
        ('ck_ml_predictions_confidence_score', 'ml_predictions'),
        ('ck_indicator_performance_period_order', 'indicator_performance'),
        ('ck_indicator_performance_signal_counts', 'indicator_performance'),
        ('ck_indicator_performance_win_rate', 'indicator_performance')
    ]

    for constraint_name, table_name in constraint_table_pairs:
        op.drop_constraint(constraint_name, table_name)