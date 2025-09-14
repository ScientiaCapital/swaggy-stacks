"""Add backtesting tables

Revision ID: 001_backtesting
Revises:
Create Date: 2024-09-14 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = '001_backtesting'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Add backtesting tables"""

    # Create backtest_runs table
    op.create_table(
        'backtest_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('strategy_id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('symbols', JSON, nullable=False),
        sa.Column('initial_capital', sa.Numeric(precision=15, scale=2), nullable=False, server_default='10000'),
        sa.Column('parameters', JSON, nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='PENDING'),
        sa.Column('progress', sa.Numeric(precision=5, scale=2), server_default='0'),
        sa.Column('total_trades', sa.Integer(), server_default='0'),
        sa.Column('winning_trades', sa.Integer(), server_default='0'),
        sa.Column('losing_trades', sa.Integer(), server_default='0'),
        sa.Column('final_capital', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('total_pnl', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('sharpe_ratio', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('win_rate', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('profit_factor', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('max_loss_per_trade', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('var_95', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('execution_time', sa.Numeric(precision=10, scale=3), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategies.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_backtest_runs_id'), 'backtest_runs', ['id'], unique=False)

    # Create pattern_detections table
    op.create_table(
        'pattern_detections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('backtest_run_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('pattern_name', sa.String(length=100), nullable=False),
        sa.Column('pattern_type', sa.String(length=50), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('strength', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('timeframe', sa.String(length=10), nullable=False),
        sa.Column('detection_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('price_at_detection', sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column('volume_at_detection', sa.Integer(), nullable=True),
        sa.Column('pattern_data', JSON, nullable=True),
        sa.Column('outcome', sa.String(length=20), nullable=True),
        sa.Column('outcome_pnl', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('outcome_price', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('outcome_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('market_condition', sa.String(length=20), nullable=True),
        sa.Column('trend_direction', sa.String(length=10), nullable=True),
        sa.Column('volatility_percentile', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['backtest_run_id'], ['backtest_runs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pattern_detections_id'), 'pattern_detections', ['id'], unique=False)
    op.create_index(op.f('ix_pattern_detections_symbol'), 'pattern_detections', ['symbol'], unique=False)
    op.create_index(op.f('ix_pattern_detections_pattern_name'), 'pattern_detections', ['pattern_name'], unique=False)

    # Create signal_history table
    op.create_table(
        'signal_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('backtest_run_id', sa.Integer(), nullable=False),
        sa.Column('pattern_detection_id', sa.Integer(), nullable=True),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('signal_type', sa.String(length=20), nullable=False),
        sa.Column('signal_strength', sa.Numeric(precision=5, scale=2), nullable=False),
        sa.Column('signal_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('price_at_signal', sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column('volume_at_signal', sa.Integer(), nullable=True),
        sa.Column('entry_price', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('stop_loss', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('take_profit', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('position_size', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('risk_amount', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('source_strategy', sa.String(length=50), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=True),
        sa.Column('confidence_factors', JSON, nullable=True),
        sa.Column('executed', sa.Boolean(), server_default='false'),
        sa.Column('execution_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('execution_price', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('execution_delay_ms', sa.Integer(), nullable=True),
        sa.Column('market_trend', sa.String(length=10), nullable=True),
        sa.Column('volatility', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('rsi', sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column('macd_signal', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['backtest_run_id'], ['backtest_runs.id'], ),
        sa.ForeignKeyConstraint(['pattern_detection_id'], ['pattern_detections.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_signal_history_id'), 'signal_history', ['id'], unique=False)
    op.create_index(op.f('ix_signal_history_symbol'), 'signal_history', ['symbol'], unique=False)
    op.create_index(op.f('ix_signal_history_signal_time'), 'signal_history', ['signal_time'], unique=False)

    # Create backtest_trades table
    op.create_table(
        'backtest_trades',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('backtest_run_id', sa.Integer(), nullable=False),
        sa.Column('signal_id', sa.Integer(), nullable=True),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('side', sa.String(length=10), nullable=False),
        sa.Column('quantity', sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column('entry_price', sa.Numeric(precision=12, scale=6), nullable=False),
        sa.Column('entry_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('entry_reason', sa.String(length=100), nullable=True),
        sa.Column('exit_price', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('exit_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('exit_reason', sa.String(length=100), nullable=True),
        sa.Column('pnl', sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column('pnl_percentage', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('commission', sa.Numeric(precision=8, scale=2), server_default='0'),
        sa.Column('slippage', sa.Numeric(precision=8, scale=4), server_default='0'),
        sa.Column('initial_stop_loss', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('initial_take_profit', sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column('position_size_usd', sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column('risk_amount', sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column('risk_reward_ratio', sa.Numeric(precision=6, scale=2), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='OPEN'),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('max_favorable_excursion', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('max_adverse_excursion', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('strategy_state', JSON, nullable=True),
        sa.Column('market_conditions', JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['backtest_run_id'], ['backtest_runs.id'], ),
        sa.ForeignKeyConstraint(['signal_id'], ['signal_history.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_backtest_trades_id'), 'backtest_trades', ['id'], unique=False)
    op.create_index(op.f('ix_backtest_trades_symbol'), 'backtest_trades', ['symbol'], unique=False)
    op.create_index(op.f('ix_backtest_trades_entry_time'), 'backtest_trades', ['entry_time'], unique=False)


def downgrade():
    """Remove backtesting tables"""
    op.drop_table('backtest_trades')
    op.drop_table('signal_history')
    op.drop_table('pattern_detections')
    op.drop_table('backtest_runs')