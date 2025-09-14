"""Add indicator performance and ML model tracking tables

Revision ID: 002
Revises: 001
Create Date: 2025-09-14 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Create indicator_performance table
    op.create_table(
        'indicator_performance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('indicator_name', sa.String(length=100), nullable=False),
        sa.Column('indicator_type', sa.String(length=20), nullable=False),
        sa.Column('total_signals', sa.Integer(), default=0),
        sa.Column('correct_signals', sa.Integer(), default=0),
        sa.Column('win_rate', sa.Numeric(precision=5, scale=4), default=0.0),
        sa.Column('total_return', sa.Numeric(precision=12, scale=4), default=0.0),
        sa.Column('sharpe_ratio', sa.Numeric(precision=8, scale=4), default=0.0),
        sa.Column('max_drawdown', sa.Numeric(precision=8, scale=4), default=0.0),
        sa.Column('avg_signal_strength', sa.Numeric(precision=5, scale=4), default=0.0),
        sa.Column('market_condition', sa.String(length=20), nullable=True),
        sa.Column('condition_win_rate', sa.Numeric(precision=5, scale=4), nullable=True),
        sa.Column('condition_avg_return', sa.Numeric(precision=10, scale=4), nullable=True),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=True),
        sa.Column('optimization_version', sa.Integer(), default=1),
        sa.Column('backtest_run_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.ForeignKeyConstraint(['backtest_run_id'], ['backtest_runs.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('indicator_name', 'indicator_type', 'period_start', 'period_end', 'market_condition',
                           name='uq_indicator_performance_unique')
    )

    # Create indexes for indicator_performance
    op.create_index('ix_indicator_performance_id', 'indicator_performance', ['id'])
    op.create_index('ix_indicator_performance_indicator_name', 'indicator_performance', ['indicator_name'])
    op.create_index('ix_indicator_performance_name_type', 'indicator_performance', ['indicator_name', 'indicator_type'])
    op.create_index('ix_indicator_performance_period', 'indicator_performance', ['period_start', 'period_end'])
    op.create_index('ix_indicator_performance_market_condition', 'indicator_performance', ['market_condition'])

    # Create ml_model_versions table
    op.create_table(
        'ml_model_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=False),
        sa.Column('is_production', sa.Boolean(), default=False),
        sa.Column('model_config', sa.JSON(), nullable=False),
        sa.Column('training_config', sa.JSON(), nullable=True),
        sa.Column('llm_provider', sa.String(length=50), nullable=True),
        sa.Column('llm_model_id', sa.String(length=100), nullable=True),
        sa.Column('prompt_template', sa.Text(), nullable=True),
        sa.Column('context_window', sa.Integer(), nullable=True),
        sa.Column('training_metrics', sa.JSON(), nullable=True),
        sa.Column('validation_metrics', sa.JSON(), nullable=True),
        sa.Column('production_metrics', sa.JSON(), nullable=True),
        sa.Column('training_start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('training_end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('features_used', sa.JSON(), nullable=True),
        sa.Column('deployed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('retired_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deployment_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name', 'version', name='uq_ml_model_version')
    )

    # Create indexes for ml_model_versions
    op.create_index('ix_ml_model_versions_id', 'ml_model_versions', ['id'])
    op.create_index('ix_ml_model_versions_model_name', 'ml_model_versions', ['model_name'])
    op.create_index('ix_ml_model_versions_active', 'ml_model_versions', ['is_active', 'is_production'])
    op.create_index('ix_ml_model_versions_model_type', 'ml_model_versions', ['model_type'])

    # Create ml_predictions table
    op.create_table(
        'ml_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_version_id', sa.Integer(), nullable=False),
        sa.Column('symbol', sa.String(length=10), nullable=False),
        sa.Column('prediction_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('prediction_horizon', sa.Integer(), nullable=False),
        sa.Column('predicted_direction', sa.String(length=10), nullable=False),
        sa.Column('predicted_return', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('confidence_score', sa.Numeric(precision=5, scale=4), nullable=False),
        sa.Column('individual_predictions', sa.JSON(), nullable=True),
        sa.Column('ensemble_weights', sa.JSON(), nullable=True),
        sa.Column('market_conditions', sa.JSON(), nullable=True),
        sa.Column('technical_indicators', sa.JSON(), nullable=True),
        sa.Column('actual_direction', sa.String(length=10), nullable=True),
        sa.Column('actual_return', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('outcome_recorded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('prediction_error', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('was_correct', sa.Boolean(), nullable=True),
        sa.Column('feedback_processed', sa.Boolean(), default=False),
        sa.Column('learning_weight', sa.Numeric(precision=5, scale=4), default=1.0),
        sa.Column('signal_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.ForeignKeyConstraint(['model_version_id'], ['ml_model_versions.id'], ),
        sa.ForeignKeyConstraint(['signal_id'], ['signal_history.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for ml_predictions
    op.create_index('ix_ml_predictions_id', 'ml_predictions', ['id'])
    op.create_index('ix_ml_predictions_symbol', 'ml_predictions', ['symbol'])
    op.create_index('ix_ml_predictions_prediction_time', 'ml_predictions', ['prediction_time'])
    op.create_index('ix_ml_predictions_symbol_time', 'ml_predictions', ['symbol', 'prediction_time'])
    op.create_index('ix_ml_predictions_outcome', 'ml_predictions', ['was_correct', 'feedback_processed'])

    # Create indicator_parameters table
    op.create_table(
        'indicator_parameters',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('indicator_name', sa.String(length=100), nullable=False),
        sa.Column('indicator_type', sa.String(length=20), nullable=False),
        sa.Column('market_condition', sa.String(length=20), nullable=True),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('parameter_hash', sa.String(length=64), nullable=False),
        sa.Column('backtest_performance', sa.JSON(), nullable=True),
        sa.Column('live_performance', sa.JSON(), nullable=True),
        sa.Column('optimization_method', sa.String(length=50), nullable=True),
        sa.Column('optimization_metric', sa.String(length=50), nullable=True),
        sa.Column('optimization_score', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('times_used', sa.Integer(), default=0),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_default', sa.Boolean(), default=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('version', sa.Integer(), default=1),
        sa.Column('previous_version_id', sa.Integer(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.ForeignKeyConstraint(['previous_version_id'], ['indicator_parameters.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('indicator_name', 'market_condition', 'parameter_hash',
                           name='uq_indicator_parameters_unique')
    )

    # Create indexes for indicator_parameters
    op.create_index('ix_indicator_parameters_id', 'indicator_parameters', ['id'])
    op.create_index('ix_indicator_parameters_indicator_name', 'indicator_parameters', ['indicator_name'])
    op.create_index('ix_indicator_parameters_lookup', 'indicator_parameters',
                    ['indicator_name', 'market_condition', 'is_active'])
    op.create_index('ix_indicator_parameters_hash', 'indicator_parameters', ['parameter_hash'])

    # Create parameter_optimizations table
    op.create_table(
        'parameter_optimizations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('optimization_id', sa.String(length=36), nullable=False),
        sa.Column('indicator_name', sa.String(length=100), nullable=True),
        sa.Column('model_version_id', sa.Integer(), nullable=True),
        sa.Column('optimization_type', sa.String(length=50), nullable=False),
        sa.Column('method', sa.String(length=50), nullable=False),
        sa.Column('metric', sa.String(length=50), nullable=False),
        sa.Column('parameter_space', sa.JSON(), nullable=False),
        sa.Column('constraints', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, default='PENDING'),
        sa.Column('progress', sa.Numeric(precision=5, scale=2), default=0.0),
        sa.Column('iterations_completed', sa.Integer(), default=0),
        sa.Column('total_iterations', sa.Integer(), nullable=True),
        sa.Column('best_parameters', sa.JSON(), nullable=True),
        sa.Column('best_score', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('all_results', sa.JSON(), nullable=True),
        sa.Column('convergence_history', sa.JSON(), nullable=True),
        sa.Column('improvement_percentage', sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column('baseline_score', sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('execution_time_seconds', sa.Integer(), nullable=True),
        sa.Column('market_condition', sa.String(length=20), nullable=True),
        sa.Column('backtest_period_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('backtest_period_end', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('warnings', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.ForeignKeyConstraint(['model_version_id'], ['ml_model_versions.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('optimization_id')
    )

    # Create indexes for parameter_optimizations
    op.create_index('ix_parameter_optimizations_id', 'parameter_optimizations', ['id'])
    op.create_index('ix_parameter_optimizations_optimization_id', 'parameter_optimizations', ['optimization_id'])
    op.create_index('ix_parameter_optimizations_status', 'parameter_optimizations', ['status', 'optimization_type'])
    op.create_index('ix_parameter_optimizations_indicator', 'parameter_optimizations', ['indicator_name'])


def downgrade():
    # Drop tables in reverse order due to foreign key constraints
    op.drop_table('parameter_optimizations')
    op.drop_table('indicator_parameters')
    op.drop_table('ml_predictions')
    op.drop_table('ml_model_versions')
    op.drop_table('indicator_performance')