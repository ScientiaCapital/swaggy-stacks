-- SuperBPE RAG System Database Schema
-- Optimized for 1536-dimensional embeddings and multi-agent learning

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Agent-specific pattern storage with 1536-dimensional SuperBPE embeddings
CREATE TABLE IF NOT EXISTS agent_patterns (
    id BIGSERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    strategy_name VARCHAR(100) NOT NULL,
    pattern_name VARCHAR(100) NOT NULL,
    pattern_embedding VECTOR(1536) NOT NULL,
    pattern_metadata JSONB DEFAULT '{}',
    market_data JSONB DEFAULT '{}',
    success_rate FLOAT DEFAULT 0.0,
    occurrence_count INT DEFAULT 0,
    total_profit_loss DECIMAL(18, 8) DEFAULT 0.0,
    avg_confidence FLOAT DEFAULT 0.0,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Create vector index for pattern similarity search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_patterns_embedding 
ON agent_patterns USING ivfflat (pattern_embedding vector_cosine_ops) 
WITH (lists = 1000);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_agent_patterns_type_success 
ON agent_patterns(agent_type, success_rate DESC, is_active);
CREATE INDEX IF NOT EXISTS idx_agent_patterns_strategy 
ON agent_patterns(strategy_name, occurrence_count DESC);

-- Agent decision history with context embeddings
CREATE TABLE IF NOT EXISTS agent_decisions (
    id BIGSERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    strategy_name VARCHAR(100),
    symbol VARCHAR(20) NOT NULL,
    decision VARCHAR(20) NOT NULL, -- BUY, SELL, HOLD
    confidence FLOAT NOT NULL,
    context_embedding VECTOR(1536),
    market_context JSONB NOT NULL,
    reasoning TEXT,
    similar_patterns_found INT DEFAULT 0,
    outcome VARCHAR(20), -- WIN, LOSS, BREAKEVEN (set later)
    outcome_pnl DECIMAL(18, 8),
    outcome_accuracy FLOAT, -- How accurate the confidence was
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('agent_decisions', 'created_at', 
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- Vector index for decision context search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_decisions_context_embedding 
ON agent_decisions USING ivfflat (context_embedding vector_cosine_ops) 
WITH (lists = 500);

-- Market knowledge base with SuperBPE embeddings
CREATE TABLE IF NOT EXISTS market_knowledge (
    id BIGSERIAL PRIMARY KEY,
    source_type VARCHAR(50) NOT NULL, -- 'news', 'social', 'analysis', 'pattern'
    content_text TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    symbols TEXT[], -- Array of related symbols
    metadata JSONB DEFAULT '{}',
    confidence_score FLOAT DEFAULT 0.0,
    relevance_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_timestamp TIMESTAMP WITH TIME ZONE,
    is_validated BOOLEAN DEFAULT FALSE
);

-- Vector index for market knowledge search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_knowledge_embedding 
ON market_knowledge USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 2000);

-- GIN index for symbol array searches
CREATE INDEX IF NOT EXISTS idx_market_knowledge_symbols 
ON market_knowledge USING gin (symbols);

-- Agent performance metrics (time-series)
CREATE TABLE IF NOT EXISTS agent_performance (
    id BIGSERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    strategy_name VARCHAR(100),
    measurement_period VARCHAR(20) NOT NULL, -- 'daily', 'weekly', 'monthly'
    total_decisions INT DEFAULT 0,
    correct_predictions INT DEFAULT 0,
    accuracy_rate FLOAT DEFAULT 0.0,
    avg_confidence FLOAT DEFAULT 0.0,
    total_pnl DECIMAL(18, 8) DEFAULT 0.0,
    sharpe_ratio FLOAT DEFAULT 0.0,
    max_drawdown FLOAT DEFAULT 0.0,
    learning_progress FLOAT DEFAULT 0.0, -- How much the agent has improved
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('agent_performance', 'measured_at', 
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE);

-- Agent learning sessions (when agents update their models)
CREATE TABLE IF NOT EXISTS agent_learning_sessions (
    id BIGSERIAL PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    session_type VARCHAR(50) NOT NULL, -- 'pattern_update', 'weight_adjustment', 'new_learning'
    patterns_learned INT DEFAULT 0,
    patterns_validated INT DEFAULT 0,
    performance_improvement FLOAT DEFAULT 0.0,
    memory_usage_mb INT DEFAULT 0,
    processing_time_seconds INT DEFAULT 0,
    session_metadata JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Multi-agent consensus decisions
CREATE TABLE IF NOT EXISTS consensus_decisions (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    final_decision VARCHAR(20) NOT NULL,
    consensus_confidence FLOAT NOT NULL,
    participating_agents JSONB NOT NULL, -- Array of agent contributions
    market_context JSONB NOT NULL,
    weight_distribution JSONB NOT NULL, -- How much each agent contributed
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance queries
CREATE INDEX IF NOT EXISTS idx_agent_performance_type_period 
ON agent_performance(agent_type, measurement_period, measured_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_symbol_outcome 
ON agent_decisions(symbol, outcome, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_consensus_decisions_symbol 
ON consensus_decisions(symbol, created_at DESC);

-- Views for common queries
CREATE OR REPLACE VIEW agent_performance_summary AS
SELECT 
    agent_type,
    strategy_name,
    COUNT(*) as total_decisions,
    AVG(confidence) as avg_confidence,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as win_rate,
    AVG(outcome_pnl) as avg_pnl,
    SUM(outcome_pnl) as total_pnl,
    MAX(created_at) as last_decision
FROM agent_decisions 
WHERE outcome IS NOT NULL 
GROUP BY agent_type, strategy_name;

-- Function to get similar patterns for an agent
CREATE OR REPLACE FUNCTION find_similar_patterns(
    p_agent_type VARCHAR(50),
    p_pattern_embedding VECTOR(1536),
    p_similarity_threshold FLOAT DEFAULT 0.8,
    p_limit INT DEFAULT 10
) RETURNS TABLE (
    id BIGINT,
    pattern_name VARCHAR(100),
    similarity FLOAT,
    success_rate FLOAT,
    occurrence_count INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ap.id,
        ap.pattern_name,
        1 - (ap.pattern_embedding <-> p_pattern_embedding) as similarity,
        ap.success_rate,
        ap.occurrence_count
    FROM agent_patterns ap
    WHERE ap.agent_type = p_agent_type 
    AND ap.is_active = TRUE
    AND 1 - (ap.pattern_embedding <-> p_pattern_embedding) >= p_similarity_threshold
    ORDER BY ap.pattern_embedding <-> p_pattern_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to update agent pattern success rate
CREATE OR REPLACE FUNCTION update_pattern_success(
    p_pattern_id BIGINT,
    p_outcome VARCHAR(20),
    p_pnl DECIMAL(18, 8)
) RETURNS VOID AS $$
BEGIN
    UPDATE agent_patterns 
    SET occurrence_count = occurrence_count + 1,
        total_profit_loss = total_profit_loss + p_pnl,
        success_rate = CASE 
            WHEN p_outcome = 'WIN' THEN 
                (success_rate * (occurrence_count - 1) + 1.0) / occurrence_count
            ELSE 
                (success_rate * (occurrence_count - 1)) / occurrence_count
        END
    WHERE id = p_pattern_id;
END;
$$ LANGUAGE plpgsql;