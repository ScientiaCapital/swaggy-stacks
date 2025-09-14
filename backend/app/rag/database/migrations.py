"""
Database migration manager for SuperBPE RAG System
Ensures proper setup order and dependency management
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import asyncpg

logger = logging.getLogger(__name__)


class RAGDatabaseMigrator:
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        if not self.connection_string:
            raise ValueError("DATABASE_URL environment variable required")

        self.migration_order = [
            "enable_extensions",
            "create_base_tables",
            "create_vector_indexes",
            "create_hypertables",
            "create_performance_indexes",
            "create_functions_and_views",
            "insert_seed_data",
        ]

    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections"""
        conn = await asyncpg.connect(self.connection_string)
        try:
            yield conn
        finally:
            await conn.close()

    async def run_migrations(self) -> Dict[str, bool]:
        """Run all migrations in proper order"""
        results = {}

        logger.info("🚀 Starting RAG database migrations...")

        async with self.get_connection() as conn:
            for migration_name in self.migration_order:
                try:
                    logger.info(f"⏳ Running migration: {migration_name}")
                    await self._run_migration(conn, migration_name)
                    results[migration_name] = True
                    logger.info(f"✅ Migration {migration_name} completed")
                except Exception as e:
                    logger.error(f"❌ Migration {migration_name} failed: {e}")
                    results[migration_name] = False
                    break  # Stop on first failure

        return results

    async def _run_migration(
        self, conn: asyncpg.Connection, migration_name: str
    ) -> None:
        """Run a specific migration"""
        if migration_name == "enable_extensions":
            await self._enable_extensions(conn)
        elif migration_name == "create_base_tables":
            await self._create_base_tables(conn)
        elif migration_name == "create_vector_indexes":
            await self._create_vector_indexes(conn)
        elif migration_name == "create_hypertables":
            await self._create_hypertables(conn)
        elif migration_name == "create_performance_indexes":
            await self._create_performance_indexes(conn)
        elif migration_name == "create_functions_and_views":
            await self._create_functions_and_views(conn)
        elif migration_name == "insert_seed_data":
            await self._insert_seed_data(conn)

    async def _enable_extensions(self, conn: asyncpg.Connection) -> None:
        """Enable required PostgreSQL extensions"""
        extensions = [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE EXTENSION IF NOT EXISTS timescaledb",
            "CREATE EXTENSION IF NOT EXISTS pg_stat_statements",
        ]

        for ext_sql in extensions:
            try:
                await conn.execute(ext_sql)
                logger.info(f"  📦 Enabled extension: {ext_sql.split()[-1]}")
            except Exception as e:
                logger.warning(f"  ⚠️  Extension warning: {e}")

    async def _create_base_tables(self, conn: asyncpg.Connection) -> None:
        """Create base tables from schema file"""
        schema_file = Path(__file__).parent / "schema.sql"

        with open(schema_file, "r") as f:
            schema_sql = f.read()

        # Extract and execute CREATE TABLE statements
        current_statement = ""
        in_table_creation = False

        for line in schema_sql.split("\n"):
            line = line.strip()

            # Skip comments and empty lines
            if line.startswith("--") or not line:
                continue

            # Skip extension and index creation for now
            if (
                line.startswith("CREATE EXTENSION")
                or line.startswith("CREATE INDEX")
                or line.startswith("CREATE VIEW")
                or line.startswith("CREATE OR REPLACE FUNCTION")
                or line.startswith("SELECT create_hypertable")
            ):
                continue

            # Handle table creation
            if line.startswith("CREATE TABLE"):
                in_table_creation = True
                current_statement = line
            elif in_table_creation:
                current_statement += "\n" + line
                if line.endswith(");"):
                    in_table_creation = False
                    try:
                        await conn.execute(current_statement)
                        table_name = current_statement.split()[5]  # Extract table name
                        logger.info(f"  🗄️  Created table: {table_name}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Table creation warning: {e}")
                    current_statement = ""

    async def _create_vector_indexes(self, conn: asyncpg.Connection) -> None:
        """Create vector indexes for similarity search"""
        vector_indexes = [
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_patterns_embedding
            ON agent_patterns USING ivfflat (pattern_embedding vector_cosine_ops)
            WITH (lists = 1000)
            """,
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_decisions_context_embedding
            ON agent_decisions USING ivfflat (context_embedding vector_cosine_ops)
            WITH (lists = 500)
            """,
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_knowledge_embedding
            ON market_knowledge USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 2000)
            """,
        ]

        for index_sql in vector_indexes:
            try:
                await conn.execute(index_sql)
                logger.info("  🔍 Created vector index")
            except Exception as e:
                logger.warning(f"  ⚠️  Vector index warning: {e}")

    async def _create_hypertables(self, conn: asyncpg.Connection) -> None:
        """Create TimescaleDB hypertables"""
        hypertables = [
            {
                "table": "agent_decisions",
                "time_column": "created_at",
                "sql": "SELECT create_hypertable('agent_decisions', 'created_at', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE)",
            },
            {
                "table": "agent_performance",
                "time_column": "measured_at",
                "sql": "SELECT create_hypertable('agent_performance', 'measured_at', chunk_time_interval => INTERVAL '1 day', if_not_exists => TRUE)",
            },
        ]

        for hypertable in hypertables:
            try:
                await conn.execute(hypertable["sql"])
                logger.info(f"  📊 Created hypertable: {hypertable['table']}")
            except Exception as e:
                logger.warning(
                    f"  ⚠️  Hypertable warning for {hypertable['table']}: {e}"
                )

    async def _create_performance_indexes(self, conn: asyncpg.Connection) -> None:
        """Create performance indexes for common queries"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_agent_patterns_type_success ON agent_patterns(agent_type, success_rate DESC, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_agent_patterns_strategy ON agent_patterns(strategy_name, occurrence_count DESC)",
            "CREATE INDEX IF NOT EXISTS idx_market_knowledge_symbols ON market_knowledge USING gin (symbols)",
            "CREATE INDEX IF NOT EXISTS idx_agent_performance_type_period ON agent_performance(agent_type, measurement_period, measured_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_agent_decisions_symbol_outcome ON agent_decisions(symbol, outcome, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_consensus_decisions_symbol ON consensus_decisions(symbol, created_at DESC)",
        ]

        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
                logger.info("  📈 Created performance index")
            except Exception as e:
                logger.warning(f"  ⚠️  Performance index warning: {e}")

    async def _create_functions_and_views(self, conn: asyncpg.Connection) -> None:
        """Create database functions and views"""

        # Create the performance summary view
        view_sql = """
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
        GROUP BY agent_type, strategy_name
        """

        # Create functions
        similarity_function = """
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
        $$ LANGUAGE plpgsql
        """

        update_function = """
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
        $$ LANGUAGE plpgsql
        """

        functions = [view_sql, similarity_function, update_function]

        for func_sql in functions:
            try:
                await conn.execute(func_sql)
                logger.info("  ⚙️  Created function/view")
            except Exception as e:
                logger.warning(f"  ⚠️  Function/view warning: {e}")

    async def _insert_seed_data(self, conn: asyncpg.Connection) -> None:
        """Insert seed/test data for validation"""

        # Create test embedding (1536 dimensions)
        test_embedding = [0.1] * 384 + [0.0] * 1152  # Pad to 1536 dimensions

        try:
            # Insert test market knowledge
            await conn.execute(
                """
                INSERT INTO market_knowledge (source_type, content_text, embedding, symbols, confidence_score)
                VALUES ('test', 'Test market knowledge for system validation', $1, ARRAY['TEST'], 0.95)
                ON CONFLICT DO NOTHING
            """,
                test_embedding,
            )

            # Insert test agent pattern
            await conn.execute(
                """
                INSERT INTO agent_patterns (
                    agent_type, strategy_name, pattern_name, pattern_embedding,
                    success_rate, occurrence_count, is_active
                )
                VALUES ('test_agent', 'test_strategy', 'bullish_momentum', $1, 0.75, 100, TRUE)
                ON CONFLICT DO NOTHING
            """,
                test_embedding,
            )

            logger.info("  🌱 Inserted seed data")

        except Exception as e:
            logger.warning(f"  ⚠️  Seed data warning: {e}")

    async def validate_setup(self) -> Dict[str, Any]:
        """Validate database setup and return health metrics"""
        validation_results = {}

        async with self.get_connection() as conn:
            try:
                # Check extensions
                extensions = await conn.fetch(
                    "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'timescaledb')"
                )
                validation_results["extensions"] = [
                    ext["extname"] for ext in extensions
                ]

                # Check tables exist
                tables = await conn.fetch(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    AND table_name IN ('agent_patterns', 'agent_decisions', 'market_knowledge',
                                     'agent_performance', 'agent_learning_sessions', 'consensus_decisions')
                """
                )
                validation_results["tables"] = [t["table_name"] for t in tables]

                # Check vector indexes
                vector_indexes = await conn.fetch(
                    """
                    SELECT indexname FROM pg_indexes
                    WHERE indexname LIKE '%embedding%'
                """
                )
                validation_results["vector_indexes"] = [
                    idx["indexname"] for idx in vector_indexes
                ]

                # Test vector similarity search
                test_vector = [0.1] * 384 + [0.0] * 1152
                result = await conn.fetchrow(
                    """
                    SELECT id, content_text, embedding <-> $1 as distance
                    FROM market_knowledge
                    ORDER BY embedding <-> $1
                    LIMIT 1
                """,
                    test_vector,
                )
                validation_results["vector_search"] = result is not None
                if result:
                    validation_results["test_distance"] = float(result["distance"])

                # Check hypertables
                try:
                    hypertables = await conn.fetch(
                        """
                        SELECT hypertable_name FROM timescaledb_information.hypertables
                    """
                    )
                    validation_results["hypertables"] = [
                        ht["hypertable_name"] for ht in hypertables
                    ]
                except Exception:
                    validation_results["hypertables"] = []

                # Test pattern similarity function
                try:
                    pattern_test = await conn.fetch(
                        """
                        SELECT * FROM find_similar_patterns('test_agent', $1, 0.5, 5)
                    """,
                        test_vector,
                    )
                    validation_results["similarity_function"] = len(pattern_test) > 0
                except Exception as e:
                    validation_results["similarity_function"] = False
                    validation_results["function_error"] = str(e)

            except Exception as e:
                validation_results["validation_error"] = str(e)

        return validation_results


async def main():
    """CLI interface for migrations"""

    # Check if we have the required environment
    if not os.getenv("DATABASE_URL"):
        print("❌ DATABASE_URL environment variable required")
        print(
            "💡 Example: export DATABASE_URL='postgresql://user:pass@localhost/swaggy_rag'"
        )
        sys.exit(1)

    migrator = RAGDatabaseMigrator()

    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        # Validation mode
        print("🔍 Running database validation...")
        results = await migrator.validate_setup()
        print("\n📊 Database Validation Results:")
        for key, value in results.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items - {value}")
            else:
                print(f"  {key}: {value}")
    else:
        # Migration mode
        results = await migrator.run_migrations()

        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)

        print(f"\n📊 Migration Results: {success_count}/{total_count} successful")
        for migration, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {migration}")

        if success_count == total_count:
            print("\n🎉 All migrations completed successfully!")

            # Run validation
            print("\n🔍 Running post-migration validation...")
            validation_results = await migrator.validate_setup()
            print(
                f"  Extensions: {len(validation_results.get('extensions', []))} enabled"
            )
            print(f"  Tables: {len(validation_results.get('tables', []))} created")
            print(
                f"  Vector indexes: {len(validation_results.get('vector_indexes', []))} active"
            )
            print(
                f"  Hypertables: {len(validation_results.get('hypertables', []))} configured"
            )
            print(
                f"  Vector search: {'✅' if validation_results.get('vector_search') else '❌'}"
            )
            print(
                f"  Similarity function: {'✅' if validation_results.get('similarity_function') else '❌'}"
            )
        else:
            print(f"\n❌ {total_count - success_count} migrations failed")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
