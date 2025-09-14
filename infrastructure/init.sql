-- Initialize SwaggyStacks Trading Database

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create extension for advanced pattern matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Success message
SELECT 'Database initialization completed successfully' as message;