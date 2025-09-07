# Agent Intelligence Infrastructure

A comprehensive AI-powered infrastructure for autonomous trading agents, providing persistent memory, retrieval-augmented generation (RAG), dynamic tool management, and intelligent context building capabilities.

## Overview

The Agent Intelligence Infrastructure consists of four core components working together to enable sophisticated AI trading agents:

1. **Memory Manager** - Persistent memory with semantic search
2. **RAG Service** - Retrieval-Augmented Generation for trading contexts
3. **Tool Registry** - Dynamic tool management and execution
4. **Context Builder** - Intelligent context assembly for decision-making

## Quick Start

```python
import asyncio
from app.rag.services.memory_manager import AgentMemoryManager
from app.rag.services.rag_service import TradingRAGService
from app.rag.services.tool_registry import TradingToolRegistry
from app.rag.services.context_builder import TradingContextBuilder

async def main():
    # Initialize embedding service (mock for example)
    from app.rag.services.embedding_factory import EmbeddingFactory
    embedding_service = await EmbeddingFactory.create_service("mock")
    
    # Initialize all components
    memory_manager = AgentMemoryManager(embedding_service=embedding_service)
    await memory_manager.initialize()
    
    rag_service = TradingRAGService(
        embedding_service=embedding_service,
        memory_manager=memory_manager
    )
    await rag_service.initialize()
    
    tool_registry = TradingToolRegistry()
    await tool_registry.initialize()
    
    context_builder = TradingContextBuilder(
        memory_manager=memory_manager,
        rag_service=rag_service,
        tool_registry=tool_registry
    )
    await context_builder.initialize()
    
    print("Agent Intelligence Infrastructure initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Memory Manager    │    │    RAG Service      │    │   Tool Registry     │
│                     │    │                     │    │                     │
│ • Vector Storage    │    │ • Document Store    │    │ • Tool Discovery    │
│ • Semantic Search   │    │ • Semantic Search   │    │ • Execution Engine  │
│ • Pattern Learning  │    │ • Context Retrieval │    │ • Permission System │
│ • Memory Pruning    │    │ • Query Expansion   │    │ • Result Validation │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           └───────────────────────────┼───────────────────────────┘
                                       │
                          ┌─────────────────────┐
                          │  Context Builder    │
                          │                     │
                          │ • Context Assembly  │
                          │ • Prioritization    │
                          │ • Summarization     │
                          │ • Template System   │
                          └─────────────────────┘
                                       │
                          ┌─────────────────────┐
                          │  Trading Agents     │
                          │                     │
                          │ • Decision Making   │
                          │ • Strategy Execute  │
                          │ • Learning & Adapt  │
                          └─────────────────────┘
```

## Core Components

### Memory Manager
Provides persistent memory capabilities with semantic search and pattern recognition.

**Key Features:**
- Vector-based similarity search
- Multiple memory types (decisions, patterns, market states)
- Automatic importance scoring
- Memory consolidation and pruning
- Cross-session persistence

### RAG Service
Retrieval-Augmented Generation system specialized for trading contexts.

**Key Features:**
- Market-specific document chunking
- Financial entity extraction
- Temporal relevance scoring
- Multi-modal search (technical, news, sentiment)
- Query expansion with financial synonyms

### Tool Registry
Dynamic tool management system for trading capabilities.

**Key Features:**
- Runtime tool discovery
- Parameter validation
- Permission-based access control
- Execution logging and monitoring
- Async and sync tool support

### Context Builder
Intelligent context assembly system for agent decision-making.

**Key Features:**
- Component prioritization
- Context window management
- Template-based assembly
- Temporal filtering
- Adaptive sizing based on decision complexity

## Environment Setup

### Required Dependencies

```bash
# Core dependencies
pip install numpy pandas asyncio aioredis asyncpg
pip install langchain langchain-community
pip install sentence-transformers faiss-cpu

# Optional dependencies for advanced features
pip install pinecone-client  # For production vector storage
pip install openai          # For OpenAI embeddings
pip install transformers    # For local embeddings
```

### Environment Variables

```bash
# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/swaggy_stacks
REDIS_URL=redis://localhost:6379

# Embedding Service Configuration
EMBEDDING_SERVICE_TYPE=local  # Options: local, openai, mock
OPENAI_API_KEY=your_openai_key_here

# Vector Storage Configuration
VECTOR_STORAGE_TYPE=local  # Options: local, pinecone
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_environment
```

### Database Schema

Run the following to set up the required database tables:

```sql
-- Execute the migration scripts
python -m alembic upgrade head

-- Or manually create tables (see schema in migration files)
```

## Usage Examples

See the [examples directory](./examples/) for comprehensive usage examples:

- [Basic Agent Setup](./examples/basic_agent_example.py)
- [Custom Tool Creation](./examples/custom_tool_example.py)
- [Workflow Integration](./examples/workflow_example.py)

## Performance Considerations

### Memory Management
- Configure appropriate memory pruning thresholds
- Monitor vector storage size and performance
- Use memory consolidation for pattern recognition

### RAG Optimization
- Optimize chunk sizes for your use case (default: 500 tokens)
- Consider caching for frequently accessed documents
- Use appropriate embedding dimensions (384 for speed, 768 for accuracy)

### Tool Execution
- Set appropriate timeouts for tools
- Monitor tool execution metrics
- Use permission levels to control access

### Context Building
- Configure context window limits based on model capacity
- Use templates to standardize context structure
- Monitor context building latency

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest backend/tests/rag/ -v

# Run specific component tests
pytest backend/tests/rag/test_memory_manager.py -v
pytest backend/tests/rag/test_rag_service.py -v
pytest backend/tests/rag/test_tool_registry.py -v
pytest backend/tests/rag/test_context_builder.py -v

# Run integration tests
pytest backend/tests/rag/integration/ -v

# Run with coverage
pytest backend/tests/rag/ --cov=app.rag --cov-report=html
```

## Monitoring and Observability

### Metrics Collection

The system automatically collects metrics for:
- Memory operations (store, retrieve, search)
- RAG queries and response times
- Tool execution statistics
- Context building performance

### Logging

Configure structured logging:

```python
import structlog

# Configure logger
logger = structlog.get_logger(__name__)

# Use in components
logger.info("Memory stored", memory_id="mem_001", agent_id="agent_1")
```

### Health Checks

Monitor system health:

```python
# Check component health
health_status = {
    "memory_manager": await memory_manager.health_check(),
    "rag_service": await rag_service.health_check(),
    "tool_registry": await tool_registry.health_check(),
    "context_builder": await context_builder.health_check()
}
```

## Contributing

1. Follow the existing code patterns and architecture
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Use type hints and docstrings
5. Run the full test suite before submitting

## License

This project is part of the SwaggyStacks trading system and follows the same license terms.

## Support

For questions or issues:
1. Check the [API Reference](./api_reference.md)
2. Review [troubleshooting guide](./troubleshooting.md)
3. Examine the test files for usage examples
4. Create an issue in the project repository