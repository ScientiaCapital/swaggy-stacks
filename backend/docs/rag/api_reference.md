# Agent Intelligence Infrastructure API Reference

Complete API documentation for all components of the Agent Intelligence Infrastructure.

## Memory Manager API

### AgentMemoryManager

The core memory management component providing persistent storage and semantic search.

#### Class Definition

```python
class AgentMemoryManager:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_dimension: int = 384,
        similarity_threshold: float = 0.7
    )
```

#### Methods

##### `initialize() -> None`

Initialize the memory manager and set up database connections.

```python
await memory_manager.initialize()
```

##### `store_memory(memory: Memory) -> bool`

Store a memory with vector embedding for semantic search.

**Parameters:**
- `memory` (Memory): Memory object containing content and metadata

**Returns:**
- `bool`: True if stored successfully

**Example:**
```python
from app.rag.services.memory_manager import Memory, MemoryType

memory = Memory(
    memory_id="unique_id",
    agent_id="agent_1",
    content="AAPL showed bullish momentum at $150",
    memory_type=MemoryType.PATTERN,
    metadata={"symbol": "AAPL", "price": 150.0},
    importance=0.8
)

success = await memory_manager.store_memory(memory)
```

##### `retrieve_memory(memory_id: str, agent_id: str) -> Optional[Memory]`

Retrieve a specific memory by ID.

**Parameters:**
- `memory_id` (str): Unique memory identifier
- `agent_id` (str): Agent identifier for security

**Returns:**
- `Optional[Memory]`: Memory object or None if not found

##### `search_memories(query: MemoryQuery) -> List[MemorySearchResult]`

Search memories using semantic similarity.

**Parameters:**
- `query` (MemoryQuery): Search query with filters and constraints

**Returns:**
- `List[MemorySearchResult]`: Ranked list of similar memories

**Example:**
```python
from app.rag.services.memory_manager import MemoryQuery, MemoryType

query = MemoryQuery(
    agent_id="agent_1",
    query_text="bullish patterns in tech stocks",
    memory_types=[MemoryType.PATTERN],
    limit=10,
    min_importance=0.7,
    time_range_days=30
)

results = await memory_manager.search_memories(query)
for result in results:
    print(f"Similarity: {result.similarity_score:.2f}")
    print(f"Content: {result.memory.content}")
```

##### `consolidate_memories(agent_id: str, topic: str) -> Optional[Memory]`

Consolidate related memories into a single pattern memory.

**Parameters:**
- `agent_id` (str): Agent identifier
- `topic` (str): Topic or theme for consolidation

**Returns:**
- `Optional[Memory]`: Consolidated memory or None if insufficient data

##### `prune_memories(agent_id: str, min_importance: float = 0.3, max_age_days: int = 180) -> int`

Remove old or low-importance memories to manage storage.

**Parameters:**
- `agent_id` (str): Agent identifier
- `min_importance` (float): Minimum importance threshold
- `max_age_days` (int): Maximum age in days

**Returns:**
- `int`: Number of memories pruned

##### `update_memory_importance(memory_id: str, agent_id: str, new_importance: float, outcome_feedback: str) -> bool`

Update memory importance based on outcome feedback.

**Parameters:**
- `memory_id` (str): Memory identifier
- `agent_id` (str): Agent identifier
- `new_importance` (float): New importance score (0.0-1.0)
- `outcome_feedback` (str): Description of outcome

**Returns:**
- `bool`: True if updated successfully

### Data Classes

#### Memory

```python
@dataclass
class Memory:
    memory_id: str
    agent_id: str
    content: str
    memory_type: MemoryType
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory': ...
```

#### MemoryType

```python
class MemoryType(Enum):
    DECISION = "decision"
    PATTERN = "pattern"
    MARKET_STATE = "market_state"
    OUTCOME = "outcome"
    CONSOLIDATED = "consolidated"
```

#### MemoryQuery

```python
@dataclass
class MemoryQuery:
    agent_id: str
    query_text: str
    memory_types: Optional[List[MemoryType]] = None
    limit: int = 10
    min_importance: float = 0.0
    time_range_days: Optional[int] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
```

## RAG Service API

### TradingRAGService

Specialized RAG system for trading contexts with market-aware document processing.

#### Class Definition

```python
class TradingRAGService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        memory_manager: AgentMemoryManager,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    )
```

#### Methods

##### `initialize() -> None`

Initialize the RAG service and set up document storage.

##### `add_document(content: str, document_type: DocumentType, metadata: Dict[str, Any] = None, source_id: str = None) -> str`

Add a document to the knowledge base with automatic chunking and embedding.

**Parameters:**
- `content` (str): Document content
- `document_type` (DocumentType): Type of document
- `metadata` (Dict): Document metadata
- `source_id` (str): Optional source identifier

**Returns:**
- `str`: Document ID

**Example:**
```python
from app.rag.services.rag_service import DocumentType

doc_id = await rag_service.add_document(
    content="AAPL technical analysis shows strong momentum...",
    document_type=DocumentType.TECHNICAL_ANALYSIS,
    metadata={
        "symbol": "AAPL",
        "analyst": "TradingBot",
        "confidence": 0.85,
        "date": "2024-01-15"
    },
    source_id="tech_analysis_001"
)
```

##### `search(query: RAGQuery) -> List[RAGResult]`

Search the knowledge base using semantic similarity.

**Parameters:**
- `query` (RAGQuery): Search query with context and filters

**Returns:**
- `List[RAGResult]`: Ranked search results

**Example:**
```python
from app.rag.services.rag_service import RAGQuery, DocumentType

query = RAGQuery(
    query_text="bullish momentum patterns in AAPL",
    context={"symbol": "AAPL", "market_regime": "bull"},
    document_types=[DocumentType.TECHNICAL_ANALYSIS, DocumentType.PATTERN_ANALYSIS],
    max_results=10,
    min_relevance_score=0.7
)

results = await rag_service.search(query)
for result in results:
    print(f"Relevance: {result.relevance_score:.2f}")
    print(f"Content: {result.content}")
```

##### `batch_add_documents(documents: List[Dict[str, Any]]) -> List[str]`

Add multiple documents in a batch operation.

**Parameters:**
- `documents` (List[Dict]): List of document data

**Returns:**
- `List[str]`: List of document IDs

##### `update_document(document_id: str, new_content: str, metadata: Dict[str, Any] = None) -> bool`

Update an existing document with new content.

**Parameters:**
- `document_id` (str): Document identifier
- `new_content` (str): Updated content
- `metadata` (Dict): Updated metadata

**Returns:**
- `bool`: True if updated successfully

### Data Classes

#### RAGQuery

```python
@dataclass
class RAGQuery:
    query_text: str
    context: Dict[str, Any] = field(default_factory=dict)
    document_types: Optional[List[DocumentType]] = None
    max_results: int = 10
    min_relevance_score: float = 0.0
    time_range_days: Optional[int] = None
    include_metadata: bool = True
```

#### RAGResult

```python
@dataclass
class RAGResult:
    content: str
    relevance_score: float
    document_type: DocumentType
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_id: Optional[str] = None
    chunk_id: Optional[str] = None
```

#### DocumentType

```python
class DocumentType(Enum):
    TECHNICAL_ANALYSIS = "technical_analysis"
    MARKET_NEWS = "market_news"
    PATTERN_ANALYSIS = "pattern_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    EARNINGS_REPORT = "earnings_report"
    RESEARCH_REPORT = "research_report"
    MARKET_ANALYSIS = "market_analysis"
```

## Tool Registry API

### TradingToolRegistry

Dynamic tool management system for trading capabilities.

#### Class Definition

```python
class TradingToolRegistry:
    def __init__(self, max_concurrent_executions: int = 10)
```

#### Methods

##### `initialize() -> None`

Initialize the tool registry and load default tools.

##### `register_tool(tool_def: ToolDefinition, allow_override: bool = False) -> bool`

Register a new tool in the registry.

**Parameters:**
- `tool_def` (ToolDefinition): Tool definition with implementation
- `allow_override` (bool): Allow overriding existing tool

**Returns:**
- `bool`: True if registered successfully

**Example:**
```python
from app.rag.services.tool_registry import ToolDefinition, ToolCategory

def calculate_rsi(prices: List[float], period: int = 14) -> Dict[str, float]:
    # RSI calculation implementation
    return {"rsi": 65.5, "signal": "neutral"}

rsi_tool = ToolDefinition(
    name="rsi_calculator",
    description="Calculate RSI technical indicator",
    category=ToolCategory.TECHNICAL_INDICATORS,
    parameters={
        "prices": {"type": "array", "items": {"type": "number"}, "required": True},
        "period": {"type": "integer", "default": 14, "minimum": 1}
    },
    implementation=calculate_rsi,
    async_capable=False,
    timeout_seconds=30
)

success = await tool_registry.register_tool(rsi_tool)
```

##### `execute_tool(tool_name: str, parameters: Dict[str, Any], context: ToolExecutionContext) -> ToolResult`

Execute a registered tool with given parameters.

**Parameters:**
- `tool_name` (str): Name of tool to execute
- `parameters` (Dict): Tool parameters
- `context` (ToolExecutionContext): Execution context with permissions

**Returns:**
- `ToolResult`: Execution result with success status and data

**Example:**
```python
from app.rag.services.tool_registry import ToolExecutionContext

context = ToolExecutionContext(
    agent_id="strategy_agent",
    session_id="trading_session_001",
    permissions=[PermissionLevel.READ_MARKET_DATA]
)

result = await tool_registry.execute_tool(
    tool_name="rsi_calculator",
    parameters={"prices": [100, 101, 102, 103, 104], "period": 14},
    context=context
)

if result.success:
    print(f"RSI: {result.data['rsi']}")
else:
    print(f"Error: {result.error}")
```

##### `discover_tools(category: Optional[ToolCategory] = None, required_permissions: Optional[List[PermissionLevel]] = None) -> List[ToolDefinition]`

Discover available tools with optional filtering.

**Parameters:**
- `category` (Optional[ToolCategory]): Filter by tool category
- `required_permissions` (Optional[List]): Filter by required permissions

**Returns:**
- `List[ToolDefinition]`: List of available tools

##### `get_tool_statistics(tool_name: str) -> Dict[str, Any]`

Get usage statistics for a specific tool.

**Parameters:**
- `tool_name` (str): Tool name

**Returns:**
- `Dict[str, Any]`: Statistics including execution count, success rate, etc.

### Data Classes

#### ToolDefinition

```python
@dataclass
class ToolDefinition:
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, Any] = field(default_factory=dict)
    permissions: List[PermissionLevel] = field(default_factory=list)
    implementation: Optional[Callable] = None
    async_capable: bool = False
    timeout_seconds: int = 30
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDefinition': ...
```

#### ToolResult

```python
@dataclass
class ToolResult:
    tool_name: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[datetime] = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### ToolCategory

```python
class ToolCategory(Enum):
    MARKET_DATA = "market_data"
    TECHNICAL_INDICATORS = "technical_indicators"
    ORDER_EXECUTION = "order_execution"
    RISK_MANAGEMENT = "risk_management"
    ANALYTICS = "analytics"
    PATTERN_RECOGNITION = "pattern_recognition"
```

## Context Builder API

### TradingContextBuilder

Intelligent context assembly system for agent decision-making.

#### Class Definition

```python
class TradingContextBuilder:
    def __init__(
        self,
        memory_manager: AgentMemoryManager,
        rag_service: TradingRAGService,
        tool_registry: TradingToolRegistry,
        max_context_tokens: int = 4000,
        default_memory_limit: int = 10
    )
```

#### Methods

##### `initialize() -> None`

Initialize the context builder and register default templates.

##### `build_context(agent_id: str, session_id: str, decision_type: str, market_data: Dict[str, Any], tool_results: Optional[List[ToolResult]] = None, template_name: Optional[str] = None) -> DecisionContext`

Build a comprehensive context for agent decision-making.

**Parameters:**
- `agent_id` (str): Agent identifier
- `session_id` (str): Session identifier
- `decision_type` (str): Type of decision being made
- `market_data` (Dict): Current market data
- `tool_results` (Optional[List]): Recent tool execution results
- `template_name` (Optional[str]): Context template to use

**Returns:**
- `DecisionContext`: Assembled context for decision-making

**Example:**
```python
context = await context_builder.build_context(
    agent_id="strategy_agent",
    session_id="session_001",
    decision_type="trade_execution",
    market_data={
        "symbol": "AAPL",
        "current_price": 150.25,
        "volume": 45000000,
        "trend": "bullish"
    },
    tool_results=[rsi_result, macd_result],
    template_name="momentum_analysis"
)

print(f"Context summary: {context.generate_summary()}")
```

##### `register_template(template: ContextTemplate) -> bool`

Register a custom context template.

**Parameters:**
- `template` (ContextTemplate): Context template definition

**Returns:**
- `bool`: True if registered successfully

##### `build_multi_symbol_context(agent_id: str, session_id: str, decision_type: str, symbols_data: Dict[str, Dict]) -> DecisionContext`

Build context for multi-symbol decisions (e.g., portfolio optimization).

**Parameters:**
- `agent_id` (str): Agent identifier  
- `session_id` (str): Session identifier
- `decision_type` (str): Decision type
- `symbols_data` (Dict): Data for multiple symbols

**Returns:**
- `DecisionContext`: Multi-symbol context

### Data Classes

#### DecisionContext

```python
@dataclass
class DecisionContext:
    agent_id: str
    session_id: str
    decision_type: str
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_data: Dict[str, Any] = field(default_factory=dict)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    rag_results: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    context_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    previous_context_id: Optional[str] = None
    
    def generate_summary(self) -> str: ...
    def to_dict(self) -> Dict[str, Any]: ...
```

#### ContextTemplate

```python
@dataclass
class ContextTemplate:
    name: str
    description: str
    required_components: List[ContextComponent]
    optional_components: List[ContextComponent] = field(default_factory=list)
    max_context_length: int = 4000
    priority_weights: Dict[ContextComponent, float] = field(default_factory=dict)
    
    def is_valid(self) -> bool: ...
```

## Error Handling

All methods can raise the following exceptions:

- `ValueError`: Invalid parameters or configuration
- `ConnectionError`: Database or service connection issues
- `TimeoutError`: Operation timeout
- `PermissionError`: Insufficient permissions for operation
- `RuntimeError`: General runtime errors

**Example Error Handling:**
```python
try:
    result = await tool_registry.execute_tool("complex_tool", params, context)
    if result.success:
        # Handle successful result
        pass
    else:
        # Handle tool execution failure
        print(f"Tool failed: {result.error}")
except TimeoutError:
    print("Tool execution timed out")
except PermissionError:
    print("Insufficient permissions")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration

### Environment Variables

```python
# Memory Manager Configuration
MEMORY_VECTOR_DIMENSION = 384
MEMORY_SIMILARITY_THRESHOLD = 0.7
MEMORY_MAX_STORAGE_SIZE = 1000000

# RAG Service Configuration  
RAG_CHUNK_SIZE = 500
RAG_CHUNK_OVERLAP = 50
RAG_MAX_RESULTS = 20

# Tool Registry Configuration
TOOL_MAX_CONCURRENT_EXECUTIONS = 10
TOOL_DEFAULT_TIMEOUT_SECONDS = 30

# Context Builder Configuration
CONTEXT_MAX_TOKENS = 4000
CONTEXT_DEFAULT_MEMORY_LIMIT = 10
```

### Health Checks

All components provide health check methods:

```python
# Check individual component health
memory_health = await memory_manager.health_check()
rag_health = await rag_service.health_check()
tool_health = await tool_registry.health_check()
context_health = await context_builder.health_check()

# Overall system health
system_healthy = all([
    memory_health["status"] == "healthy",
    rag_health["status"] == "healthy", 
    tool_health["status"] == "healthy",
    context_health["status"] == "healthy"
])
```