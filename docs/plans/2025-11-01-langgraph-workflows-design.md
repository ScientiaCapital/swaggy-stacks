# LangGraph Workflows Design

**Date**: 2025-11-01
**Status**: Approved for Implementation
**Architecture**: Supervisor Pattern with PostgreSQL Checkpointing

---

## Overview

This design connects five specialized trading agents into two coordinated workflows using LangGraph's supervisor pattern. A supervisor agent orchestrates specialized agents (Research, Strategy, Risk, Execution, Learning) to analyze markets, select strategies, validate risk, execute trades, and learn from outcomes.

---

## Architecture

### Supervisor Pattern

The supervisor pattern uses a central coordinator that calls specialized agents as tools:

**Components**:
- **Supervisor Agent**: Claude Sonnet 4.5 makes routing decisions
- **Specialized Agents**: Research, Strategy, Risk, Execution, Learning (exposed as tools)
- **State Graph**: Shared state flows between supervisor and agents
- **PostgreSQL Checkpointer**: Persists state across restarts

**Why Supervisor Pattern**:
- Dynamic routing: Supervisor adapts to state changes
- Conditional execution: Skip agents when conditions fail
- Clear responsibility: Supervisor coordinates, agents execute
- Resumable workflows: State persists in PostgreSQL

---

## Workflows

### Trading Workflow (Real-Time)

Executes during market hours when trading opportunities arise.

**Flow**:
1. Supervisor receives market data and symbol
2. Calls Research Agent → detects market regime
3. Calls Strategy Agent → selects options strategy
4. Calls Risk Agent → validates portfolio risk
5. If Risk approves → Calls Execution Agent
6. If Risk rejects → Logs rejection and ends

**State Schema**:
```python
class TradingState(TypedDict):
    # Input
    symbol: str
    market_data: Dict[str, Any]

    # Research output
    market_regime: str
    regime_confidence: float
    signals: List[Dict[str, Any]]

    # Strategy output
    recommended_strategy: str
    strategy_params: Dict[str, Any]
    strategy_confidence: float

    # Risk output
    risk_approved: bool
    position_size: float
    risk_assessment: Dict[str, Any]

    # Execution output
    execution_status: str
    orders: List[Dict[str, Any]]

    # Control
    next_agent: str
    messages: List[BaseMessage]
    completed: bool
```

**Trigger**: API endpoint `POST /api/v1/trading/execute`

---

### Learning Workflow (Batch)

Runs overnight to process completed trades and improve strategies.

**Flow**:
1. Supervisor receives list of completed trades
2. Calls Learning Agent → analyzes trades, extracts insights
3. Updates pattern memory and regime-strategy matrix
4. Generates next-day recommendations

**State Schema**:
```python
class LearningState(TypedDict):
    completed_trades: List[Dict[str, Any]]
    learning_summary: str
    patterns_updated: int
    next_day_recommendations: List[str]
    completed: bool
```

**Trigger**: Celery scheduled task at 11:00 PM ET

---

## Implementation

### File Structure

```
backend/app/agents/workflows/
├── __init__.py
├── state_schemas.py          # TypedDict definitions
├── supervisor.py             # Supervisor logic
├── trading_workflow.py       # Trading workflow
└── learning_workflow.py      # Learning workflow

backend/tests/workflows/
├── __init__.py
├── test_trading_workflow.py
└── test_learning_workflow.py
```

### Supervisor Implementation

```python
class TradingSupervisor:
    """Coordinates specialized agents."""

    def __init__(self):
        self.model = ChatAnthropic(model="claude-sonnet-4.5")
        self.checkpointer = PostgresSaver(connection_string=LANGGRAPH_DB_URI)
        self.tools = self._create_agent_tools()

    def _create_agent_tools(self) -> List[Tool]:
        """Convert agents to tools for supervisor."""
        agents = [
            ResearchAgent(),
            StrategyAgent(),
            RiskAgent(),
            ExecutionAgent()
        ]
        return [self._agent_to_tool(agent) for agent in agents]

    def _agent_to_tool(self, agent: BaseAgent) -> Tool:
        """Convert agent into LangGraph tool."""
        return Tool(
            name=agent.name.lower().replace(" ", "_"),
            description=agent.description,
            func=agent.invoke
        )
```

### StateGraph Construction

```python
workflow = StateGraph(TradingState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("call_agent", agent_caller_node)

# Add edges
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    should_continue,
    {"continue": "call_agent", "end": END}
)
workflow.add_edge("call_agent", "supervisor")

# Compile with checkpointing
app = workflow.compile(checkpointer=checkpointer)
```

---

## Error Handling

### Fail-Fast Strategy

Workflows stop immediately when any agent fails. This prevents partial trades and ensures clean failure states.

```python
class AgentFailureError(WorkflowError):
    """Raised when an agent fails."""
    def __init__(self, agent_name: str, error: str):
        self.agent_name = agent_name
        self.error = error
        super().__init__(f"{agent_name} failed: {error}")

def agent_caller_node(state: TradingState) -> TradingState:
    """Call agent and fail-fast on error."""
    try:
        agent_output = invoke_agent(state["next_agent"], state)
        return {**state, **agent_output}
    except Exception as e:
        logger.error("agent_failure", agent=state["next_agent"], error=str(e))
        raise AgentFailureError(state["next_agent"], str(e))
```

**Rationale**: In financial systems, partial execution creates inconsistent state. Fail-fast ensures all-or-nothing execution.

---

## Testing Strategy

### TDD Approach

Follow RED-GREEN cycle used for agent implementation:

1. **RED**: Write failing test
2. **GREEN**: Implement minimal code to pass
3. **COMMIT**: Save working implementation

### Test Coverage

**Trading Workflow Tests**:
- `test_complete_flow`: Research → Strategy → Risk → Execution succeeds
- `test_risk_rejection`: Risk rejects → no execution occurs
- `test_agent_failure`: Agent fails → WorkflowError raised
- `test_checkpoint_persistence`: State persists to PostgreSQL
- `test_supervisor_routing`: Supervisor selects correct next agent

**Learning Workflow Tests**:
- `test_learning_complete`: Processes trades and generates insights
- `test_no_trades`: Handles empty trade list gracefully
- `test_pattern_updates`: Verifies pattern memory updates

---

## Configuration

### Environment Variables

Add to `.env`:
```bash
# LangGraph PostgreSQL Checkpointing
LANGGRAPH_DB_URI=postgresql://user:pass@localhost:5432/swaggy_stacks
```

### Database Setup

LangGraph creates these tables automatically:
- `langgraph_checkpoints`: Stores workflow state
- `langgraph_writes`: Stores state mutations

**No migration required** - tables create on first run.

---

## Integration Points

### API Endpoint

```python
@router.post("/trading/execute")
async def execute_trade(request: TradeRequest):
    """Trigger trading workflow."""
    workflow = TradingWorkflow()

    initial_state = {
        "symbol": request.symbol,
        "market_data": await fetch_market_data(request.symbol),
        "completed": False
    }

    result = await workflow.run(initial_state)
    return result
```

### Scheduled Learning

```python
@celery_app.task
def run_learning_workflow():
    """Runs at 11:00 PM ET daily."""
    workflow = LearningWorkflow()

    trades = fetch_completed_trades_today()

    initial_state = {
        "completed_trades": trades,
        "completed": False
    }

    result = workflow.run(initial_state)
    logger.info("learning_complete", insights=result["patterns_updated"])
```

---

## Implementation Order

### Task 3.7.1: Setup Infrastructure
- Create `state_schemas.py` with TypedDict definitions
- Configure `LANGGRAPH_DB_URI` in `.env`
- Verify PostgreSQL connection

### Task 3.7.2: Implement Supervisor
- Create `supervisor.py` with TradingSupervisor class
- Implement agent-to-tool conversion
- Write supervisor system prompt

### Task 3.7.3: Implement Trading Workflow (TDD)
- Write failing tests in `test_trading_workflow.py`
- Implement `trading_workflow.py`
- Verify all tests pass (5 tests)

### Task 3.8: Implement Learning Workflow (TDD)
- Write failing tests in `test_learning_workflow.py`
- Implement `learning_workflow.py`
- Verify all tests pass (3 tests)

### Task 3.9: Integration Testing
- End-to-end test with real agents
- Checkpoint persistence verification
- Error handling validation

---

## Success Criteria

- [ ] All workflow tests pass (8 total)
- [ ] Trading workflow executes: Research → Strategy → Risk → Execution
- [ ] Risk rejection prevents execution
- [ ] Agent failures raise WorkflowError
- [ ] State persists to PostgreSQL
- [ ] Learning workflow processes trades and updates patterns
- [ ] Workflows resume after restart using checkpoints

---

## References

- LangGraph Documentation: https://langchain-ai.github.io/langgraph/
- Supervisor Pattern: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
- PostgresSaver: https://langchain-ai.github.io/langgraph/reference/checkpoints/#postgresaver
