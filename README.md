# OpenTelemetry GenAI Hydrator

A Python toolkit for analyzing and hydrating OpenTelemetry distributed tracing data, with a focus on GenAI applications. This package provides tools for retrieving spans with enriched event data, analyzing trace hierarchies, and working with Application Insights data.

## What This Project Does

The **otel_gen_ai_hydrator** project provides infrastructure for:

1. **Span Hydration**: Retrieves OpenTelemetry spans and enriches them with detailed event data (messages, choices, tool calls) specifically for GenAI applications
2. **Application Insights Integration**: Connects to Azure Application Insights to query and retrieve distributed tracing data
3. **GenAI Event Processing**: Parses and structures GenAI-specific events like user messages, assistant responses, system messages, tool calls, and choice events
4. **Trace Analysis**: Analyzes parent-child relationships in distributed traces and provides span hierarchy navigation
5. **Data Models**: Provides Pydantic models for spans, traces, and various GenAI event types

### Key Capabilities

- **Query spans by ID** from Application Insights with configurable time ranges
- **Retrieve child spans** for a given parent span with optional GenAI operation filtering
- **Parse GenAI events** into strongly-typed Pydantic models (user messages, assistant messages, system messages, tool messages, choice events)

## Features

- **SpanHydrator**: Core class for retrieving and enriching span data
- **Application Insights Connector**: Native integration with Azure Application Insights
- **GenAI Event Models**: Pydantic models for various GenAI event types
- **Flexible Authentication**: Support for multiple Azure authentication methods

## Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### With Optional Dependencies

```bash
# For Azure Application Insights integration
pip install -e ".[azure]"
```

## Quick Start

### Basic SpanHydrator Usage

```python
from otel_gen_ai_hydrator.span_hydrator import SpanHydrator
from otel_gen_ai_hydrator.sources.application_insights import (
    ApplicationInsightsConnector, 
    ApplicationInsightsConfig
)
from datetime import timedelta

# Configure Application Insights
config = ApplicationInsightsConfig(
    resource_id="/subscriptions/{sub-id}/resourceGroups/{rg}/providers/Microsoft.Insights/components/{app-insights-name}"
)

# Create connector and hydrator
connector = ApplicationInsightsConnector(config)
hydrator = SpanHydrator(connector)

# Test the connection
if connector.test_connection():
    print("✅ Connected to Application Insights!")
    
    # Get a span by ID with enriched data
    span = hydrator.get_span_by_id("your-span-id", time_range=timedelta(days=30))
    
    if span:
        print(f"Span: {span.name}")
        print(f"Duration: {span.duration_ms} ms")
        print(f"Status: {span.status}")
        print(f"Events: {len(span.events)} events")
        print(f"Attributes: {list(span.attributes.keys())}")
        
        # Access GenAI events
        for event in span.events:
            if hasattr(event, 'name') and 'gen_ai' in event.name:
                print(f"GenAI Event: {event.name}")
    
    # Get child spans
    children = hydrator.get_child_spans("parent-span-id", time_range=timedelta(days=1))
    print(f"Found {len(children)} child spans")
    
    # Get child spans filtered by GenAI operation
    chat_spans = hydrator.get_child_spans(
        "parent-span-id", 
        time_range=timedelta(days=1),
        gen_ai_operation_name="chat.completions"
    )
    print(f"Found {len(chat_spans)} chat completion spans")
```

### Working with GenAI Events

```python
from otel_gen_ai_hydrator.models.events import (
    GenAIUserMessageEvent,
    GenAIAssistantMessageEvent,
    GenAISystemMessageEvent,
    GenAIToolMessageEvent,
    GenAIChoiceEvent
)

# The hydrator automatically parses events into typed objects
span = hydrator.get_span_by_id("your-genai-span-id")

if span:
    for event in span.events:
        if isinstance(event, GenAIUserMessageEvent):
            content = event.attributes.get("gen_ai.user.message.content")
            print(f"User: {content}")
        elif isinstance(event, GenAIAssistantMessageEvent):
            content = event.attributes.get("gen_ai.assistant.message.content")
            print(f"Assistant: {content}")
        elif isinstance(event, GenAISystemMessageEvent):
            content = event.attributes.get("gen_ai.system.message.content")
            print(f"System: {content}")
        elif isinstance(event, GenAIToolMessageEvent):
            tool_name = event.attributes.get("gen_ai.tool.name")
            content = event.attributes.get("gen_ai.tool.message.content")
            print(f"Tool ({tool_name}): {content}")
        elif isinstance(event, GenAIChoiceEvent):
            finish_reason = event.attributes.get("gen_ai.choice.finish_reason")
            print(f"Choice completed: {finish_reason}")
```

### Azure Authentication

```python
from azure.identity import DefaultAzureCredential, ClientSecretCredential

# Option 1: Use DefaultAzureCredential (tries multiple auth methods)
config = ApplicationInsightsConfig(
    resource_id="your-resource-id",
    credential=DefaultAzureCredential()
)

# Option 2: Use Service Principal
config = ApplicationInsightsConfig(
    resource_id="your-resource-id",
    credential=ClientSecretCredential(
        tenant_id="your-tenant-id",
        client_id="your-client-id",
        client_secret="your-client-secret"
    )
)

# Option 3: Use environment variables
# Set AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET
# Then use DefaultAzureCredential()
```

### Data Models

```python
from otel_gen_ai_hydrator.models.span import Span
from otel_gen_ai_hydrator.models.trace import Trace
from otel_gen_ai_hydrator.models.events import (
    GenAIUserMessageEvent,
    GenAIAssistantMessageEvent,
    GenAISystemMessageEvent,
    GenAIToolMessageEvent,
    GenAIChoiceEvent
)

# All models are Pydantic models with full validation
span = Span(
    span_id="test-123",
    trace_id="trace-456",
    operation_id="op-789",
    name="chat-completion",
    start_time="2024-01-01T12:00:00Z",
    end_time="2024-01-01T12:00:05Z",
    duration_ms=5000.0,
    status="Success",
    attributes={"gen_ai.system": "openai"},
    events=[],
    span_type="dependency"
)
```

## Architecture

### Core Components

- **SpanHydrator**: Main class that orchestrates span retrieval and enrichment
- **SourceConnector**: Abstract interface for data source connectors
- **ApplicationInsightsConnector**: Concrete implementation for Azure Application Insights
- **Span/Trace Models**: Pydantic data models for structured trace data
- **GenAI Event Models**: Specialized models for GenAI-specific telemetry events

### Key Classes

#### SpanHydrator
- `get_span_by_id(span_id, time_range)`: Retrieve a single span with full event data
- `get_child_spans(parent_span_id, time_range, gen_ai_operation_name)`: Get child spans with optional filtering

#### ApplicationInsightsConnector
- `query_span_by_id(span_id, time_range)`: Query Application Insights for a specific span
- `query_child_spans(parent_span_id, time_range, gen_ai_operation_name)`: Query for child spans
- `test_connection()`: Verify connectivity to Application Insights

#### Data Models

**Core Models:**
- **Span**: Represents a single span with metadata, attributes, and events
- **Trace**: Represents a complete trace (collection of related spans)

**GenAI Event Models:**
- **GenAIUserMessageEvent**: User input messages
- **GenAIAssistantMessageEvent**: AI assistant responses  
- **GenAISystemMessageEvent**: System prompts and instructions
- **GenAIToolMessageEvent**: Tool/function call results
- **GenAIChoiceEvent**: AI choice/completion events

### Data Flow

1. **Query**: SpanHydrator receives a request for span data
2. **Retrieve**: Calls the SourceConnector to fetch raw data from Application Insights
3. **Parse**: Converts raw telemetry data into structured Span objects
4. **Enrich**: Parses events into typed GenAI event objects
5. **Return**: Provides fully hydrated Span objects with typed events

## Current Limitations

- **Single Data Source**: Currently only supports Azure Application Insights
- **GenAI Focus**: Optimized for GenAI applications (OpenAI, Azure OpenAI, etc.)
- **Read-Only**: Only supports querying existing data, not writing new traces
- **Time-Based Queries**: Requires time range specifications for efficient querying

## Testing

The project includes comprehensive test coverage:

### Unit Tests
```bash
# Run unit tests
pytest tests/unit/ -v
```

### Integration Tests (Requires Azure Setup)
```bash
# Set environment variables
export AZURE_APPLICATION_INSIGHTS_RESOURCE_ID="/subscriptions/.../components/your-app-insights"

# Run integration tests
pytest tests/integration/ -v -m "integration"

# Run with real data (requires actual Azure credentials)
pytest tests/integration/ -v -m "real_data"
```

### Test Categories
- **Unit Tests**: Test individual components in isolation with mocks
- **Integration Tests**: Test real Application Insights connections and data
- **Performance Tests**: Measure query performance and timeouts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/singankit/otel_gen_ai_hydrator.git
cd otel_gen_ai_hydrator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,azure]"

# Run tests
pytest

# Run linting and formatting
black src/ tests/
flake8 src/ tests/
mypy src/
```

## Environment Variables

For integration tests with real Azure data, set these environment variables:

```bash
# Required for integration tests
export AZURE_APPLICATION_INSIGHTS_RESOURCE_ID="/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Insights/components/{name}"

# Optional for service principal auth
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"  
export AZURE_CLIENT_SECRET="your-client-secret"
```

## Examples and Use Cases

### Analyzing a GenAI Conversation Flow
```python
# Get the root conversation span
conversation = hydrator.get_span_by_id("conversation-root-span-id")

# Get all GenAI operations in the conversation
genai_spans = hydrator.get_child_spans(
    "conversation-root-span-id",
    gen_ai_operation_name="chat.completions"
)

# Analyze the conversation flow
for span in genai_spans:
    print(f"Operation: {span.name} ({span.duration_ms}ms)")
    for event in span.events:
        if isinstance(event, GenAIUserMessageEvent):
            print(f"  User: {event.attributes.get('gen_ai.user.message.content', '')}")
        elif isinstance(event, GenAIAssistantMessageEvent):
            print(f"  Assistant: {event.attributes.get('gen_ai.assistant.message.content', '')}")
```

### Monitoring GenAI Performance
```python
# Get all embedding operations in the last hour
embedding_spans = hydrator.get_child_spans(
    "service-root-span",
    time_range=timedelta(hours=1),
    gen_ai_operation_name="embeddings"
)

# Calculate average embedding time
if embedding_spans:
    avg_duration = sum(span.duration_ms for span in embedding_spans) / len(embedding_spans)
    print(f"Average embedding time: {avg_duration:.2f}ms")
    print(f"Total embedding operations: {len(embedding_spans)}")
```

## API Reference

### SpanHydrator

```python
class SpanHydrator:
    def __init__(self, source_connector: SourceConnector)
    
    def get_span_by_id(
        self, 
        span_id: str, 
        time_range: timedelta = timedelta(days=30)
    ) -> Optional[Span]
    
    def get_child_spans(
        self,
        parent_span_id: str,
        time_range: timedelta = timedelta(days=30),
        gen_ai_operation_name: str = None
    ) -> List[Span]
```

### ApplicationInsightsConnector

```python
class ApplicationInsightsConnector(SourceConnector):
    def __init__(self, config: ApplicationInsightsConfig)
    
    def test_connection(self) -> bool
    
    def query_span_by_id(
        self, 
        span_id: str, 
        time_range: timedelta = timedelta(days=30)
    ) -> Optional[Span]
    
    def query_child_spans(
        self,
        parent_span_id: str,
        time_range: timedelta = timedelta(days=30),
        gen_ai_operation_name: str = None
    ) -> List[Span]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support and Resources

- **GitHub Repository**: [https://github.com/singankit/otel_gen_ai_hydrator](https://github.com/singankit/otel_gen_ai_hydrator)
- **Issues**: [GitHub Issues](https://github.com/singankit/otel_gen_ai_hydrator/issues)
- **Integration Test Documentation**: See [tests/integration/README.md](tests/integration/README.md)

## Acknowledgments

- Built for OpenTelemetry-based GenAI application observability
- Designed to work with Azure Application Insights and OpenAI tracing
- Inspired by the need for better GenAI application monitoring and analysis
