# Evaluate with Traces

A comprehensive Python toolkit for evaluating applications using distributed tracing data. This package provides tools and utilities for collecting, processing, and analyzing distributed traces to compute performance and reliability metrics.

## Features

- **Trace Analysis**: Tools for analyzing distributed traces from various sources
- **Evaluation Metrics**: Frameworks for computing performance and reliability metrics
- **Integration Support**: Connectors for Application Insights, Jaeger, OpenTelemetry
- **Data Processing**: Utilities for processing and analyzing trace data
- **Visualization**: Tools for creating trace visualizations and reports

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

# For Jupyter notebook support
pip install -e ".[jupyter]"

# For advanced visualizations
pip install -e ".[visualization]"

# Install all optional dependencies
pip install -e ".[dev,azure,jupyter,visualization]"
```

## Quick Start

### Basic Usage

```python
from otel_gen_ai_hydrator import parse_spans, build_trace_tree, get_span_hierarchy
from otel_gen_ai_hydrator.integrations import ApplicationInsightsConnector

# Load trace data (example with mock data)
spans_data = [
    {
        'id': 'span1',
        'operation_Id': 'trace1',
        'operation_ParentId': None,
        'name': 'HTTP GET /api/users',
        'timestamp': '2024-01-01T10:00:00Z',
        'duration': 1500,
        'resultCode': '200',
        'itemType': 'request'
    },
    # ... more spans
]

# Parse and analyze traces
spans = parse_spans(spans_data)
trace = build_trace_tree(spans)

if trace:
    # Analyze the trace structure
    hierarchy = get_span_hierarchy(trace)
    
    # Print basic trace information
    print(f"Trace ID: {trace.trace_id}")
    print(f"Total Duration: {trace.total_duration_ms:.2f} ms")
    print(f"Span Count: {trace.span_count}")
    print(f"Error Count: {trace.error_count}")
    print(f"Hierarchy Levels: {len(hierarchy)}")
```

### SpanHydrator for Enhanced Span Data

```python
from otel_gen_ai_hydrator import SpanHydrator
from otel_gen_ai_hydrator.integrations import ApplicationInsightsConnector, ApplicationInsightsConfig
from datetime import timedelta

# Configure and initialize connector
config = ApplicationInsightsConfig(resource_id="your-resource-id")
connector = ApplicationInsightsConnector(config)

# Create SpanHydrator
hydrator = SpanHydrator(connector)

# Get a fully hydrated span with events and attributes
span = hydrator.get_span_by_id("your-span-id", time_range=timedelta(hours=2))

if span:
    print(f"Span: {span.name}")
    print(f"Duration: {span.duration_ms} ms")
    print(f"Attributes: {len(span.attributes)} items")
    print(f"Events: {len(span.events)} items")

# Or get specific data
events = hydrator.get_span_events("your-span-id")
attributes = hydrator.get_span_attributes("your-span-id")
```

### Application Insights Integration

```python
from otel_gen_ai_hydrator.integrations import ApplicationInsightsConnector, ApplicationInsightsConfig
from datetime import timedelta

# Configure Application Insights
config = ApplicationInsightsConfig(
    workspace_id="your-workspace-id"
)

# Initialize connector
connector = ApplicationInsightsConnector(config)

# Test connection
if connector.test_connection():
    print("Connected to Application Insights successfully!")
    
    # Query spans by operation ID
    spans_data = connector.query_spans_by_operation_id(
        operation_id="your-operation-id",
        time_range=timedelta(hours=1)
    )
    
    # Query child spans
    child_spans = connector.query_child_spans(
        parent_span_id="your-parent-span-id",
        time_range=timedelta(hours=1)
    )
    
    # Query events for a span
    events = connector.query_events_for_span(
        span_id="your-span-id",
        time_range=timedelta(hours=1)
    )
```

### Performance Metrics

```python
from otel_gen_ai_hydrator.metrics import PerformanceMetrics, PerformanceThresholds

# Set custom thresholds
thresholds = PerformanceThresholds(
    response_time_ms=3000,  # 3 seconds
    error_rate_percent=2.0,  # 2%
    p95_response_time_ms=8000  # 8 seconds
)

performance = PerformanceMetrics(thresholds)

# Calculate various performance metrics
response_time_results = performance.calculate_response_time_metrics(traces)
throughput_result = performance.calculate_throughput(traces)
latency_dist = performance.calculate_latency_distribution(traces)
apdex_score = performance.calculate_apdex_score(traces)

# Print performance summary
for result in response_time_results:
    threshold_status = "✓" if not result.metadata.get("threshold_exceeded", False) else "✗"
    print(f"{threshold_status} {result.metric_name}: {result.value} {result.unit}")
```

### Reliability Metrics

```python
from otel_gen_ai_hydrator.metrics import ReliabilityMetrics, ReliabilityThresholds

# Set custom thresholds
thresholds = ReliabilityThresholds(
    max_error_rate_percent=1.0,  # 1% max error rate
    min_success_rate_percent=99.5  # 99.5% min success rate
)

reliability = ReliabilityMetrics(thresholds)

# Calculate reliability metrics
error_rate = reliability.calculate_error_rate(traces)
success_rate = reliability.calculate_success_rate(traces)
error_distribution = reliability.calculate_error_types_distribution(traces)
cascade_analysis = reliability.calculate_failure_cascade_analysis(traces)

# Print reliability summary
print(f"Error Rate: {error_rate.value}% (Threshold: {error_rate.metadata['threshold']}%)")
print(f"Success Rate: {success_rate.value}% (Threshold: {success_rate.metadata['threshold']}%)")
```

## Architecture

### Core Components

- **Span and Trace**: Core data structures for representing distributed traces
- **parse_spans()**: Function to parse raw span data into Span objects
- **build_trace_tree()**: Function to build trace trees from spans
- **get_span_hierarchy()**: Function to analyze span hierarchical structure
- **SpanHydrator**: Class for enriching spans with events and attributes from source connectors
- **SourceConnector**: Abstract interface for data source connectors
- **ApplicationInsightsConnector**: Integration with Azure Application Insights that implements SourceConnector

### Data Models

- **TraceSpan**: Represents a single span in a distributed trace
- **TraceTree**: Represents a complete trace as a tree of spans
- **EvaluationResult**: Result of a metric calculation
- **EvaluationReport**: Comprehensive evaluation report

## Supported Metrics

### Performance Metrics
- Average, median, min, max response time
- Percentiles (P50, P90, P95, P99)
- Throughput (requests per second)
- Latency distribution
- Apdex score
- Per-service performance metrics

### Reliability Metrics
- Error rate and success rate
- Error type distribution
- Failure cascade analysis
- Timeout rate
- Mean Time To Recovery (MTTR)
- Mean Time Between Failures (MTBF)
- Per-service reliability metrics

## Examples

See the `examples/` directory for complete examples:

- `basic_analysis.py`: Basic trace analysis workflow
- `application_insights_integration.py`: Integration with Application Insights
- `custom_metrics.py`: Creating custom evaluation metrics
- `batch_processing.py`: Processing large datasets of traces
- `jupyter_analysis.ipynb`: Interactive analysis in Jupyter

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
git clone https://github.com/yourusername/otel_gen_ai_hydrator.git
cd otel_gen_ai_hydrator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 src/
black src/
mypy src/
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=otel_gen_ai_hydrator

# Run specific test file
pytest tests/test_span_hydrator.py

# Run with verbose output
pytest -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- Documentation: [Link to documentation]
- Issues: [GitHub Issues](https://github.com/yourusername/otel_gen_ai_hydrator/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/otel_gen_ai_hydrator/discussions)

## Roadmap

- [ ] Support for more tracing backends (Jaeger, Zipkin)
- [ ] Real-time evaluation capabilities
- [ ] Advanced visualization dashboards
- [ ] Machine learning-based anomaly detection
- [ ] Custom alerting and notification system
- [ ] Integration with popular observability platforms
