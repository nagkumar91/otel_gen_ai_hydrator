# Tests for otel_gen_ai_hydrator

This directory contains comprehensive tests for the `otel_gen_ai_hydrator` project.

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and shared fixtures
├── unit/                    # Unit tests
│   ├── __init__.py
│   ├── test_span_hydrator.py    # Unit tests for SpanHydrator class
│   └── test_models.py          # Unit tests for GenAI event models
└── integration/             # Integration tests
    ├── __init__.py
    └── test_span_hydrator_integration.py  # Integration tests for SpanHydrator
```

## Running Tests

### Using the test runner script

The project includes a test runner script (`run_tests.py`) that provides convenient ways to run different types of tests:

```bash
# Run unit tests only
python run_tests.py unit

# Run integration tests only  
python run_tests.py integration

# Run all tests
python run_tests.py all

# Run tests with coverage report
python run_tests.py coverage

# Run a specific test file
python run_tests.py specific --path tests/unit/test_span_hydrator.py
```

### Using pytest directly

You can also run tests directly with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v -m integration

# Run with coverage
pytest tests/ -v --cov=src/otel_gen_ai_hydrator --cov-report=html

# Run a specific test file
pytest tests/unit/test_span_hydrator.py -v

# Run a specific test method
pytest tests/unit/test_span_hydrator.py::TestSpanHydrator::test_span_hydrator_initialization -v
```

## Test Categories

### Unit Tests (`tests/unit/`)

Unit tests focus on testing individual components in isolation:

- **`test_span_hydrator.py`**: Tests for the `SpanHydrator` class
  - Initialization
  - Span retrieval by ID
  - Child span retrieval
  - Error handling
  
- **`test_models.py`**: Tests for GenAI event models
  - Pydantic model validation
  - Event serialization/deserialization
  - Field validation and constraints

### Integration Tests (`tests/integration/`)

Integration tests verify that components work together correctly:

- **`test_span_hydrator_integration.py`**: End-to-end tests
  - SpanHydrator with source connectors
  - Complete workflow testing
  - Error handling across components

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestSpanHydrator`)
- Test methods: `test_*` (e.g., `test_span_hydrator_initialization`)

### Example Unit Test

```python
def test_span_hydrator_initialization():
    \"\"\"Test SpanHydrator can be initialized with a source connector.\"\"\"
    mock_connector = Mock(spec=SourceConnector)
    hydrator = SpanHydrator(mock_connector)
    
    assert hydrator.source_connector == mock_connector
    assert hydrator.logger is not None
```

### Example Integration Test

```python
@pytest.mark.integration
def test_span_hydrator_end_to_end():
    \"\"\"Test complete SpanHydrator workflow.\"\"\"
    # Setup
    connector = create_test_connector()
    hydrator = SpanHydrator(connector)
    
    # Test
    span = hydrator.get_span_by_id("test-span-id")
    
    # Verify
    assert span is not None
    assert span.span_id == "test-span-id"
```

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `sample_span_data`: Sample span data for testing
- `sample_event_data`: Sample GenAI event data for testing

## Mocking

Tests use the `unittest.mock` module for mocking dependencies:

- Mock external services (Application Insights)
- Mock source connectors for unit tests
- Patch Azure SDK components for integration tests

## Coverage

The project aims for high test coverage. Run coverage reports to identify untested code:

```bash
python run_tests.py coverage
```

This generates:
- Terminal coverage report
- HTML coverage report in `htmlcov/` directory

## Continuous Integration

Tests are designed to run in CI/CD environments. All tests should:
- Be deterministic (no random behavior)
- Not depend on external services (use mocks)
- Complete quickly (especially unit tests)
- Clean up after themselves

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running tests from the project root directory
2. **Missing dependencies**: Install test dependencies with `pip install pytest pytest-cov`
3. **Azure SDK errors**: Integration tests mock Azure components, but ensure the package is installed

### Debug Mode

Run tests with additional debugging:

```bash
pytest tests/ -v -s --tb=long
```

- `-v`: Verbose output
- `-s`: Don't capture stdout (shows print statements)
- `--tb=long`: Long traceback format
