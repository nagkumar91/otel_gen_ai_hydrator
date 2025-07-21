# Integration Testing Configuration

This document explains how to set up and run real integration tests with Application Insights.

## Prerequisites

1. **Azure Application Insights Resource**: You need access to an Azure Application Insights instance
2. **Azure Credentials**: Appropriate permissions to query the Application Insights resource
3. **Python Dependencies**: Azure SDK packages installed

## Installation

Install the required Azure dependencies:

```bash
pip install azure-monitor-query azure-identity
```

Or install with the azure extras:

```bash
pip install .[azure]
```

## Environment Variables

Set the following environment variables before running the tests:

### Required

- `AZURE_APPLICATION_INSIGHTS_RESOURCE_ID`: The full resource ID of your Application Insights instance
  ```
  /subscriptions/{subscription-id}/resourceGroups/{resource-group}/providers/Microsoft.Insights/components/{app-insights-name}
  ```

### Optional (for specific authentication)

- `AZURE_TENANT_ID`: Your Azure tenant ID
- `AZURE_CLIENT_ID`: Your Azure client ID (for service principal authentication)
- `AZURE_CLIENT_SECRET`: Your Azure client secret (for service principal authentication)

If these are not provided, the tests will use `DefaultAzureCredential` which tries multiple authentication methods.

## Authentication Methods

The tests support multiple authentication methods:

1. **Service Principal** (recommended for CI/CD):
   ```bash
   export AZURE_TENANT_ID="your-tenant-id"
   export AZURE_CLIENT_ID="your-client-id"
   export AZURE_CLIENT_SECRET="your-client-secret"
   ```

2. **Azure CLI** (for local development):
   ```bash
   az login
   ```

3. **Managed Identity** (when running on Azure):
   - No additional configuration needed

4. **Visual Studio Code** (for local development):
   - Install Azure extension and sign in

## Running the Tests

### Run All Integration Tests

```bash
pytest tests/integration/test_span_hydrator.py::TestRealApplicationInsightsIntegration -v
```

### Run Specific Test Categories

```bash
# Run connection tests only
pytest tests/integration/test_span_hydrator.py::TestRealApplicationInsightsIntegration::test_connection_to_application_insights -v

# Run data discovery tests
pytest tests/integration/test_span_hydrator.py::TestRealDataScenarios -v

# Run performance tests (these take longer)
pytest tests/integration/test_span_hydrator.py::TestPerformanceWithRealData -v
```

### Run with Specific Markers

```bash
# Run all integration tests
pytest -m "integration" -v

# Run tests that use real data
pytest -m "real_data" -v

# Run slow tests
pytest -m "slow" -v
```

## Example Configuration

### Windows (PowerShell)
```powershell
$env:AZURE_APPLICATION_INSIGHTS_RESOURCE_ID = "/subscriptions/12345678-1234-1234-1234-123456789012/resourceGroups/my-rg/providers/Microsoft.Insights/components/my-app-insights"
$env:AZURE_TENANT_ID = "your-tenant-id"
$env:AZURE_CLIENT_ID = "your-client-id"
$env:AZURE_CLIENT_SECRET = "your-client-secret"

pytest tests/integration/test_span_hydrator.py::TestRealApplicationInsightsIntegration -v
```

### Linux/macOS (Bash)
```bash
export AZURE_APPLICATION_INSIGHTS_RESOURCE_ID="/subscriptions/12345678-1234-1234-1234-123456789012/resourceGroups/my-rg/providers/Microsoft.Insights/components/my-app-insights"
export AZURE_TENANT_ID="your-tenant-id"
export AZURE_CLIENT_ID="your-client-id"
export AZURE_CLIENT_SECRET="your-client-secret"

pytest tests/integration/test_span_hydrator.py::TestRealApplicationInsightsIntegration -v
```

## Test Categories

### Connection Tests
- `test_connection_to_application_insights`: Verifies basic connectivity

### Data Query Tests
- `test_query_recent_spans`: Tests span retrieval functionality
- `test_query_child_spans_with_real_data`: Tests child span queries
- `test_query_child_spans_with_gen_ai_filter`: Tests filtered queries

### Error Handling Tests
- `test_error_handling_with_invalid_time_range`: Tests large time ranges
- `test_connector_authentication_error_handling`: Tests auth error handling

### Data Discovery Tests
- `test_discover_available_data`: Explores available data in your instance
- `test_discover_gen_ai_data`: Looks for GenAI-specific data

### Performance Tests
- `test_query_performance`: Measures single query performance
- `test_multiple_queries_performance`: Measures multiple query performance

## Testing with Known Data

If you have specific span IDs in your Application Insights data:

1. Edit the `manual_test_with_known_span_id()` function in the test file
2. Replace `"your-actual-span-id-here"` with a real span ID
3. Run the function directly:

```python
python tests/integration/test_span_hydrator.py
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Verify your credentials are correct
   - Check that you have read permissions on the Application Insights resource
   - Try using `az login` for local testing

2. **Resource Not Found**:
   - Verify the resource ID format is correct
   - Check that the Application Insights resource exists
   - Ensure you have access to the subscription and resource group

3. **No Data Returned**:
   - Check the time range (try increasing from hours to days)
   - Verify that your Application Insights instance has data
   - Use the data discovery tests to explore available data

4. **Timeout Errors**:
   - Increase the timeout in `ApplicationInsightsConfig`
   - Reduce the time range for queries
   - Check your network connection to Azure

### Debug Mode

Enable debug logging to see detailed query information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Query Validation

Test your queries directly using the Azure portal or Azure CLI:

```bash
az monitor log-analytics query \
  --workspace "/subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Insights/components/{name}" \
  --analytics-query "dependencies | take 5"
```

## CI/CD Integration

For automated testing in CI/CD pipelines:

1. Store credentials as secrets in your CI system
2. Use service principal authentication
3. Run with appropriate test markers:

```bash
# In CI pipeline
pytest -m "integration and not slow" tests/integration/test_span_hydrator.py
```
