"""
Real integration tests for SpanHydrator with Application Insights.

These tests use real Application Insights connections and data.
They require Azure credentials and a valid Application Insights resource.

Set the following environment variables to run these tests:
- AZURE_APPLICATION_INSIGHTS_RESOURCE_ID: The resource ID of your Application Insights instance
- AZURE_TENANT_ID: Your Azure tenant ID (optional, for specific credential configuration)
- AZURE_CLIENT_ID: Your Azure client ID (optional, for service principal auth)
- AZURE_CLIENT_SECRET: Your Azure client secret (optional, for service principal auth)

Run with: pytest tests/integration/test_span_hydrator.py::TestRealApplicationInsightsIntegration -v
"""

import pytest
import os
from datetime import datetime, timedelta
from typing import Optional

from otel_gen_ai_hydrator.span_hydrator import SpanHydrator
from otel_gen_ai_hydrator.sources.application_insights import (
    ApplicationInsightsConnector, 
    ApplicationInsightsConfig
)

try:
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.monitor.query import LogsQueryClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


def get_azure_credential():
    """Get Azure credential based on environment variables."""
    tenant_id = os.getenv('AZURE_TENANT_ID')
    client_id = os.getenv('AZURE_CLIENT_ID')
    client_secret = os.getenv('AZURE_CLIENT_SECRET')
    
    if tenant_id and client_id and client_secret:
        return ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
    else:
        return DefaultAzureCredential()


@pytest.fixture
def application_insights_resource_id():
    """Get Application Insights resource ID from environment."""
    resource_id = os.getenv('AZURE_APPLICATION_INSIGHTS_RESOURCE_ID', "/subscriptions/b17253fa-f327-42d6-9686-f3e553e24763/resourceGroups/anksing-vanilla-eval/providers/microsoft.insights/components/anksing-app-insights")
    if not resource_id:
        pytest.skip("AZURE_APPLICATION_INSIGHTS_RESOURCE_ID environment variable not set")
    return resource_id


@pytest.fixture
def azure_credential():
    """Get Azure credential for authentication."""
    if not AZURE_AVAILABLE:
        pytest.skip("Azure SDK not available. Install with: pip install azure-monitor-query azure-identity")
    return get_azure_credential()


@pytest.fixture
def app_insights_config(application_insights_resource_id, azure_credential):
    """Create Application Insights configuration."""
    return ApplicationInsightsConfig(
        resource_id=application_insights_resource_id,
        credential=azure_credential,
        timeout_seconds=60,
        max_retries=3
    )


@pytest.fixture
def app_insights_connector(app_insights_config):
    """Create Application Insights connector."""
    return ApplicationInsightsConnector(app_insights_config)


@pytest.fixture
def span_hydrator_real(app_insights_connector):
    """Create SpanHydrator with real Application Insights connector."""
    return SpanHydrator(app_insights_connector)


@pytest.mark.integration
@pytest.mark.real_data
class TestRealApplicationInsightsIntegration:
    """Integration tests using real Application Insights data."""
    
    def test_connection_to_application_insights(self, app_insights_connector):
        """Test that we can connect to Application Insights."""
        # Act
        connection_result = app_insights_connector.test_connection()
        
        # Assert
        assert connection_result is True, "Failed to connect to Application Insights"
    
    def test_query_recent_spans(self, span_hydrator_real):
        """Test querying for recent spans in Application Insights."""
        # This test looks for any spans in the last 7 days
        # Note: This might return empty results if no spans exist in the timeframe
        
        # We'll use a broad time range to increase chances of finding data
        time_range = timedelta(days=7)
        
        # We can't test specific span IDs since we don't know what exists
        # Instead, let's test the connector's ability to handle the query
        # This would typically be used with known span IDs from your application
        
        # For demonstration, we'll try a span ID that might not exist
        result = span_hydrator_real.get_span_by_id("2bdfce7bb6a26baa", time_range=time_range)
        
        # This should return None for a non-existent span, which is expected behavior
        assert hasattr(result, 'span_id'), "Query should return None or a valid Span object"
    
    def test_query_child_spans_with_real_data(self, span_hydrator_real):
        """Test querying for child spans."""
        # This test demonstrates how to query for child spans
        # In a real scenario, you would have actual parent span IDs
        
        time_range = timedelta(days=7)
        
        # Try to get child spans for a non-existent parent
        children = span_hydrator_real.get_child_spans("non-existent-parent", time_range=time_range)
        
        # Should return empty list for non-existent parent
        assert isinstance(children, list), "get_child_spans should return a list"
        # The list might be empty, which is expected for non-existent parent
    
    def test_query_child_spans_with_gen_ai_filter(self, span_hydrator_real):
        """Test querying for child spans with GenAI operation filter."""
        time_range = timedelta(days=7)
        
        # Test filtering by gen_ai.operation.name
        chat_completion_spans = span_hydrator_real.get_child_spans(
            "non-existent-parent", 
            time_range=time_range,
            gen_ai_operation_name="chat.completions"
        )
        
        assert isinstance(chat_completion_spans, list), "Filtered query should return a list"
        
        # If any spans are returned, they should have the correct operation name
        for span in chat_completion_spans:
            assert hasattr(span, 'attributes'), "Span should have attributes"
            if 'gen_ai.operation.name' in span.attributes:
                assert span.attributes['gen_ai.operation.name'] == 'chat.completions'
    
    def test_error_handling_with_invalid_time_range(self, span_hydrator_real):
        """Test error handling with invalid time ranges."""
        # Test with a very large time range that might cause issues
        large_time_range = timedelta(days=365)  # 1 year
        
        result = span_hydrator_real.get_span_by_id("test-span", time_range=large_time_range)
        
        # Should handle large time ranges gracefully
        assert result is None or hasattr(result, 'span_id'), "Should handle large time ranges gracefully"
    
    def test_connector_authentication_error_handling(self, application_insights_resource_id):
        """Test handling of authentication errors."""
        # Create a connector with invalid credentials (None)
        invalid_config = ApplicationInsightsConfig(
            resource_id=application_insights_resource_id,
            credential=None,  # This should cause authentication issues
            timeout_seconds=30
        )
        
        # This might raise an exception during initialization or during first query
        try:
            invalid_connector = ApplicationInsightsConnector(invalid_config)
            span_hydrator = SpanHydrator(invalid_connector)
            
            # Try to query - this should handle authentication errors gracefully
            result = span_hydrator.get_span_by_id("test-span")
            
            # Should return None on authentication failure
            assert result is None, "Should return None on authentication failure"
            
        except Exception as e:
            # Authentication errors during initialization are also acceptable
            assert "credential" in str(e).lower() or "auth" in str(e).lower(), f"Unexpected error: {e}"


@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.parametrize("time_range_days", [1, 7, 30])
class TestTimeRangeVariations:
    """Test different time ranges with real data."""
    
    def test_different_time_ranges(self, span_hydrator_real, time_range_days):
        """Test querying with different time ranges."""
        time_range = timedelta(days=time_range_days)
        
        # Query for a non-existent span across different time ranges
        result = span_hydrator_real.get_span_by_id(f"test-span-{time_range_days}d", time_range=time_range)
        
        # Should handle all time ranges consistently
        assert result is None or hasattr(result, 'span_id'), f"Should handle {time_range_days}-day time range"


@pytest.mark.integration
@pytest.mark.real_data
class TestRealDataScenarios:
    """Tests that work with actual data if available."""
    
    def test_discover_available_data(self, app_insights_connector):
        """Discover what data is available in the Application Insights instance."""
        # This test helps understand what data is in your Application Insights
        # It uses a direct query to explore the data
        
        try:
            # Query for recent dependencies (which might include GenAI spans)
            recent_deps_query = """
            dependencies
            | where timestamp > ago(7d)
            | where isnotnull(customDimensions)
            | project id, name, timestamp, duration, success, customDimensions
            | take 5
            """
            
            results = app_insights_connector._execute_query(recent_deps_query, timespan=timedelta(days=7))
            
            # This test is informational - it helps you understand your data
            print(f"\\nFound {len(results)} recent dependency records")
            for i, result in enumerate(results[:3]):  # Show first 3 records
                print(f"Record {i+1}: ID={result.get('id')}, Name={result.get('name')}")
                if result.get('customDimensions'):
                    print(f"  CustomDimensions keys: {list(result.get('customDimensions', {}).keys())}")
            
            # Assert that we can execute queries (even if no results)
            assert isinstance(results, list), "Query should return a list"
            
        except Exception as e:
            pytest.fail(f"Failed to query for available data: {e}")
    
    def test_discover_gen_ai_data(self, app_insights_connector):
        """Discover GenAI-specific data in Application Insights."""
        try:
            # Query for any records that might be GenAI related
            gen_ai_query = '''
            union dependencies, traces
            | where timestamp > ago(7d)
            | where 
                tostring(customDimensions) contains "gen_ai" or
                message contains "gen_ai" or
                name contains "openai" or
                name contains "azure" or
                name contains "chat" or
                name contains "embedding"
            | project timestamp, itemType, name, message, customDimensions
            | take 10
            '''
            
            results = app_insights_connector._execute_query(gen_ai_query, timespan=timedelta(days=7))
            
            print(f"\\nFound {len(results)} potentially GenAI-related records")
            for i, result in enumerate(results[:3]):
                print(f"GenAI Record {i+1}: Type={result.get('itemType')}, Name={result.get('name')}")
                if result.get('message'):
                    print(f"  Message: {result.get('message')[:100]}...")
            
            assert isinstance(results, list), "GenAI query should return a list"
            
        except Exception as e:
            pytest.fail(f"Failed to query for GenAI data: {e}")


@pytest.mark.integration
@pytest.mark.real_data
@pytest.mark.slow
class TestPerformanceWithRealData:
    """Performance tests with real Application Insights data."""
    
    def test_query_performance(self, span_hydrator_real):
        """Test query performance with real data."""
        import time
        
        start_time = time.time()
        
        # Perform a query that should complete reasonably quickly
        result = span_hydrator_real.get_span_by_id("perf-test-span", time_range=timedelta(days=1))
        
        end_time = time.time()
        query_duration = end_time - start_time
        
        # Query should complete within reasonable time (adjust as needed)
        assert query_duration < 30.0, f"Query took too long: {query_duration:.2f} seconds"
        
        # Result handling should be consistent
        assert result is None or hasattr(result, 'span_id'), "Query result should be valid"
    
    def test_multiple_queries_performance(self, span_hydrator_real):
        """Test performance of multiple consecutive queries."""
        import time
        
        start_time = time.time()
        
        # Perform multiple queries
        test_span_ids = [f"perf-test-{i}" for i in range(3)]
        results = []
        
        for span_id in test_span_ids:
            result = span_hydrator_real.get_span_by_id(span_id, time_range=timedelta(hours=6))
            results.append(result)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Multiple queries should complete within reasonable time
        assert total_duration < 60.0, f"Multiple queries took too long: {total_duration:.2f} seconds"
        
        # All results should be None or valid Span objects
        for result in results:
            assert result is None or hasattr(result, 'span_id'), "All query results should be valid"


# Helper function for manual testing with specific span IDs
def manual_test_with_known_span_id():
    """
    Helper function for manual testing when you have known span IDs.
    
    Usage:
    1. Set environment variables for your Application Insights resource
    2. Replace 'your-actual-span-id' with a real span ID from your data
    3. Run this function directly to test with real data
    """
    if not AZURE_AVAILABLE:
        print("Azure SDK not available")
        return
    
    resource_id = os.getenv('AZURE_APPLICATION_INSIGHTS_RESOURCE_ID')
    if not resource_id:
        print("AZURE_APPLICATION_INSIGHTS_RESOURCE_ID not set")
        return
    
    config = ApplicationInsightsConfig(
        resource_id=resource_id,
        credential=get_azure_credential()
    )
    
    connector = ApplicationInsightsConnector(config)
    hydrator = SpanHydrator(connector)
    
    # Test connection
    if not connector.test_connection():
        print("Failed to connect to Application Insights")
        return
    
    # Replace with actual span ID from your Application Insights data
    known_span_id = "your-actual-span-id-here"
    
    print(f"Testing with span ID: {known_span_id}")
    
    # Test span retrieval
    span = hydrator.get_span_by_id(known_span_id, time_range=timedelta(days=7))
    
    if span:
        print(f"✅ Found span: {span.name}")
        print(f"   Trace ID: {span.trace_id}")
        print(f"   Duration: {span.duration_ms}ms")
        print(f"   Status: {span.status}")
        print(f"   Events: {len(span.events)}")
        print(f"   Attributes: {list(span.attributes.keys())}")
        
        # Test child span retrieval
        children = hydrator.get_child_spans(known_span_id, time_range=timedelta(days=7))
        print(f"   Child spans: {len(children)}")
        
    else:
        print("❌ Span not found")


if __name__ == "__main__":
    # Run manual test if executed directly
    manual_test_with_known_span_id()
