"""
Application Insights connector for retrieving trace data.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json

from .interfaces import SourceConnector
from .utils import parse_timestamp, calculate_end_time
from ..models.events import (
    GenAIUserMessageEvent,
    GenAIAssistantMessageEvent,
    GenAISystemMessageEvent,
    GenAIToolMessageEvent,
    GenAIEventBase,
    GenAIChoiceEvent
)
from ..models import Span

try:
    from azure.monitor.query import LogsQueryClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ApplicationInsightsConfig:
    """Configuration for Application Insights connection."""
    resource_id: str
    credential: Optional[Any] = None
    timeout_seconds: int = 30
    max_retries: int = 3


class ApplicationInsightsConnector(SourceConnector):
    """
    Connector for retrieving trace data from Azure Application Insights.
    """
    
    def __init__(self, config: ApplicationInsightsConfig):
        """
        Initialize the Application Insights connector.
        
        Args:
            config: Configuration object for the connection
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure SDK not available. Install with: pip install azure-monitor-query azure-identity"
            )
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize the client
        credential = config.credential or DefaultAzureCredential()
        self.client = LogsQueryClient(credential)
    
    def query_span_by_id(self, span_id: str, time_range: timedelta = timedelta(days=30)) -> Optional[Span]:
        """
        Query a specific span by its ID using the GenAI-aware query and return a Span object.
        
        Args:
            span_id: The span ID to query for
            time_range: Time range to search within
            
        Returns:
            Span object or None if not found
        """
        # Calculate time range for the query
        end_time = datetime.utcnow()
        start_time = end_time - time_range
        
        kql_query = f"""
        let gen_ai_spans = dependencies
        // | where isnotnull(customDimensions["gen_ai.system"]) 
        | where timestamp between (datetime({start_time.isoformat()}Z) .. datetime({end_time.isoformat()}Z))
        | extend trace_id = operation_Id
        | extend span_id = id
        | extend parent_span_id = operation_ParentId
        | extend span_name = name
        | where id == "{span_id}"
        | project 
            operation_ParentId,
            operation_Id,
            customDimensions,
            span_id,
            parent_span_id,
            trace_id,
            duration,
            success,
            status = iff(success == "True", "Success", "Fail"),
            start_time = tostring(timestamp),
            span_name,
            span_type = tostring(customDimensions["gen_ai.operation.name"]);

        let get_role = (event_name: string) {{ 
            iff(event_name == "gen_ai.choice", "assistant", split(event_name, ".")[1])
        }};
        
        let gen_ai_events = traces
            | where message startswith ("gen_ai") or tostring(customDimensions["event.name"]) startswith ("gen_ai")
            | where timestamp between (datetime({start_time.isoformat()}Z) .. datetime({end_time.isoformat()}Z))
            | extend event_name = iff(message startswith ("gen_ai"), message, tostring(customDimensions["event.name"]))
            | extend gen_ai_system = tostring(customDimensions["gen_ai.system"])
            | extend event_content = iff(message startswith ("gen_ai"), tostring(customDimensions["gen_ai.event.content"]), message)
            | extend json = parse_json(event_content)
            | extend message_json = iff(event_name == "gen_ai.choice", json.message, json)
            | extend content = message_json.content
            | extend role = message_json.role
            | extend tool_calls = message_json.tool_calls
            | extend gen_ai_event = bag_pack(
                "event_name", event_name,
                "gen_ai_system", gen_ai_system,
                "body", json
                )
            | project 
                operation_ParentId,
                span_id = operation_ParentId,
                operation_Id,
                event_name,
                gen_ai_system,
                gen_ai_event;

        gen_ai_spans
        | join kind=leftouter ( 
            gen_ai_events
            | summarize events = make_list(coalesce(gen_ai_event, dynamic([])))
            by span_id
        ) on span_id
        | project 
             span_id,
             events, 
             trace_id, 
             parent_span_id, 
             span_name, 
             start_time, 
             status, 
             span_type,
             attributes = customDimensions,
             duration
        """
        
        results = self._execute_query(kql_query)
        if not results:
            return None
        
        span_data = results[0]
        
        # Parse events into typed GenAI events
        events = self._parse_span_events(span_data.get('events', []), span_data.get('span_id', ''))
        
        # Parse attributes
        attributes = self._parse_custom_dimensions(span_data.get('attributes', {}))
        
        # Create and return Span object
        return Span(
            span_id=span_data.get('span_id', ''),
            trace_id=span_data.get('trace_id', ''),
            operation_id=span_data.get('trace_id', ''),
            parent_span_id=span_data.get('parent_span_id'),
            name=span_data.get('span_name', ''),
            start_time=parse_timestamp(span_data.get('start_time')),
            end_time=calculate_end_time(
                span_data.get('start_time'), 
                span_data.get('duration', 0)
            ),
            duration_ms=float(span_data.get('duration', 0)),
            status=span_data.get('status', 'Unknown'),
            attributes=attributes,
            events=events,
            span_type=span_data.get('span_type', 'unknown')
        )
    
    def _parse_custom_dimensions(self, custom_dims: Any) -> Dict[str, Any]:
        """
        Parse customDimensions field which can be a string, dict, or None.
        
        Args:
            custom_dims: The customDimensions value from Application Insights
            
        Returns:
            Dictionary of parsed custom dimensions
        """
        if not custom_dims:
            return {}
        
        # If it's already a dictionary, return it
        if isinstance(custom_dims, dict):
            return custom_dims
        
        # If it's a string, try to parse it as JSON
        if isinstance(custom_dims, str):
            try:
                parsed = json.loads(custom_dims)
                # Ensure the parsed result is a dictionary
                if isinstance(parsed, dict):
                    return parsed
                else:
                    self.logger.warning(f"customDimensions parsed to non-dict type: {type(parsed)}")
                    return {}
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.warning(f"Failed to parse customDimensions as JSON: {e}")
                # If JSON parsing fails, treat as a single key-value pair
                return {"raw_value": str(custom_dims)}
        
        # For any other type, convert to string and store
        self.logger.warning(f"Unknown customDimensions type: {type(custom_dims)}")
        return {"raw_value": str(custom_dims)}
    
    def query_spans_by_operation_id(self, operation_id: str, time_range: timedelta = timedelta(hours=1)) -> List[Dict[str, Any]]:
        """
        Query all spans for a specific operation ID using the GenAI-aware query.
        
        Args:
            operation_id: The operation ID to query for
            time_range: Time range to search within
            
        Returns:
            List of span dictionaries
        """
        # Calculate time range for the query
        end_time = datetime.utcnow()
        start_time = end_time - time_range
        
        kql_query = f"""
        let gen_ai_spans = dependencies
        // | where isnotnull(customDimensions["gen_ai.system"]) 
        | where timestamp between (datetime({start_time.isoformat()}Z) .. datetime({end_time.isoformat()}Z))
        | extend trace_id = operation_Id
        | extend span_id = id
        | extend parent_span_id = operation_ParentId
        | extend span_name = name
        | where operation_Id == "{operation_id}"
        | project 
            operation_ParentId,
            operation_Id,
            customDimensions,
            span_id,
            parent_span_id,
            trace_id,
            duration,
            success,
            status = iff(success == "True", "Success", "Fail"),
            start_time = tostring(timestamp),
            span_name,
            span_type = tostring(customDimensions["gen_ai.operation.name"]);

        let get_role = (event_name: string) {{ 
            iff(event_name == "gen_ai.choice", "assistant", split(event_name, ".")[1])
        }};

        let gen_ai_events = traces
            | where operation_Id == "{operation_id}"
            | where message startswith ("gen_ai") or tostring(customDimensions["event.name"]) startswith ("gen_ai")
            | where timestamp between (datetime({start_time.isoformat()}Z) .. datetime({end_time.isoformat()}Z))
            | extend event_name = iff(message startswith ("gen_ai"), message, tostring(customDimensions["event.name"]))
            | extend gen_ai_system = iff(message startswith ("gen_ai"), message, tostring(customDimensions["gen_ai.system"]))
            | extend event_content = iff(message startswith ("gen_ai"), tostring(customDimensions["gen_ai.event.content"]), message)
            | extend json = parse_json(event_content)
            | extend content = iff(event_name == "gen_ai.choice", json.message, json)
            | extend gen_ai_event = bag_pack(
                "event_name", event_name,
                "gen_ai_system", gen_ai_system,
                "content", content
                )
            | project 
                operation_ParentId,
                span_id = operation_ParentId,
                operation_Id,
                event_name,
                gen_ai_system,
                content = bag_merge(bag_pack("role", get_role(event_name)), content),
                gen_ai_event;

        gen_ai_spans
        | join kind=leftouter ( 
            gen_ai_events
            | summarize events = make_list(coalesce(gen_ai_event, dynamic([])))
            by span_id
        ) on span_id
        | project 
             id = span_id,
             operation_Id = trace_id,
             operation_ParentId = parent_span_id,
             name = span_name,
             timestamp = start_time,
             duration,
             resultCode = status,
             itemType = span_type,
             customDimensions = attributes,
             events
        | order by timestamp asc
        """
        
        return self._execute_query(kql_query)
    
    def query_child_spans(self, parent_span_id: str, time_range: timedelta = timedelta(days=30), 
                         gen_ai_operation_name: str = None) -> List[Span]:
        """
        Query all child spans of a given parent span using the same GenAI-aware query as query_span_by_id.
        
        Args:
            parent_span_id: The parent span ID
            time_range: Time range to search within
            gen_ai_operation_name: Optional filter for gen_ai.operation.name in customDimensions
            
        Returns:
            List of hydrated Span objects
        """
        # Calculate time range for the query
        end_time = datetime.utcnow()
        start_time = end_time - time_range
        
        # Build the filter clause for gen_ai.operation.name if provided
        gen_ai_filter = ""
        if gen_ai_operation_name:
            gen_ai_filter = f'| where tostring(customDimensions["gen_ai.operation.name"]) == "{gen_ai_operation_name}"'
        
        kql_query = f"""
        let gen_ai_spans = dependencies
        // | where isnotnull(customDimensions["gen_ai.system"]) 
        | where timestamp between (datetime({start_time.isoformat()}Z) .. datetime({end_time.isoformat()}Z))
        | extend trace_id = operation_Id
        | extend span_id = id
        | extend parent_span_id = operation_ParentId
        | extend span_name = name
        | where operation_ParentId == "{parent_span_id}"
        | project 
            operation_ParentId,
            operation_Id,
            customDimensions,
            span_id,
            parent_span_id,
            trace_id,
            duration,
            success,
            status = iff(success == "True", "Success", "Fail"),
            start_time = tostring(timestamp),
            span_name,
            span_type = tostring(customDimensions["gen_ai.operation.name"]);

        let get_role = (event_name: string) {{ 
            iff(event_name == "gen_ai.choice", "assistant", split(event_name, ".")[1])
        }};
        
        let gen_ai_events = traces
            | where message startswith ("gen_ai") or tostring(customDimensions["event.name"]) startswith ("gen_ai")
            | where timestamp between (datetime({start_time.isoformat()}Z) .. datetime({end_time.isoformat()}Z))
            | extend event_name = iff(message startswith ("gen_ai"), message, tostring(customDimensions["event.name"]))
            | extend gen_ai_system = tostring(customDimensions["gen_ai.system"])
            | extend event_content = iff(message startswith ("gen_ai"), tostring(customDimensions["gen_ai.event.content"]), message)
            | extend json = parse_json(event_content)
            | extend message_json = iff(event_name == "gen_ai.choice", json.message, json)
            | extend content = message_json.content
            | extend role = message_json.role
            | extend tool_calls = message_json.tool_calls
            | extend gen_ai_event = bag_pack(
                "event_name", event_name,
                "gen_ai_system", gen_ai_system,
                "body", json
                )
            | project 
                operation_ParentId,
                span_id = operation_ParentId,
                operation_Id,
                event_name,
                gen_ai_system,
                gen_ai_event;

        gen_ai_spans
        | join kind=leftouter ( 
            gen_ai_events
            | summarize events = make_list(coalesce(gen_ai_event, dynamic([])))
            by span_id
        ) on span_id
        | project 
             span_id,
             events, 
             trace_id, 
             parent_span_id, 
             span_name, 
             start_time, 
             status, 
             span_type,
             attributes = customDimensions,
             duration
        | order by start_time asc
        """
        
        results = self._execute_query(kql_query, timespan=time_range)
        
        # Convert results to Span objects using the same logic as query_span_by_id
        spans = []
        for span_data in results:
            # Parse events into typed GenAI events
            events = self._parse_span_events(span_data.get('events', []), span_data.get('span_id', ''))
            
            # Parse attributes
            attributes = self._parse_custom_dimensions(span_data.get('attributes', {}))
            
            # Create Span object
            span = Span(
                span_id=span_data.get('span_id', ''),
                trace_id=span_data.get('trace_id', ''),
                operation_id=span_data.get('trace_id', ''),
                parent_span_id=span_data.get('parent_span_id'),
                name=span_data.get('span_name', ''),
                start_time=parse_timestamp(span_data.get('start_time')),
                end_time=calculate_end_time(
                    span_data.get('start_time'), 
                    span_data.get('duration', 0)
                ),
                duration_ms=float(span_data.get('duration', 0)),
                status=span_data.get('status', 'Unknown'),
                attributes=attributes,
                events=events,
                span_type=span_data.get('span_type', 'unknown')
            )
            spans.append(span)
        
        return spans
    
    def _execute_query(self, kql_query: str, timespan=timedelta(30)) -> List[Dict[str, Any]]:
        """
        Execute a KQL query against Application Insights.
        
        Args:
            kql_query: The KQL query to execute
            
        Returns:
            List of result dictionaries
        """
        try:
            self.logger.debug(f"Executing KQL query: {kql_query}")
            
            response = self.client.query_resource(
                resource_id=self.config.resource_id,
                query=kql_query,
                timespan=timespan  # Default timespan - 30 days
            )
            
            if response.status == "Success":
                # Convert response to list of dictionaries
                results = []
                if response.tables:
                    table = response.tables[0]
                    # columns = [col.name for col in table.columns]
                    
                    for row in table.rows:
                        row_dict = {}
                        for i, value in enumerate(row):
                            column_name = table.columns[i]
                            # Special handling for customDimensions column
                            if column_name == 'customDimensions':
                                row_dict[column_name] = self._parse_custom_dimensions(value)
                            else:
                                row_dict[column_name] = value
                        results.append(row_dict)
                
                self.logger.info(f"Query returned {len(results)} results")
                return results
            else:
                self.logger.error(f"Query failed with status: {response.status}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test the connection to Application Insights.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            test_query = "union dependencies, requests | take 1"
            self._execute_query(test_query)
            self.logger.info("Connection to Application Insights successful")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def _parse_span_events(self, events_data: List[Dict[str, Any]], span_id: str) -> List[Union[GenAIEventBase, Dict[str, Any]]]:
        """
        Parse events from KQL query results into typed Pydantic GenAI events when possible.
        
        Args:
            events_data: List of event dictionaries from KQL query
            span_id: The span ID for logging purposes
            
        Returns:
            List of typed Pydantic GenAI event objects or raw event dictionaries
        """
        if not events_data:
            return []
        
        # First, try to parse events_data as JSON if it's a string
        parsed_events_data = events_data
        if isinstance(events_data, str):
            try:
                parsed_events_data = json.loads(events_data)
                self.logger.debug(f"Successfully parsed events_data as JSON for span '{span_id}'")
            except (json.JSONDecodeError, TypeError) as e:
                self.logger.warning(f"Failed to parse events_data as JSON for span '{span_id}': {e}")
                # If it's a string but not valid JSON, treat as empty
                return []
        elif not isinstance(events_data, list):
            # If it's not a string or list, try to convert to list
            try:
                parsed_events_data = [events_data] if events_data else []
            except:
                self.logger.warning(f"events_data is not a list or string for span '{span_id}', got: {type(events_data)}")
                return []
        
        processed_events = []
        
        # Mapping of event names to their corresponding Pydantic classes
        event_class_map = {
            'gen_ai.user.message': GenAIUserMessageEvent,
            'gen_ai.assistant.message': GenAIAssistantMessageEvent,
            'gen_ai.system.message': GenAISystemMessageEvent,
            'gen_ai.tool.message': GenAIToolMessageEvent,
            'gen_ai.choice': GenAIChoiceEvent  # Placeholder for choice events
        }
        
        for event in parsed_events_data:
            if not isinstance(event, dict):
                continue
                
            try:
                # Extract event information
                event_name = event.get('event_name')
                
                # Check if this is a known GenAI event type
                if event_name in event_class_map:
                    event_class = event_class_map[event_name]
                    
                    try:
                        # Use Pydantic's model_validate to create the event instance
                        parsed_event = event_class.model_validate(event)
                        
                        processed_events.append(parsed_event)
                        self.logger.debug(f"Successfully parsed Pydantic GenAI event '{event_name}' of type {type(parsed_event).__name__} for span '{span_id}'")
                        continue
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to parse '{event_name}' as Pydantic event for span '{span_id}': {e}. Keeping as raw event.")
                
                # If not a known GenAI event type or parsing failed, keep as raw dictionary
                processed_events.append(event)
                self.logger.debug(f"Kept raw event '{event_name}' for span '{span_id}' (not a recognized GenAI event type or parsing failed)")
                    
            except Exception as e:
                self.logger.warning(f"Error parsing event for span '{span_id}': {e}. Keeping as raw event.")
                # Keep the original event data if parsing fails
                processed_events.append(event)
        
        return processed_events
