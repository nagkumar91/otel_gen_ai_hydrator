"""
Complete script to extract traces from Application Insights, convert to OpenTelemetry format,
evaluate them, and export everything including evaluation results.
Follows OpenTelemetry semantic conventions for GenAI: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict
import hashlib

from dotenv import load_dotenv
from otel_gen_ai_hydrator.sources.application_insights import (
    ApplicationInsightsConnector,
    ApplicationInsightsConfig,
)

from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    ToolCallAccuracyEvaluator,
    IntentResolutionEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
)

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry._events import Event, get_event_logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Azure Monitor for evaluation events
configure_azure_monitor(
    connection_string=os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING", "")
)

# Event logger for emitting evaluation events
event_logger = get_event_logger(__name__)


class SpanDataExtractor:
    """Helper class to extract data from spans for evaluation."""
    
    @staticmethod
    def extract_messages_from_span(span: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract all messages from a span's logs, organized by role."""
        messages = {
            "system": [],
            "user": [],
            "assistant": [],
            "tool": []
        }
        
        for log in span.get("logs", []):
            fields = {f.get("key", ""): f.get("value", "") for f in log.get("fields", [])}
            
            # Check for GenAI prompt/completion events
            if fields.get("event.name") == "gen_ai.content.prompt":
                if "gen_ai.prompt" in fields:
                    try:
                        prompt_data = json.loads(fields["gen_ai.prompt"]) if isinstance(fields["gen_ai.prompt"], str) else fields["gen_ai.prompt"]
                        role = prompt_data.get("role", "")
                        content = prompt_data.get("content", "")
                        
                        if role == "system":
                            messages["system"].append({"role": role, "content": content})
                        elif role in ["user", "human"]:
                            messages["user"].append({"role": "user", "content": content})
                        elif role in ["ai", "assistant"]:
                            messages["assistant"].append({"role": "assistant", "content": content})
                        elif role == "tool":
                            messages["tool"].append({"role": "tool", "content": content})
                    except Exception as e:
                        logger.debug(f"Error parsing prompt data: {e}")
            
            elif fields.get("event.name") == "gen_ai.content.completion":
                if "gen_ai.event.content" in fields:
                    try:
                        event_data = json.loads(fields["gen_ai.event.content"]) if isinstance(fields["gen_ai.event.content"], str) else fields["gen_ai.event.content"]
                        content = event_data.get("content", "")
                        if content:
                            messages["assistant"].append({"role": "assistant", "content": content})
                    except Exception as e:
                        logger.debug(f"Error parsing completion data: {e}")
        
        return messages
    
    @staticmethod
    def extract_tool_calls_from_span(span: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from a span."""
        tool_calls = []
        
        for log in span.get("logs", []):
            fields = {f.get("key", ""): f.get("value", "") for f in log.get("fields", [])}
            
            # Check for tool calls in completion events
            if "gen_ai.tool_calls" in fields:
                try:
                    calls_data = json.loads(fields["gen_ai.tool_calls"]) if isinstance(fields["gen_ai.tool_calls"], str) else fields["gen_ai.tool_calls"]
                    if isinstance(calls_data, list):
                        tool_calls.extend(calls_data)
                except Exception as e:
                    logger.debug(f"Error parsing tool calls: {e}")
        
        return tool_calls
       
    @staticmethod
    def find_final_response_content(span: Dict[str, Any]) -> Optional[str]:
        """Find the final response content in a span."""
        messages = SpanDataExtractor.extract_messages_from_span(span)
        
        for msg in messages["assistant"]:
            content = msg.get("content", "")
            if "FINAL TRAVEL PLAN" in content.upper():
                return content
        
        if messages["assistant"]:
            return messages["assistant"][-1].get("content", "")
        
        return None


def create_evaluation_event(
    *,
    name: str,
    score: float,
    reasoning: str,
    span_id: str,
    trace_id: str,
    metadata: Dict[str, Any] = None
) -> Event:
    """Create an OpenTelemetry event for AI evaluation results."""
    try:
        span_id_int = int(span_id, 16) if isinstance(span_id, str) else span_id
        trace_id_int = int(trace_id, 16) if isinstance(trace_id, str) else trace_id
    except:
        span_id_int = 0
        trace_id_int = 0
    
    attributes = {
        "gen_ai.evaluation.name": name,
        "gen_ai.evaluation.score": score,
        "gen_ai.evaluation.reasoning": reasoning,
        "gen_ai.evaluation.result": "pass" if score >= 3.0 else "fail",
        "gen_ai.evaluation.span_id": span_id,
        "gen_ai.evaluation.trace_id": trace_id,
    }
    
    if metadata:
        for key, value in metadata.items():
            attributes[f"gen_ai.evaluation.{key}"] = str(value)
        
    return Event(
        name=f"gen_ai.evaluation.{name}",
        attributes=attributes,
        body=f"Evaluation for {name}: score={score}",
        span_id=span_id_int,
        trace_id=trace_id_int,
    )


def emit_evaluation_event(event: Event):
    """Emit an OTel event into App Insights."""
    event_logger.emit(event)


def convert_to_otel_compliant_format(trace_records: List[Dict[str, Any]], trace_id: str) -> Dict[str, Any]:
    """
    Convert Application Insights trace records to OpenTelemetry compliant format.
    Follows the semantic conventions from: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
    """
    grouped_records = defaultdict(list)
    for record in trace_records:
        item_type = record.get('itemType', 'unknown')
        grouped_records[item_type].append(record)
    
    spans = []
    
    # Process dependencies as spans
    for dep in grouped_records.get('dependency', []):
        span = convert_dependency_to_otel_span(dep, trace_id)
        if span:
            spans.append(span)
    
    # Process requests as spans  
    for req in grouped_records.get('request', []):
        span = convert_request_to_otel_span(req, trace_id)
        if span:
            spans.append(span)
    
    # Add trace events as logs to their corresponding spans
    add_trace_logs_to_spans(spans, grouped_records.get('trace', []))
    
    # Build the OpenTelemetry compliant structure
    otel_trace = {
        "data": [
            {
                "traceID": trace_id,
                "spans": spans,
                "processes": {
                    "p1": {
                        "serviceName": "travel-planning-system",
                        "tags": [
                            {"key": "service.name", "type": "string", "value": "travel-planning-system"},
                            {"key": "service.version", "type": "string", "value": "1.0.0"},
                            {"key": "deployment.environment", "type": "string", "value": "development"},
                            {"key": "telemetry.sdk.name", "type": "string", "value": "opentelemetry"},
                            {"key": "telemetry.sdk.language", "type": "string", "value": "python"},
                            {"key": "telemetry.sdk.version", "type": "string", "value": "1.27.0"}
                        ]
                    }
                },
                "warnings": None
            }
        ],
        "total": 0,
        "limit": 0,
        "offset": 0,
        "errors": None
    }
    
    return otel_trace


def convert_dependency_to_otel_span(dep: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
    """Convert a dependency record to an OpenTelemetry compliant span."""
    try:
        custom_dims = dep.get('customDimensions', {})
        if isinstance(custom_dims, str):
            try:
                custom_dims = json.loads(custom_dims)
            except:
                custom_dims = {}
        
        span_id = dep.get('id', '').replace('-', '')[:16] or generate_span_id(dep)
        parent_span_id = dep.get('operation_ParentId', '').replace('-', '')[:16]
        
        timestamp = parse_timestamp(dep.get('timestamp'))
        duration_ms = float(dep.get('duration', 0))
        
        operation_name = dep.get('name', 'unknown')
        gen_ai_operation_name = custom_dims.get('gen_ai.operation.name', '')
        
        # Build span with OpenTelemetry semantic conventions
        span = {
            "traceID": trace_id,
            "spanID": span_id,
            "operationName": operation_name,
            "references": [],
            "startTime": int(timestamp.timestamp() * 1_000_000),  # microseconds
            "duration": int(duration_ms * 1000),  # microseconds
            "tags": [],
            "logs": [],
            "processID": "p1",
            "warnings": None
        }
        
        # Add parent reference if exists
        if parent_span_id:
            span["references"].append({
                "refType": "CHILD_OF",
                "traceID": trace_id,
                "spanID": parent_span_id
            })
        
        # Add OpenTelemetry semantic convention tags
        otel_tags = []
        
        # Core span attributes
        otel_tags.extend([
            {"key": "span.kind", "type": "string", "value": custom_dims.get('span.kind', 'internal')},
            {"key": "otel.status_code", "type": "string", "value": "OK" if dep.get('success', True) else "ERROR"},
        ])
        
        # GenAI specific attributes
        if gen_ai_operation_name:
            otel_tags.append({"key": "gen_ai.operation.name", "type": "string", "value": gen_ai_operation_name})
        
        # System and model info
        for key in ['gen_ai.system', 'gen_ai.request.model', 'gen_ai.response.model']:
            if key in custom_dims:
                otel_tags.append({"key": key, "type": "string", "value": custom_dims[key]})
        
        # Usage metrics
        for key in ['gen_ai.usage.input_tokens', 'gen_ai.usage.output_tokens', 'gen_ai.usage.total_tokens']:
            if key in custom_dims:
                otel_tags.append({"key": key, "type": "int64", "value": int(custom_dims[key])})
        
        # Tool-specific attributes
        if "execute_tool" in operation_name.lower():
            tool_name = custom_dims.get('gen_ai.tool.name', operation_name.replace('execute_tool', '').strip())
            otel_tags.append({"key": "gen_ai.tool.name", "type": "string", "value": tool_name})
            
            if 'gen_ai.tool.call.id' in custom_dims:
                otel_tags.append({"key": "gen_ai.tool.call.id", "type": "string", "value": custom_dims['gen_ai.tool.call.id']})
        
        # Add resource attributes
        if '_MS.ResourceAttributeId' in custom_dims:
            otel_tags.append({"key": "azure.resource.id", "type": "string", "value": custom_dims['_MS.ResourceAttributeId']})
        
        span["tags"] = otel_tags
        
        return span
        
    except Exception as e:
        logger.error(f"Error converting dependency to OTEL span: {e}")
        return None


def convert_request_to_otel_span(req: Dict[str, Any], trace_id: str) -> Optional[Dict[str, Any]]:
    """Convert a request record to an OpenTelemetry compliant span."""
    try:
        custom_dims = req.get('customDimensions', {})
        if isinstance(custom_dims, str):
            try:
                custom_dims = json.loads(custom_dims)
            except:
                custom_dims = {}
        
        span_id = req.get('id', '').replace('-', '')[:16] or generate_span_id(req)
        
        timestamp = parse_timestamp(req.get('timestamp'))
        duration_ms = float(req.get('duration', 0))
        
        span = {
            "traceID": trace_id,
            "spanID": span_id,
            "operationName": req.get('name', 'unknown'),
            "references": [],  # Requests are typically root spans
            "startTime": int(timestamp.timestamp() * 1_000_000),  # microseconds
            "duration": int(duration_ms * 1000),  # microseconds
            "tags": [],
            "logs": [],
            "processID": "p1",
            "warnings": None
        }
        
        # Add OpenTelemetry semantic convention tags
        otel_tags = [
            {"key": "span.kind", "type": "string", "value": "server"},
            {"key": "otel.status_code", "type": "string", "value": "OK" if req.get('success', True) else "ERROR"},
        ]
        
        # HTTP semantic conventions
        if req.get('url'):
            otel_tags.extend([
                {"key": "http.method", "type": "string", "value": "POST"},
                {"key": "http.url", "type": "string", "value": req.get('url', '')},
                {"key": "http.status_code", "type": "int64", "value": int(req.get('resultCode', 200))},
            ])
        
        # Add GenAI attributes from custom dimensions
        for key, value in custom_dims.items():
            if key.startswith('gen_ai.'):
                otel_tags.append({
                    "key": key,
                    "type": get_value_type(value),
                    "value": str(value)
                })
        
        span["tags"] = otel_tags
        
        return span
        
    except Exception as e:
        logger.error(f"Error converting request to OTEL span: {e}")
        return None


def add_trace_logs_to_spans(spans: List[Dict[str, Any]], traces: List[Dict[str, Any]]) -> None:
    """Add trace records as logs/events to their corresponding spans following OTEL conventions."""
    span_map = {span["spanID"]: span for span in spans}
    
    for trace in traces:
        parent_id = trace.get('operation_ParentId', '').replace('-', '')[:16]
        if parent_id in span_map:
            timestamp = parse_timestamp(trace.get('timestamp'))
            
            custom_dims = trace.get('customDimensions', {})
            if isinstance(custom_dims, str):
                try:
                    custom_dims = json.loads(custom_dims)
                except:
                    custom_dims = {}
            
            # Check if this is a GenAI event
            message = trace.get('message', '')
            if message in ['gen_ai.content.prompt', 'gen_ai.content.completion', 'gen_ai.tool.call']:
                event_fields = [
                    {"key": "event.name", "type": "string", "value": message}
                ]
                
                # Add GenAI specific fields
                if 'gen_ai.prompt' in custom_dims:
                    event_fields.append({
                        "key": "gen_ai.prompt",
                        "type": "string", 
                        "value": custom_dims['gen_ai.prompt']
                    })
                
                if 'gen_ai.event.content' in custom_dims:
                    event_fields.append({
                        "key": "gen_ai.event.content",
                        "type": "string",
                        "value": custom_dims['gen_ai.event.content']
                    })
                
                if 'tool_calls' in custom_dims:
                    event_fields.append({
                        "key": "gen_ai.tool_calls",
                        "type": "string",
                        "value": custom_dims['tool_calls']
                    })
                
                log_entry = {
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": event_fields
                }
            else:
                # Regular log message
                log_entry = {
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {"key": "message", "type": "string", "value": message},
                        {"key": "level", "type": "string", "value": get_severity_level(trace.get('severityLevel', 0))}
                    ]
                }
            
            # Add additional custom dimension fields
            for key, value in custom_dims.items():
                if key not in ['gen_ai.prompt', 'gen_ai.event.content', 'tool_calls'] and value:
                    log_entry["fields"].append({
                        "key": key,
                        "type": get_value_type(value),
                        "value": str(value)
                    })
            
            span_map[parent_id]["logs"].append(log_entry)


def get_severity_level(level: int) -> str:
    """Convert numeric severity level to string."""
    levels = {0: "INFO", 1: "DEBUG", 2: "WARNING", 3: "ERROR", 4: "CRITICAL"}
    return levels.get(level, "INFO")


def parse_timestamp(timestamp_str: Any) -> datetime:
    """Parse timestamp from various formats."""
    if isinstance(timestamp_str, datetime):
        return timestamp_str
    
    if isinstance(timestamp_str, str):
        formats = [
            "%m/%d/%Y, %I:%M:%S.%f %p",  # Format from CSV
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                if timestamp_str.endswith(' PM') or timestamp_str.endswith(' AM'):
                    return datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
                else:
                    return datetime.strptime(timestamp_str.replace('Z', ''), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    
    logger.warning(f"Failed to parse timestamp: {timestamp_str}")
    return datetime.now(timezone.utc)


def get_value_type(value: Any) -> str:
    """Determine the type of a value for OpenTelemetry tags."""
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int64"
    elif isinstance(value, float):
        return "float64"
    else:
        return "string"


def generate_span_id(record: Dict[str, Any]) -> str:
    """Generate a deterministic span ID from record data."""
    unique_str = f"{record.get('timestamp', '')}-{record.get('name', '')}-{record.get('message', '')}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]


def evaluate_spans(otel_data: Dict[str, Any], model_config: AzureOpenAIModelConfiguration, trace_id: str) -> Dict[str, Any]:
    """Evaluate spans and return results."""
    spans = otel_data["data"][0]["spans"]
    
    # Initialize evaluators
    tool_accuracy_eval = ToolCallAccuracyEvaluator(model_config=model_config)
    intent_resolution_eval = IntentResolutionEvaluator(model_config=model_config)
    fluency_eval = FluencyEvaluator(model_config=model_config)
    groundedness_eval = GroundednessEvaluator(model_config=model_config)
    relevance_eval = RelevanceEvaluator(model_config=model_config)
    
    # Categorize spans
    llm_spans = []
    tool_spans = []
    final_plan_spans = []
    
    for span in spans:
        operation_name = span.get("operationName", "").lower()
        
        # Check tags for gen_ai.operation.name
        gen_ai_op = None
        for tag in span.get("tags", []):
            if tag["key"] == "gen_ai.operation.name":
                gen_ai_op = tag["value"]
                break
        
        # Identify LLM spans
        if "chat.completions" in operation_name or "gpt" in operation_name or gen_ai_op == "chat":
            llm_spans.append(span)
            
            # Check if this span has tool calls
            tool_calls = SpanDataExtractor.extract_tool_calls_from_span(span)
            if tool_calls:
                tool_spans.append(span)
        
        # Check for final plan content
        final_content = SpanDataExtractor.find_final_response_content(span)
        if final_content and "FINAL TRAVEL PLAN" in final_content.upper():
            final_plan_spans.append(span)
    
    logger.info(f"Identified spans - LLM: {len(llm_spans)}, Tool: {len(tool_spans)}, Final: {len(final_plan_spans)}")
    
    all_results = {
        "intent_resolution": [],
        "tool_accuracy": [],
        "response_quality": [],
        "plan_completeness": []
    }
    
    # 1. Intent Resolution Evaluation
    for span in llm_spans[:3]:  # Limit to avoid too many API calls
        try:
            messages = SpanDataExtractor.extract_messages_from_span(span)
            
            # Build conversation
            conversation = []
            if messages["system"]:
                conversation.append({"role": "system", "content": messages["system"][0]["content"]})
            if messages["user"]:
                conversation.append({"role": "user", "content": messages["user"][0]["content"]})
            if messages["assistant"]:
                conversation.append({"role": "assistant", "content": messages["assistant"][-1]["content"]})
            
            if len(conversation) >= 2:
                try:
                    result = intent_resolution_eval(conversation=conversation)
                    if isinstance(result, dict):  # Add this check
                        score = float(result.get("intent_resolution_score", 0.0)) * 5
                        reasoning = result.get("intent_resolution_reason", "")
                        emit_evaluation_event(
                            create_evaluation_event(
                                name="IntentResolution",
                                score=score,
                                reasoning=reasoning,
                                span_id=span["spanID"],
                                trace_id=trace_id
                            )
                        )
                        
                        all_results["intent_resolution"].append({
                            "span_id": span["spanID"],
                            "result": {"metric": "intent_resolution", "score": score, "reasoning": reasoning}
                        })
                except Exception as e:
                    logger.error(f"Intent evaluation error: {e}")
        except Exception as e:
            logger.error(f"Intent evaluation error: {e}")
    
    # 2. Tool Call Accuracy Evaluation
    for span in tool_spans[:3]:
        try:
            tool_calls = SpanDataExtractor.extract_tool_calls_from_span(span)
            messages = SpanDataExtractor.extract_messages_from_span(span)
            
            if tool_calls and messages["user"]:
                # Simple evaluation based on tool relevance
                travel_tools = ['get_match_schedule', 'search_flights', 'search_hotels', 'search_rental_cars']
                tools_used = [tc.get("name", "") for tc in tool_calls]
                relevant_tools = sum(1 for tool in tools_used if any(tt in tool.lower() for tt in travel_tools))
                
                score = (relevant_tools / len(tools_used)) * 5 if tools_used else 0
                reasoning = f"Used {relevant_tools} travel-related tools out of {len(tools_used)} total"
                
                emit_evaluation_event(
                    create_evaluation_event(
                        name="ToolRelevance",
                        score=score,
                        reasoning=reasoning,
                        span_id=span["spanID"],
                        trace_id=trace_id,
                        metadata={"tools_used": ", ".join(tools_used)}
                    )
                )
                
                all_results["tool_accuracy"].append({
                    "span_id": span["spanID"],
                    "result": {"metric": "tool_relevance", "score": score, "reasoning": reasoning}
                })
        except Exception as e:
            logger.error(f"Tool evaluation error: {e}")
    
    # 3. Response Quality Evaluation
    for span in llm_spans[:2] + final_plan_spans[:1]:
        try:
            messages = SpanDataExtractor.extract_messages_from_span(span)
            
            if messages["assistant"] and messages["user"]:
                response = messages["assistant"][-1]["content"]
                query = messages["user"][0]["content"]
                
                # Fluency evaluation
                fluency_result = fluency_eval(response=response)
                score = float(fluency_result.get("fluency", 0))
                reasoning = fluency_result.get("fluency_reason", "")
                
                emit_evaluation_event(
                    create_evaluation_event(
                        name="Fluency",
                        score=score,
                        reasoning=reasoning,
                        span_id=span["spanID"],
                        trace_id=trace_id
                    )
                )
                
                results = [{"metric": "fluency", "score": score, "reasoning": reasoning}]
                
                # Relevance evaluation
                relevance_result = relevance_eval(query=query, response=response)
                score = float(relevance_result.get("relevance", 0))
                reasoning = relevance_result.get("relevance_reason", "")
                
                emit_evaluation_event(
                    create_evaluation_event(
                        name="Relevance",
                        score=score,
                        reasoning=reasoning,
                        span_id=span["spanID"],
                        trace_id=trace_id
                    )
                )
                
                results.append({"metric": "relevance", "score": score, "reasoning": reasoning})
                
                all_results["response_quality"].append({
                    "span_id": span["spanID"],
                    "results": results
                })
        except Exception as e:
            logger.error(f"Response quality evaluation error: {e}")
    
    # 4. Final Plan Completeness
    for span in final_plan_spans:
        try:
            final_response = SpanDataExtractor.find_final_response_content(span)
            
            if final_response:
                # Check for key elements
                plan_elements = {
                    'flights': any(word in final_response.lower() for word in ['flight', 'airline', 'departure']),
                    'accommodation': any(word in final_response.lower() for word in ['hotel', 'accommodation']),
                    'dates': any(word in final_response.lower() for word in ['august', 'september', '2024']),
                    'pricing': '$' in final_response,
                    'matches': any(word in final_response.lower() for word in ['chelsea', 'match', 'game']),
                }
                
                elements_present = sum(plan_elements.values())
                score = (elements_present / len(plan_elements)) * 5
                reasoning = f"Plan contains {elements_present}/{len(plan_elements)} key elements"
                
                emit_evaluation_event(
                    create_evaluation_event(
                        name="PlanCompleteness",
                        score=score,
                        reasoning=reasoning,
                        span_id=span["spanID"],
                        trace_id=trace_id
                    )
                )
                
                all_results["plan_completeness"].append({
                    "span_id": span["spanID"],
                    "result": {"metric": "plan_completeness", "score": score, "reasoning": reasoning}
                })
        except Exception as e:
            logger.error(f"Plan completeness evaluation error: {e}")
    
    return all_results


def create_evaluation_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of all evaluations."""
    total_score = 0
    total_count = 0
    evaluation_counts = defaultdict(int)
    
    for category, items in evaluation_results.items():
        for item in items:
            if "result" in item:
                result = item["result"]
                total_score += result["score"]
                total_count += 1
                evaluation_counts[category] += 1
            elif "results" in item:
                for result in item["results"]:
                    total_score += result["score"]
                    total_count += 1
                    evaluation_counts[result["metric"]] += 1
    
    avg_score = total_score / total_count if total_count > 0 else 0
    
    return {
        "total_evaluations": total_count,
        "average_score": avg_score,
        "percentage": (avg_score / 5) * 100,
        "evaluation_counts": dict(evaluation_counts),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


def extract_and_evaluate_trace(
    trace_id: str,
    output_file: str = "trace_with_evaluations_otel.json",
    time_range_hours: int = 24,
    evaluate: bool = True
):
    """
    Extract traces, optionally evaluate them, and save everything to JSON following OTEL conventions.
    
    Args:
        trace_id: The operation/trace ID to search for
        output_file: Output JSON file path
        time_range_hours: How many hours back to search
        evaluate: Whether to perform evaluations
    """
    # Initialize Application Insights connector
    app_config = ApplicationInsightsConfig(
        resource_id=os.environ["APPLICATION_INSIGHTS_RESOURCE_ID"]
    )
    connector = ApplicationInsightsConnector(app_config)
    
    # Query for all records with this trace ID
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=time_range_hours)
    
    query = f"""
    union *
    | where timestamp between (datetime('{start_time.isoformat()}') .. datetime('{end_time.isoformat()}'))
    | where operation_Id == "{trace_id}"
    | project timestamp, itemType, message, severityLevel, customDimensions, 
              id, operation_ParentId, name, url, success, resultCode, duration, 
              target, type, data, operation_Id, operation_Name
    | order by timestamp asc
    """
    
    logger.info(f"Extracting traces for ID: {trace_id}")
    logger.info(f"Time range: {start_time} to {end_time}")
    
    try:
        # Execute query
        trace_records = connector._execute_query(query, timespan=timedelta(hours=time_range_hours))
        logger.info(f"Found {len(trace_records)} records")
        
        if not trace_records:
            logger.warning("No records found for the given trace ID")
            return None
        
        # Convert to OpenTelemetry compliant format
        otel_data = convert_to_otel_compliant_format(trace_records, trace_id)
        otel_data["total"] = len(otel_data["data"][0]["spans"])
        
        # Add evaluation results if requested
        if evaluate:
            logger.info("Starting evaluation...")
            
            # Azure OpenAI model config
            model_config = AzureOpenAIModelConfiguration(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            )
            
            evaluation_results = evaluate_spans(otel_data, model_config, trace_id)
            
            # Add evaluation results to the JSON structure
            otel_data["evaluations"] = evaluation_results
            
            # Also add evaluation summary
            otel_data["evaluation_summary"] = create_evaluation_summary(evaluation_results)
        
        # Add metadata following OTEL conventions
        otel_data["metadata"] = {
            "extraction_time": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "total_records": len(trace_records),
            "total_spans": len(otel_data["data"][0]["spans"]),
            "evaluated": evaluate,
            "otel_semantic_version": "1.27.0",
            "gen_ai_semantic_version": "1.0.0"
        }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(otel_data, f, indent=2, default=str)
        
        logger.info(f"Successfully saved OTEL-compliant trace data to {output_file}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"EXTRACTION AND EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"Trace ID: {trace_id}")
        print(f"Total records: {len(trace_records)}")
        print(f"Total spans: {len(otel_data['data'][0]['spans'])}")
        print(f"OTEL Semantic Convention: v1.27.0")
        print(f"GenAI Semantic Convention: v1.0.0")
        
        if evaluate and "evaluation_summary" in otel_data:
            summary = otel_data["evaluation_summary"]
            print(f"\nEvaluation Results:")
            print(f"- Total evaluations: {summary['total_evaluations']}")
            print(f"- Average score: {summary['average_score']:.1f}/5 ({summary['percentage']:.0f}%)")
            print(f"- Evaluation breakdown:")
            for metric, count in summary['evaluation_counts'].items():
                print(f"  - {metric}: {count} evaluations")
        
        print(f"\nOutput file: {output_file}")
        
        return otel_data
        
    except Exception as e:
        logger.error(f"Error in extraction/evaluation: {e}")
        raise


def main():
    """Main function to extract and evaluate traces with OTEL compliance."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and evaluate traces from Application Insights')
    parser.add_argument('--trace-id', type=str, default="a33d66627a40c88e7ec94860fa967893",
                        help='Trace ID to extract')
    parser.add_argument('--output', type=str, default="trace_with_evaluations_otel.json",
                        help='Output file path')
    parser.add_argument('--hours', type=int, default=48,
                        help='Number of hours to look back')
    parser.add_argument('--no-evaluate', action='store_true',
                        help='Skip evaluation and only extract traces')
    
    args = parser.parse_args()
    
    # Extract, evaluate, and save with OTEL semantic conventions
    extract_and_evaluate_trace(
        trace_id=args.trace_id,
        output_file=args.output,
        time_range_hours=args.hours,
        evaluate=not args.no_evaluate
    )


if __name__ == "__main__":
    main()