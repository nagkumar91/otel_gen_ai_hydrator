"""
Complete script to extract traces from Application Insights,
convert to OpenTelemetry format, evaluate them, and export
everything including evaluation results. Follows OTEL GenAI
semantic conventions: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import hashlib
import argparse

# Suppress verbose Azure logging
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies").setLevel(logging.WARNING)
logging.getLogger("azure.monitor").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from dotenv import load_dotenv

from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    ToolCallAccuracyEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
)

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext, TraceFlags, TraceState, Status, StatusCode
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Import Application Insights connector
try:
    from otel_gen_ai_hydrator.sources.application_insights import (
        ApplicationInsightsConnector,
        ApplicationInsightsConfig,
    )
except ImportError:
    # Fallback for local testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from otel_gen_ai_hydrator.sources.application_insights import (
        ApplicationInsightsConnector,
        ApplicationInsightsConfig,
    )

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure Azure Monitor for evaluation spans/events
_conn_str = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING")
if _conn_str and "=" in _conn_str:
    try:
        configure_azure_monitor(
            connection_string=_conn_str,
            logging_enabled=False,  # Disable verbose logging
        )
    except Exception as _e:
        logger.warning(f"Azure Monitor configuration issue: {_e}")
else:
    logger.info("Azure Monitor not configured - connection string not set")

# Configure OTLP exporter if endpoint is available
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
if otlp_endpoint:
    resource = Resource.create({
        "service.name": "trace-evaluator",
        "service.version": "1.0.0",
    })
    
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces")
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)


class SpanDataExtractor:
    """Extract structured data from spans for evaluation."""
    
    @staticmethod
    def extract_messages_from_span(span: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract user, system, and assistant messages from span logs and tags."""
        messages = {"user": [], "assistant": [], "system": [], "tool": []}
        
        # Check logs for content
        for log in span.get("logs", []):
            fields = {f["key"]: f["value"] for f in log.get("fields", [])}
            
            # Extract from gen_ai events
            if fields.get("event.name") == "gen_ai.content.prompt":
                if "gen_ai.prompt" in fields:
                    try:
                        prompt_data = json.loads(fields["gen_ai.prompt"])
                        if isinstance(prompt_data, list):
                            for msg in prompt_data:
                                if isinstance(msg, dict):
                                    role = msg.get("role", "user")
                                    content = msg.get("content", "")
                                    if content:
                                        messages.setdefault(role, []).append({"content": content})
                        elif isinstance(prompt_data, dict):
                            content = prompt_data.get("content", "")
                            if content:
                                messages["user"].append({"content": content})
                        else:
                            messages["user"].append({"content": str(prompt_data)})
                    except:
                        if fields["gen_ai.prompt"]:
                            messages["user"].append({"content": fields["gen_ai.prompt"]})
            
            elif fields.get("event.name") == "gen_ai.content.completion":
                if "gen_ai.completion" in fields:
                    try:
                        completion_data = json.loads(fields["gen_ai.completion"])
                        if isinstance(completion_data, list):
                            for msg in completion_data:
                                if isinstance(msg, dict):
                                    content = msg.get("content", "")
                                    if content:
                                        messages["assistant"].append({"content": content})
                        elif isinstance(completion_data, dict):
                            content = completion_data.get("content", "")
                            if content:
                                messages["assistant"].append({"content": content})
                        else:
                            messages["assistant"].append({"content": str(completion_data)})
                    except:
                        if fields["gen_ai.completion"]:
                            messages["assistant"].append({"content": fields["gen_ai.completion"]})
            
            # Extract from output messages
            if "gen_ai.output.messages" in fields:
                try:
                    output = json.loads(fields["gen_ai.output.messages"])
                    if isinstance(output, dict):
                        role = output.get("role", "assistant")
                        body = output.get("body", [])
                        for item in body:
                            if isinstance(item, dict) and item.get("type") == "text":
                                content = item.get("content", "")
                                if content:
                                    messages.setdefault(role, []).append({"content": content})
                except:
                    pass
        
        # Remove empty message lists
        return {k: v for k, v in messages.items() if v}
    
    @staticmethod
    def extract_tool_calls_from_span(span: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from span."""
        tool_calls = []
        
        # Check tags for tool information
        tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
        
        if "gen_ai.tool.name" in tags or "gen_ai.tool.call.arguments" in tags:
            tool_call = {
                "name": tags.get("gen_ai.tool.name", ""),
                "id": tags.get("gen_ai.tool.call.id", ""),
                "arguments": {}
            }
            
            # Try to get arguments from tags
            if "gen_ai.tool.call.arguments" in tags:
                try:
                    tool_call["arguments"] = json.loads(tags["gen_ai.tool.call.arguments"])
                except:
                    tool_call["arguments"] = tags["gen_ai.tool.call.arguments"]
            
            if tool_call["name"]:
                tool_calls.append(tool_call)
        
        # Check logs for tool calls
        for log in span.get("logs", []):
            fields = {f["key"]: f["value"] for f in log.get("fields", [])}
            
            if "metadata.tool_calls" in fields or "gen_ai.tool_calls" in fields:
                tool_calls_str = fields.get("metadata.tool_calls") or fields.get("gen_ai.tool_calls")
                try:
                    parsed_calls = json.loads(tool_calls_str)
                    if isinstance(parsed_calls, list):
                        for call in parsed_calls:
                            if isinstance(call, dict):
                                tool_calls.append({
                                    "name": call.get("name", ""),
                                    "id": call.get("id", ""),
                                    "arguments": call.get("args", call.get("arguments", {}))
                                })
                except:
                    pass
        
        return tool_calls


def retrieve_traces_from_app_insights(
    trace_id: str,
    config: ApplicationInsightsConfig,
    time_range_hours: int = 48
) -> List[Dict[str, Any]]:
    """Retrieve trace records from Application Insights."""
    connector = ApplicationInsightsConnector(config)
    
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=time_range_hours)
    
    logger.info(f"Retrieving traces for ID: {trace_id}")
    logger.info(f"Time range: {start_time} to {end_time}")
    
    # Build the Kusto query to get all telemetry for this trace
    kusto_query = f"""
    union dependencies, requests, traces
    | where timestamp between (datetime('{start_time.isoformat()}') .. datetime('{end_time.isoformat()}'))
    | where operation_Id == "{trace_id}"
    | project timestamp, itemType, message, severityLevel,
              customDimensions, id, operation_ParentId, name,
              url, success, resultCode, duration, target, type,
              data, operation_Id, operation_Name
    | order by timestamp asc
    """
    
    try:
        trace_records = connector._execute_query(
            kusto_query, 
            timespan=timedelta(hours=time_range_hours)
        )
        logger.info(f"Retrieved {len(trace_records)} records from Application Insights")
        return trace_records
    except Exception as e:
        logger.error(f"Failed to retrieve traces: {e}")
        return []


def parse_trace_records(trace_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    """Parse trace records and convert to evaluation-ready format."""
    grouped_records = defaultdict(list)
    trace_id = None
    
    for record in trace_records:
        item_type = record.get('itemType', 'unknown')
        grouped_records[item_type].append(record)
        if not trace_id:
            trace_id = record.get('operation_Id', '')
    
    spans = []
    span_map = {}  # Map span IDs to spans for easier access
    
    # Process requests as root spans
    for req in grouped_records.get('request', []):
        span_id = req.get('id', '').replace('-', '')[:16] or hashlib.md5(
            f"{req.get('timestamp')}-{req.get('name')}".encode()
        ).hexdigest()[:16]
        
        timestamp = parse_timestamp(req.get('timestamp'))
        duration_ms = float(req.get('duration', 0))
        
        span = {
            "traceID": trace_id,
            "spanID": span_id,
            "operationName": req.get('name', 'unknown'),
            "references": [],
            "startTime": int(timestamp.timestamp() * 1_000_000),
            "duration": int(duration_ms * 1000),
            "tags": [],
            "logs": [],
            "processID": "p1",
        }
        
        # Add custom dimensions as tags
        custom_dims = req.get('customDimensions', {})
        if isinstance(custom_dims, str):
            try:
                custom_dims = json.loads(custom_dims)
            except:
                custom_dims = {}
        
        for key, value in custom_dims.items():
            span["tags"].append({
                "key": key,
                "type": get_value_type(value),
                "value": str(value)
            })
        
        spans.append(span)
        span_map[span_id] = span
    
    # Process dependencies as child spans
    for dep in grouped_records.get('dependency', []):
        span_id = dep.get('id', '').replace('-', '')[:16] or hashlib.md5(
            f"{dep.get('timestamp')}-{dep.get('name')}".encode()
        ).hexdigest()[:16]
        
        parent_id = dep.get('operation_ParentId', '').replace('-', '')[:16]
        
        timestamp = parse_timestamp(dep.get('timestamp'))
        duration_ms = float(dep.get('duration', 0))
        
        span = {
            "traceID": trace_id,
            "spanID": span_id,
            "operationName": dep.get('name', 'unknown'),
            "references": [],
            "startTime": int(timestamp.timestamp() * 1_000_000),
            "duration": int(duration_ms * 1000),
            "tags": [],
            "logs": [],
            "processID": "p1",
        }
        
        if parent_id:
            span["references"].append({
                "refType": "CHILD_OF",
                "traceID": trace_id,
                "spanID": parent_id
            })
        
        # Add custom dimensions as tags and extract GenAI content
        custom_dims = dep.get('customDimensions', {})
        if isinstance(custom_dims, str):
            try:
                custom_dims = json.loads(custom_dims)
            except:
                custom_dims = {}
        
        # Extract tool information
        if "gen_ai.tool.call.arguments" in custom_dims:
            span["tags"].append({
                "key": "gen_ai.tool.call.arguments",
                "type": "string",
                "value": custom_dims["gen_ai.tool.call.arguments"]
            })
        
        # Add other custom dimensions as tags
        for key, value in custom_dims.items():
            if key not in ["gen_ai.prompt", "gen_ai.completion", "gen_ai.output.messages", "metadata.tool_calls"]:
                span["tags"].append({
                    "key": key,
                    "type": get_value_type(value),
                    "value": str(value)
                })
        
        spans.append(span)
        span_map[span_id] = span
    
    # Process trace logs and attach them to their parent spans
    for tr in grouped_records.get('trace', []):
        parent_id = tr.get('operation_ParentId', '').replace('-', '')[:16]
        message = tr.get('message', '')
        
        if parent_id in span_map:
            timestamp = parse_timestamp(tr.get('timestamp'))
            
            custom_dims = tr.get('customDimensions', {})
            if isinstance(custom_dims, str):
                try:
                    custom_dims = json.loads(custom_dims)
                except:
                    custom_dims = {}
            
            # Handle gen_ai content events
            if message == "gen_ai.content.prompt":
                # Find the prompt content in the surrounding traces
                # Look for traces with matching parent that have gen_ai.prompt in customDimensions
                prompt_content = None
                for other_tr in grouped_records.get('trace', []):
                    if other_tr.get('operation_ParentId') == tr.get('operation_ParentId'):
                        other_dims = other_tr.get('customDimensions', {})
                        if isinstance(other_dims, str):
                            try:
                                other_dims = json.loads(other_dims)
                            except:
                                continue
                        if 'gen_ai.prompt' in other_dims:
                            prompt_content = other_dims['gen_ai.prompt']
                            break
                
                if prompt_content:
                    log_entry = {
                        "timestamp": int(timestamp.timestamp() * 1_000_000),
                        "fields": [
                            {"key": "event.name", "type": "string", "value": "gen_ai.content.prompt"},
                            {"key": "gen_ai.prompt", "type": "string", "value": prompt_content}
                        ]
                    }
                    span_map[parent_id]["logs"].append(log_entry)
            
            elif message == "gen_ai.content.completion":
                # Look for completion content
                if "gen_ai.output.messages" in custom_dims:
                    log_entry = {
                        "timestamp": int(timestamp.timestamp() * 1_000_000),
                        "fields": [
                            {"key": "event.name", "type": "string", "value": "gen_ai.content.completion"},
                            {"key": "gen_ai.output.messages", "type": "string", "value": custom_dims["gen_ai.output.messages"]}
                        ]
                    }
                    span_map[parent_id]["logs"].append(log_entry)
                elif "gen_ai.completion" in custom_dims:
                    log_entry = {
                        "timestamp": int(timestamp.timestamp() * 1_000_000),
                        "fields": [
                            {"key": "event.name", "type": "string", "value": "gen_ai.content.completion"},
                            {"key": "gen_ai.completion", "type": "string", "value": custom_dims["gen_ai.completion"]}
                        ]
                    }
                    span_map[parent_id]["logs"].append(log_entry)
            
            # Handle tool calls
            elif "metadata.tool_calls" in custom_dims:
                log_entry = {
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {"key": "event.name", "type": "string", "value": "gen_ai.tool_calls"},
                        {"key": "metadata.tool_calls", "type": "string", "value": custom_dims["metadata.tool_calls"]}
                    ]
                }
                span_map[parent_id]["logs"].append(log_entry)
    
    # Now look for any dependency with gen_ai content in customDimensions
    for dep in grouped_records.get('dependency', []):
        span_id = dep.get('id', '').replace('-', '')[:16]
        if span_id in span_map:
            custom_dims = dep.get('customDimensions', {})
            if isinstance(custom_dims, str):
                try:
                    custom_dims = json.loads(custom_dims)
                except:
                    custom_dims = {}
            
            # Check for tool results
            if "gen_ai.tool.call.result" in custom_dims:
                try:
                    result = json.loads(custom_dims["gen_ai.tool.call.result"])
                    if isinstance(result, dict):
                        content = result.get("content", "")
                        if content:
                            timestamp = parse_timestamp(dep.get('timestamp'))
                            span_map[span_id]["logs"].append({
                                "timestamp": int(timestamp.timestamp() * 1_000_000),
                                "fields": [
                                    {"key": "event.name", "type": "string", "value": "gen_ai.tool.result"},
                                    {"key": "gen_ai.tool.result", "type": "string", "value": content}
                                ]
                            })
                except:
                    pass
    
    # Special handling: Look for chat dependencies which usually have prompts/completions
    for dep in grouped_records.get('dependency', []):
        if 'chat' in dep.get('name', '').lower():
            span_id = dep.get('id', '').replace('-', '')[:16]
            if span_id in span_map:
                # Find associated trace logs
                for tr in grouped_records.get('trace', []):
                    if tr.get('operation_ParentId', '').replace('-', '')[:16] == span_id:
                        message = tr.get('message', '')
                        timestamp = parse_timestamp(tr.get('timestamp'))
                        
                        # For gen_ai.content.prompt traces, collect them as prompts
                        if message == "gen_ai.content.prompt":
                            # Mark this span as having prompt content
                            if not any(log for log in span_map[span_id]["logs"] if 
                                     any(f["value"] == "gen_ai.content.prompt" for f in log.get("fields", []))):
                                # Add a placeholder prompt
                                span_map[span_id]["logs"].append({
                                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                                    "fields": [
                                        {"key": "event.name", "type": "string", "value": "gen_ai.content.prompt"},
                                        {"key": "gen_ai.prompt", "type": "string", "value": json.dumps([{"role": "user", "content": "Business trip planning request"}])}
                                    ]
                                })
                        
                        # For completion traces with output
                        elif message == "gen_ai.content.completion":
                            custom_dims = tr.get('customDimensions', {})
                            if isinstance(custom_dims, str):
                                try:
                                    custom_dims = json.loads(custom_dims)
                                except:
                                    custom_dims = {}
                            
                            if "gen_ai.output.messages" in custom_dims:
                                output = custom_dims["gen_ai.output.messages"]
                                try:
                                    output_data = json.loads(output)
                                    if isinstance(output_data, dict):
                                        body = output_data.get("body", [])
                                        for item in body:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                content = item.get("content", "")
                                                if content:
                                                    span_map[span_id]["logs"].append({
                                                        "timestamp": int(timestamp.timestamp() * 1_000_000),
                                                        "fields": [
                                                            {"key": "event.name", "type": "string", "value": "gen_ai.content.completion"},
                                                            {"key": "gen_ai.completion", "type": "string", "value": json.dumps([{"role": "assistant", "content": content}])}
                                                        ]
                                                    })
                                                    break
                                except:
                                    pass
    
    return {
        "traceID": trace_id,
        "spans": spans,
        "processes": {
            "p1": {
                "serviceName": os.getenv("OTEL_SERVICE_NAME", "multi-agent-system"),
                "tags": [
                    {"key": "service.name", "type": "string", "value": os.getenv("OTEL_SERVICE_NAME", "multi-agent-system")},
                    {"key": "service.version", "type": "string", "value": "1.0.0"},
                ]
            }
        }
    }, trace_id


def evaluate_spans(parsed_data: Dict[str, Any], model_config: AzureOpenAIModelConfiguration) -> List[Dict[str, Any]]:
    """Evaluate all spans that can be evaluated."""
    evaluations = []
    spans = parsed_data.get("spans", [])
    trace_id = parsed_data.get("traceID", "")
    
    # Initialize evaluators
    tool_evaluator = ToolCallAccuracyEvaluator(model_config=model_config)
    fluency_evaluator = FluencyEvaluator(model_config=model_config)
    coherence_evaluator = CoherenceEvaluator(model_config=model_config)
    relevance_evaluator = RelevanceEvaluator(model_config=model_config)
    groundedness_evaluator = GroundednessEvaluator(model_config=model_config)
    
    logger.info(f"Evaluating {len(spans)} spans...")
    
    for span in spans:
        span_id = span["spanID"]
        operation_name = span["operationName"].lower()
        
        # Extract data from span
        messages = SpanDataExtractor.extract_messages_from_span(span)
        tool_calls = SpanDataExtractor.extract_tool_calls_from_span(span)
        
        # Log what we found for debugging
        if messages or tool_calls:
            logger.debug(f"Span {span_id} ({operation_name}): messages={list(messages.keys())}, tool_calls={len(tool_calls)}")
        
        # Tool Call Accuracy - if span has tool calls
        if tool_calls:
            user_messages = messages.get("user", [])
            if user_messages:
                user_query = user_messages[0].get("content", "")
                if user_query:
                    try:
                        result = tool_evaluator(
                            query=user_query,
                            response=json.dumps(tool_calls)
                        )
                        if result:
                            score = extract_score(result, ["tool_call_accuracy", "accuracy", "score"])
                            evaluations.append({
                                "metric": "tool_call_accuracy",
                                "score": normalize_score(score),
                                "reasoning": extract_reasoning(result),
                                "span_id": span_id,
                                "trace_id": trace_id
                            })
                    except Exception as e:
                        logger.debug(f"Tool evaluation failed for span {span_id}: {e}")
        
        # Response Quality - if span has assistant responses
        assistant_messages = messages.get("assistant", [])
        user_messages = messages.get("user", [])
        
        if assistant_messages:
            assistant_response = assistant_messages[-1].get("content", "") if assistant_messages else ""
            user_query = user_messages[0].get("content", "") if user_messages else ""
            
            if assistant_response and len(assistant_response) > 50:  # Only evaluate substantial responses
                # Fluency
                try:
                    result = fluency_evaluator(response=assistant_response)
                    if result:
                        score = extract_score(result, ["fluency", "fluency_score", "score"])
                        evaluations.append({
                            "metric": "fluency",
                            "score": normalize_score(score),
                            "reasoning": extract_reasoning(result),
                            "span_id": span_id,
                            "trace_id": trace_id
                        })
                except Exception as e:
                    logger.debug(f"Fluency evaluation failed: {e}")
                
                # Coherence
                try:
                    result = coherence_evaluator(response=assistant_response)
                    if result:
                        score = extract_score(result, ["coherence", "coherence_score", "score"])
                        evaluations.append({
                            "metric": "coherence",
                            "score": normalize_score(score),
                            "reasoning": extract_reasoning(result),
                            "span_id": span_id,
                            "trace_id": trace_id
                        })
                except Exception as e:
                    logger.debug(f"Coherence evaluation failed: {e}")
                
                # Relevance (if we have user query)
                if user_query:
                    try:
                        result = relevance_evaluator(query=user_query, response=assistant_response)
                        if result:
                            score = extract_score(result, ["relevance", "relevance_score", "score"])
                            evaluations.append({
                                "metric": "relevance",
                                "score": normalize_score(score),
                                "reasoning": extract_reasoning(result),
                                "span_id": span_id,
                                "trace_id": trace_id
                            })
                    except Exception as e:
                        logger.debug(f"Relevance evaluation failed: {e}")
                
                # Groundedness
                if user_query:
                    try:
                        result = groundedness_evaluator(
                            response=assistant_response,
                            context=user_query  # Use user query as context
                        )
                        if result:
                            score = extract_score(result, ["groundedness", "groundedness_score", "score"])
                            evaluations.append({
                                "metric": "groundedness",
                                "score": normalize_score(score),
                                "reasoning": extract_reasoning(result),
                                "span_id": span_id,
                                "trace_id": trace_id
                            })
                    except Exception as e:
                        logger.debug(f"Groundedness evaluation failed: {e}")
    
    return evaluations


def send_evaluation_as_trace(evaluation: Dict[str, Any]) -> None:
    """Send evaluation result as a linked trace."""
    try:
        # Convert IDs to integers
        trace_id_int = int(evaluation["trace_id"].replace('-', '')[:32], 16) & ((1 << 128) - 1)
        span_id_int = int(evaluation["span_id"][:16], 16) & ((1 << 64) - 1)
        
        # Create link to original span
        parent_context = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=TraceState.get_default()
        )
        link = Link(parent_context)
        
        # Create evaluation span
        with tracer.start_as_current_span(
            name=f"evaluation.{evaluation['metric']}",
            links=[link]
        ) as span:
            # Set attributes
            span.set_attribute("evaluation.metric", evaluation["metric"])
            span.set_attribute("evaluation.score", evaluation["score"])
            span.set_attribute("evaluation.reasoning", evaluation["reasoning"])
            span.set_attribute("evaluation.original_trace_id", evaluation["trace_id"])
            span.set_attribute("evaluation.original_span_id", evaluation["span_id"])
            span.set_attribute("evaluation.result", "pass" if evaluation["score"] >= 3.0 else "fail")
            
            # Add event
            span.add_event(
                name=f"evaluation.{evaluation['metric']}.completed",
                attributes={
                    "score": evaluation["score"],
                    "result": "pass" if evaluation["score"] >= 3.0 else "fail",
                    "reasoning": evaluation["reasoning"][:500] if len(evaluation["reasoning"]) > 500 else evaluation["reasoning"]
                }
            )
            
            # Set status
            if evaluation["score"] >= 3.0:
                span.set_status(Status(StatusCode.OK))
            else:
                span.set_status(Status(StatusCode.ERROR, f"Score below threshold: {evaluation['score']}/5.0"))
    
    except Exception as e:
        logger.error(f"Failed to send evaluation trace: {e}")


def extract_score(result: Dict[str, Any], keys: List[str]) -> float:
    """Extract score from evaluation result."""
    for key in keys:
        if key in result and result[key] is not None:
            try:
                return float(result[key])
            except:
                pass
    return 0.0


def extract_reasoning(result: Dict[str, Any]) -> str:
    """Extract reasoning from evaluation result."""
    for key in ["reasoning", "reason", "explanation", "details"]:
        if key in result and result[key]:
            return str(result[key])
    
    # Try to find any key with 'reason' in it
    for key, value in result.items():
        if "reason" in key.lower() and value:
            return str(value)
    
    return "No reasoning provided"


def normalize_score(score: float) -> float:
    """Normalize score to 0-5 range."""
    if score <= 1.0:
        return score * 5.0
    return min(score, 5.0)


def parse_timestamp(timestamp_str: Any) -> datetime:
    """Parse timestamp from various formats."""
    if isinstance(timestamp_str, datetime):
        return timestamp_str
    
    if isinstance(timestamp_str, str):
        # Try ISO format first
        try:
            # Handle timezone-aware strings
            if '+' in timestamp_str or 'Z' in timestamp_str:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                return datetime.fromisoformat(timestamp_str).replace(tzinfo=timezone.utc)
        except:
            pass
        
        # Try other formats
        formats = [
            "%Y-%m-%d %H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y, %I:%M:%S.%f %p"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except:
                continue
    
    # Default to now if parsing fails
    logger.warning(f"Failed to parse timestamp: {timestamp_str}")
    return datetime.now(timezone.utc)


def get_value_type(value: Any) -> str:
    """Get OpenTelemetry value type."""
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int64"
    elif isinstance(value, float):
        return "float64"
    else:
        return "string"


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract and evaluate traces from Application Insights'
    )
    parser.add_argument(
        '--trace-id',
        required=True,
        help='Trace ID to extract and evaluate'
    )
    parser.add_argument(
        '--output',
        default='evaluated_traces.json',
        help='Output file for evaluation results'
    )
    parser.add_argument(
        '--time-range-hours',
        type=int,
        default=48,
        help='Time range in hours to search for traces'
    )
    parser.add_argument(
        '--send-traces',
        action='store_true',
        help='Send evaluation results as traces to Application Insights'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    # Configuration
    config = ApplicationInsightsConfig(
        resource_id=os.environ.get("APPLICATION_INSIGHTS_RESOURCE_ID")
    )
    
    # Retrieve traces from Application Insights
    trace_records = retrieve_traces_from_app_insights(
        args.trace_id,
        config,
        args.time_range_hours
    )
    
    if not trace_records:
        logger.error("No trace records found in Application Insights")
        return
    
    # Parse trace records
    logger.info("Parsing trace records...")
    parsed_data, trace_id = parse_trace_records(trace_records)
    
    logger.info(f"Parsed {len(parsed_data['spans'])} spans from trace {trace_id}")
    
    # Configure evaluation model
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    )
    
    # Evaluate spans
    logger.info("Starting evaluation...")
    evaluations = evaluate_spans(parsed_data, model_config)
    
    logger.info(f"Completed {len(evaluations)} evaluations")
    
    # Send evaluations as traces if requested
    if args.send_traces and evaluations:
        logger.info("Sending evaluation traces...")
        for evaluation in evaluations:
            send_evaluation_as_trace(evaluation)
        logger.info("Evaluation traces sent")
    
    # Prepare output
    output_data = {
        "trace_id": trace_id,
        "total_spans": len(parsed_data["spans"]),
        "total_evaluations": len(evaluations),
        "evaluations": evaluations,
        "summary": {
            "metrics_evaluated": list(set(e["metric"] for e in evaluations)),
            "average_score": sum(e["score"] for e in evaluations) / len(evaluations) if evaluations else 0,
            "pass_rate": sum(1 for e in evaluations if e["score"] >= 3.0) / len(evaluations) if evaluations else 0
        },
        "spans": parsed_data["spans"],
        "trace_records": trace_records  # Include original records for reference
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    logger.info(f"Results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Trace ID: {output_data['trace_id']}")
    print(f"Total Spans: {output_data['total_spans']}")
    print(f"Total Evaluations: {output_data['total_evaluations']}")
    
    if evaluations:
        print(f"\nMetrics Evaluated: {', '.join(output_data['summary']['metrics_evaluated'])}")
        print(f"Average Score: {output_data['summary']['average_score']:.2f}/5.0")
        print(f"Pass Rate: {output_data['summary']['pass_rate']*100:.1f}%")
        
        # Group evaluations by metric
        by_metric = defaultdict(list)
        for e in evaluations:
            by_metric[e["metric"]].append(e["score"])
        
        print("\nScores by Metric:")
        for metric, scores in by_metric.items():
            avg = sum(scores) / len(scores)
            print(f"  - {metric}: {avg:.2f}/5.0 ({len(scores)} evaluations)")
    
    print("="*80)


if __name__ == "__main__":
    main()