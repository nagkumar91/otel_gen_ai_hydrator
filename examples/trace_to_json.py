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
from typing import Dict, List, Any, Optional
from collections import defaultdict
import hashlib
import argparse

from dotenv import load_dotenv
try:
    # Module-relative import
    from .span_extraction import (
        extract_messages_from_span as sx_extract_messages,
        extract_tool_calls_from_span as sx_extract_tools,
    )
except Exception:  # pragma: no cover - script execution fallback
    from span_extraction import (
        extract_messages_from_span as sx_extract_messages,
        extract_tool_calls_from_span as sx_extract_tools,
    )
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
from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext, TraceFlags, TraceState
from opentelemetry._events import Event, get_event_logger

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure Azure Monitor for evaluation events only if a plausible
# connection string is present. This avoids crashes in offline/dry-run mode.
_conn_str = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING")
if _conn_str and "=" in _conn_str:
    try:
        configure_azure_monitor(connection_string=_conn_str)
    except Exception as _e:  # pragma: no cover - best-effort config
        logger.warning(
            "Azure Monitor not configured: invalid connection string: %s",
            _e,
        )
else:
    logger.info(
        "Skipping Azure Monitor config: AI connection string not set"
    )

# Event logger for emitting evaluation events
event_logger = get_event_logger(__name__)


class SpanDataExtractor:
    """Helper class to extract data from spans for evaluation."""

    @staticmethod
    def extract_messages_from_span(
        span: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        # Delegate to robust extractor
        return sx_extract_messages(span)

    @staticmethod
    def extract_tool_calls_from_span(
        span: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        # Delegate to robust extractor
        return sx_extract_tools(span)

    @staticmethod
    def find_final_response_content(span: Dict[str, Any]) -> Optional[str]:
        messages = SpanDataExtractor.extract_messages_from_span(span)
        if messages.get("assistant"):
            last_message = messages["assistant"][-1].get("content", "")
            if len(last_message) > 500:
                return last_message
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
    """
    Create an OpenTelemetry event for AI evaluation results following the
    semantic convention.
    """
    try:
        span_id_int = int(span_id, 16) if isinstance(span_id, str) else span_id
        trace_id_int = (
            int(trace_id, 16) if isinstance(trace_id, str) else trace_id
        )
    except Exception:
        span_id_int = 0
        trace_id_int = 0
    
    # Create the evaluation result body as per the semantic convention
    evaluation_result = {
        "score": score,
        "reasoning": reasoning,
        "result": "pass" if score >= 3.0 else "fail",
        "span_id": span_id,
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if metadata:
        evaluation_result.update(metadata)
    
    # Attributes following the semantic convention
    # https://github.com/singankit/semantic-conventions/blob/f334fee3e56057355ebf326a06d08973b7a9c554/docs/registry/attributes/gen-ai.md
    attributes = {
        "gen_ai.evaluation.score": score,
        "gen_ai.evaluation.name": name.lower(),
        "gen_ai.evaluation.result": "pass" if score >= 3.0 else "fail"
    }
        
    return Event(
        name=f"gen_ai.evaluation.{name.lower()}",
        attributes=attributes,
        body=json.dumps(evaluation_result),
        span_id=span_id_int,
        trace_id=trace_id_int,
    )


def emit_evaluation_event(event: Event):
    """Emit an OTel event into App Insights."""
    event_logger.emit(event)


def emit_evaluation_span(
    *,
    name: str,
    score: float,
    reasoning: str,
    span_id: str,
    trace_id: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Create and export a span for the evaluation result and link it to
    the evaluated span. Attributes follow GenAI evaluation attributes.
    """
    try:
        # Convert hex IDs to integers as required by SpanContext
        trace_id_int = (
            int(trace_id, 16) if isinstance(trace_id, str) else int(trace_id)
        )
        span_id_int = (
            int(span_id, 16) if isinstance(span_id, str) else int(span_id)
        )
        parent_ctx = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=TraceState.get_default(),
        )
        link = Link(parent_ctx)

        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(
            name=f"gen_ai.evaluation.{name.lower()}", links=[link]
        ) as eval_span:
            eval_span.set_attribute("gen_ai.evaluation.name", name.lower())
            eval_span.set_attribute("gen_ai.evaluation.score", score)
            eval_span.set_attribute(
                "gen_ai.evaluation.result",
                "pass" if score >= 3.0 else "fail",
            )
            if metadata:
                for k, v in metadata.items():
                    # Namespaced under gen_ai.evaluation.* when possible
                    mkey = (
                        k
                        if k.startswith("gen_ai.evaluation.")
                        else f"gen_ai.evaluation.{k}"
                    )
                    eval_span.set_attribute(mkey, v)
            # Include references for ease of querying
            eval_span.set_attribute("gen_ai.evaluation.trace_id", trace_id)
            eval_span.set_attribute("gen_ai.evaluation.span_id", span_id)
            if reasoning:
                eval_span.set_attribute(
                    "gen_ai.evaluation.reasoning", reasoning
                )
    except Exception as e:
        logger.debug("Failed to emit evaluation span: %s", e)


def convert_to_otel_compliant_format(
    trace_records: List[Dict[str, Any]], trace_id: str
) -> Dict[str, Any]:
    """
    Convert Application Insights trace records to OpenTelemetry compliant
    format. See GenAI semantic conventions:
    https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
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
                        "serviceName": os.getenv(
                            "OTEL_SERVICE_NAME", "multi-agent-system"
                        ),
                        "tags": [
                            {
                                "key": "service.name",
                                "type": "string",
                                "value": os.getenv(
                                    "OTEL_SERVICE_NAME", "multi-agent-system"
                                ),
                            },
                            {
                                "key": "service.version",
                                "type": "string",
                                "value": "1.0.0",
                            },
                            {
                                "key": "deployment.environment",
                                "type": "string",
                                "value": "development",
                            },
                            {
                                "key": "telemetry.sdk.name",
                                "type": "string",
                                "value": "opentelemetry",
                            },
                            {
                                "key": "telemetry.sdk.language",
                                "type": "string",
                                "value": "python",
                            },
                            {
                                "key": "telemetry.sdk.version",
                                "type": "string",
                                "value": "1.27.0",
                            },
                        ],
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


def convert_dependency_to_otel_span(
    dep: Dict[str, Any], trace_id: str
) -> Optional[Dict[str, Any]]:
    """Convert a dependency record to an OpenTelemetry compliant span."""
    try:
        custom_dims = dep.get('customDimensions', {})
        if isinstance(custom_dims, str):
            try:
                custom_dims = json.loads(custom_dims)
            except Exception:
                custom_dims = {}

        span_id = (
            dep.get('id', '').replace('-', '')[:16] or generate_span_id(dep)
        )
        parent_span_id = (
            dep.get('operation_ParentId', '').replace('-', '')[:16]
        )

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
            "startTime": int(timestamp.timestamp() * 1_000_000),
            "duration": int(duration_ms * 1000),
            "tags": [],
            "logs": [],
            "processID": "p1",
            "warnings": None,
        }

        # Add parent reference if exists
        if parent_span_id:
            span["references"].append(
                {
                    "refType": "CHILD_OF",
                    "traceID": trace_id,
                    "spanID": parent_span_id,
                }
            )

        # Add OpenTelemetry semantic convention tags
        otel_tags = []

        # Core span attributes
        otel_tags.extend(
            [
                {
                    "key": "span.kind",
                    "type": "string",
                    "value": custom_dims.get('span.kind', 'internal'),
                },
                {
                    "key": "otel.status_code",
                    "type": "string",
                    "value": "OK" if dep.get('success', True) else "ERROR",
                },
            ]
        )

        # GenAI specific attributes
        if gen_ai_operation_name:
            otel_tags.append(
                {
                    "key": "gen_ai.operation.name",
                    "type": "string",
                    "value": gen_ai_operation_name,
                }
            )

        # System and model info
        for key in [
            'gen_ai.system',
            'gen_ai.request.model',
            'gen_ai.response.model',
        ]:
            if key in custom_dims:
                otel_tags.append(
                    {"key": key, "type": "string", "value": custom_dims[key]}
                )

        # Usage metrics
        for key in [
            'gen_ai.usage.input_tokens',
            'gen_ai.usage.output_tokens',
            'gen_ai.usage.total_tokens',
        ]:
            if key in custom_dims:
                otel_tags.append(
                    {
                        "key": key,
                        "type": "int64",
                        "value": int(custom_dims[key]),
                    }
                )

        # Tool-specific attributes
        if "execute_tool" in operation_name.lower() or custom_dims.get(
            'gen_ai.tool.name'
        ):
            tool_name = custom_dims.get(
                'gen_ai.tool.name',
                operation_name.replace('execute_tool', '').strip(),
            )
            otel_tags.append(
                {
                    "key": "gen_ai.tool.name",
                    "type": "string",
                    "value": tool_name,
                }
            )

            if 'gen_ai.tool.call.id' in custom_dims:
                otel_tags.append(
                    {
                        "key": "gen_ai.tool.call.id",
                        "type": "string",
                        "value": custom_dims['gen_ai.tool.call.id'],
                    }
                )

        # Add resource attributes
        if '_MS.ResourceAttributeId' in custom_dims:
            otel_tags.append(
                {
                    "key": "azure.resource.id",
                    "type": "string",
                    "value": custom_dims['_MS.ResourceAttributeId'],
                }
            )

        span["tags"] = otel_tags
        # Add content logs from dependency custom dimensions as separate events
        # so extractors can reliably detect both user and assistant content.
        try:
            if 'gen_ai.prompt' in custom_dims:
                span["logs"].append({
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {
                            "key": "event.name",
                            "type": "string",
                            "value": "gen_ai.content.prompt",
                        },
                        {
                            "key": "gen_ai.prompt",
                            "type": "string",
                            "value": str(custom_dims['gen_ai.prompt']),
                        },
                    ],
                })
            if 'gen_ai.completion' in custom_dims:
                span["logs"].append({
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {
                            "key": "event.name",
                            "type": "string",
                            "value": "gen_ai.content.completion",
                        },
                        {
                            "key": "gen_ai.completion",
                            "type": "string",
                            "value": str(custom_dims['gen_ai.completion']),
                        },
                    ],
                })
            if 'tool_calls' in custom_dims:
                span["logs"].append({
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {
                            "key": "event.name",
                            "type": "string",
                            "value": "gen_ai.tool.call",
                        },
                        {
                            "key": "gen_ai.tool_calls",
                            "type": "string",
                            "value": str(custom_dims['tool_calls']),
                        },
                    ],
                })
        except Exception:
            pass

        return span

    except Exception as e:
        logger.error(f"Error converting dependency to OTEL span: {e}")
        return None


def convert_request_to_otel_span(
    req: Dict[str, Any], trace_id: str
) -> Optional[Dict[str, Any]]:
    """Convert a request record to an OpenTelemetry compliant span."""
    try:
        custom_dims = req.get('customDimensions', {})
        if isinstance(custom_dims, str):
            try:
                custom_dims = json.loads(custom_dims)
            except Exception:
                custom_dims = {}
        
        span_id = (
            req.get('id', '').replace('-', '')[:16] or generate_span_id(req)
        )
        
        timestamp = parse_timestamp(req.get('timestamp'))
        duration_ms = float(req.get('duration', 0))
        
        span = {
            "traceID": trace_id,
            "spanID": span_id,
            "operationName": req.get('name', 'unknown'),
            "references": [],  # Requests are typically root spans
            "startTime": int(
                timestamp.timestamp() * 1_000_000
            ),  # microseconds
            "duration": int(duration_ms * 1000),  # microseconds
            "tags": [],
            "logs": [],
            "processID": "p1",
            "warnings": None
        }
        
        # Add OpenTelemetry semantic convention tags
        otel_tags = [
            {"key": "span.kind", "type": "string", "value": "server"},
            {
                "key": "otel.status_code",
                "type": "string",
                "value": "OK" if req.get('success', True) else "ERROR",
            },
        ]
        
        # HTTP semantic conventions
        if req.get('url'):
            otel_tags.extend(
                [
                    {"key": "http.method", "type": "string", "value": "POST"},
                    {
                        "key": "http.url",
                        "type": "string",
                        "value": req.get('url', ''),
                    },
                    {
                        "key": "http.status_code",
                        "type": "int64",
                        "value": int(req.get('resultCode', 200)),
                    },
                ]
            )
        
        # Add GenAI attributes from custom dimensions
        for key, value in custom_dims.items():
            if key.startswith('gen_ai.'):
                otel_tags.append({
                    "key": key,
                    "type": get_value_type(value),
                    "value": str(value)
                })
        
        span["tags"] = otel_tags
        # Add content logs from request custom dimensions as separate events
        try:
            if 'gen_ai.prompt' in custom_dims:
                span["logs"].append({
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {
                            "key": "event.name",
                            "type": "string",
                            "value": "gen_ai.content.prompt",
                        },
                        {
                            "key": "gen_ai.prompt",
                            "type": "string",
                            "value": str(custom_dims['gen_ai.prompt']),
                        },
                    ],
                })
            if 'gen_ai.completion' in custom_dims:
                span["logs"].append({
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {
                            "key": "event.name",
                            "type": "string",
                            "value": "gen_ai.content.completion",
                        },
                        {
                            "key": "gen_ai.completion",
                            "type": "string",
                            "value": str(custom_dims['gen_ai.completion']),
                        },
                    ],
                })
            if 'tool_calls' in custom_dims:
                span["logs"].append({
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {
                            "key": "event.name",
                            "type": "string",
                            "value": "gen_ai.tool.call",
                        },
                        {
                            "key": "gen_ai.tool_calls",
                            "type": "string",
                            "value": str(custom_dims['tool_calls']),
                        },
                    ],
                })
        except Exception:
            pass
        
        return span
        
    except Exception as e:
        logger.error(f"Error converting request to OTEL span: {e}")
        return None


def add_trace_logs_to_spans(
    spans: List[Dict[str, Any]], traces: List[Dict[str, Any]]
) -> None:
    """
    Add trace records as logs/events to their corresponding spans following
    OTEL conventions.
    """
    span_map = {span["spanID"]: span for span in spans}
    
    for tr in traces:
        parent_id = tr.get('operation_ParentId', '').replace('-', '')[:16]
        if parent_id in span_map:
            timestamp = parse_timestamp(tr.get('timestamp'))
            
            custom_dims = tr.get('customDimensions', {})
            if isinstance(custom_dims, str):
                try:
                    custom_dims = json.loads(custom_dims)
                except Exception:
                    custom_dims = {}
            
            # Check if this is a GenAI event
            message = tr.get('message', '')
            if message in [
                'gen_ai.content.prompt',
                'gen_ai.content.completion',
                'gen_ai.tool.call',
            ]:
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
                # Regular log message, but still surface GenAI fields
                # if present
                log_entry = {
                    "timestamp": int(timestamp.timestamp() * 1_000_000),
                    "fields": [
                        {"key": "message", "type": "string", "value": message},
                        {
                            "key": "level",
                            "type": "string",
                            "value": get_severity_level(
                                tr.get('severityLevel', 0)
                            ),
                        },
                    ]
                }
                # Surface well-known GenAI content even when there is no
                # sentinel event name
                if 'gen_ai.prompt' in custom_dims:
                    log_entry["fields"].append({
                        "key": "gen_ai.prompt",
                        "type": "string",
                        "value": str(custom_dims['gen_ai.prompt'])
                    })
                if 'gen_ai.completion' in custom_dims:
                    log_entry["fields"].append({
                        "key": "gen_ai.completion",
                        "type": "string",
                        "value": str(custom_dims['gen_ai.completion'])
                    })
                if 'gen_ai.event.content' in custom_dims:
                    log_entry["fields"].append({
                        "key": "gen_ai.event.content",
                        "type": "string",
                        "value": str(custom_dims['gen_ai.event.content'])
                    })
                if 'tool_calls' in custom_dims:
                    log_entry["fields"].append({
                        "key": "gen_ai.tool_calls",
                        "type": "string",
                        "value": str(custom_dims['tool_calls'])
                    })
            
            # Add additional custom dimension fields
            for key, value in custom_dims.items():
                if (
                    key not in [
                        'gen_ai.prompt',
                        'gen_ai.completion',
                        'gen_ai.event.content',
                        'tool_calls',
                    ]
                    and value
                ):
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
                if (
                    timestamp_str.endswith(' PM')
                    or timestamp_str.endswith(' AM')
                ):
                    return datetime.strptime(timestamp_str, fmt).replace(
                        tzinfo=timezone.utc
                    )
                else:
                    return datetime.strptime(
                        timestamp_str.replace('Z', ''), fmt
                    ).replace(tzinfo=timezone.utc)
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
    unique_str = (
        f"{record.get('timestamp', '')}-"
        f"{record.get('name', '')}-"
        f"{record.get('message', '')}"
    )
    return hashlib.md5(unique_str.encode()).hexdigest()[:16]


def evaluate_spans(
    otel_data: Dict[str, Any],
    model_config: AzureOpenAIModelConfiguration,
    trace_id: str,
) -> Dict[str, Any]:
    """
    Evaluate ALL relevant spans using Azure AI Evaluation SDK.
    NO LIMITS, NO SHORTCUTS.
    """
    spans = otel_data["data"][0]["spans"]
    # Build index for quick lookups
    span_by_id: Dict[str, Dict[str, Any]] = {s.get("spanID"): s for s in spans}

    def _get_parent_id(sp: Dict[str, Any]) -> Optional[str]:
        try:
            for ref in sp.get("references", []) or []:
                if ref.get("refType") == "CHILD_OF":
                    return ref.get("spanID")
        except Exception:
            pass
        return None
    
    # Initialize evaluators
    tool_accuracy_eval = ToolCallAccuracyEvaluator(model_config=model_config)
    intent_resolution_eval = IntentResolutionEvaluator(
        model_config=model_config
    )
    fluency_eval = FluencyEvaluator(model_config=model_config)
    coherence_eval = CoherenceEvaluator(model_config=model_config)
    groundedness_eval = GroundednessEvaluator(model_config=model_config)
    relevance_eval = RelevanceEvaluator(model_config=model_config)

    def _pick_number(d: Dict[str, Any], keys: List[str]) -> float:
        for k in keys:
            try:
                if k in d and d[k] is not None:
                    return float(d[k])
            except Exception:
                continue
        return 0.0

    def _pick_text(d: Dict[str, Any], keys: List[str]) -> str:
        for k in keys:
            v = d.get(k)
            if isinstance(v, str) and v:
                return v
        # Fallback to first string value
        for v in d.values():
            if isinstance(v, str) and v:
                return v
        return ""

    def _normalize_score_to_five(score: float) -> float:
        # If score is in 0..1, upscale; if already 1..5, keep; cap to [0,5]
        try:
            if score <= 1.0:
                score *= 5.0
            if score < 0:
                score = 0.0
            if score > 5.0:
                score = 5.0
        except Exception:
            score = 0.0
        return score
    
    # Categorize ALL spans - NO LIMITS
    llm_spans: List[Dict[str, Any]] = []
    tool_spans: List[Dict[str, Any]] = []
    tool_span_ids = set()
    final_response_spans: List[Dict[str, Any]] = []
    
    logger.info(f"Processing {len(spans)} spans for evaluation...")
    
    for span in spans:
        operation_name = span.get("operationName", "").lower()
        
        # Check tags for gen_ai.operation.name
        gen_ai_op = None
        has_tool_calls = False
        for tag in span.get("tags", []):
            if tag["key"] == "gen_ai.operation.name":
                gen_ai_op = tag["value"]
            elif tag["key"] == "gen_ai.tool.name":
                has_tool_calls = True
        
        # Identify LLM spans
        if (
            "chat.completions" in operation_name
            or "gpt" in operation_name
            or gen_ai_op == "chat"
        ):
            llm_spans.append(span)
            
            # Check if this span has tool calls
            tool_calls = SpanDataExtractor.extract_tool_calls_from_span(span)
            if tool_calls or has_tool_calls:
                if span["spanID"] not in tool_span_ids:
                    tool_spans.append(span)
                    tool_span_ids.add(span["spanID"])

        # Identify explicit tool spans even if they are not LLM spans
        is_tool_span = (
            "execute_tool" in operation_name
            or (
                gen_ai_op
                and gen_ai_op.lower()
                in {"execute_tool", "tool", "tool_call"}
            )
            or has_tool_calls
        )
        if is_tool_span:
            if span["spanID"] not in tool_span_ids:
                tool_spans.append(span)
                tool_span_ids.add(span["spanID"])
        
        # Check for final response content
        messages = SpanDataExtractor.extract_messages_from_span(span)
        if messages.get("assistant"):
            last_message = (
                messages["assistant"][-1].get("content", "")
                if messages["assistant"]
                else ""
            )
            if (
                len(last_message) > 500
                or "complete" in last_message.lower()
                or "recommendation" in last_message.lower()
            ):
                final_response_spans.append(span)
    
    # Note: we keep llm_spans as a list; we'll search/sort when needed.

    logger.info(
        "Identified spans - LLM: %s, Tool: %s, Final Response: %s",
        len(llm_spans),
        len(tool_spans),
        len(final_response_spans),
    )
    logger.info("Evaluating ALL identified spans - NO LIMITS, NO SHORTCUTS...")
    
    all_results = {
        "intent_resolution": [],
        "tool_accuracy": [],
        "response_quality": [],
        "completeness": []
    }
    
    # 1. Intent Resolution Evaluation - EVALUATE ALL LLM SPANS
    logger.info(
        "Evaluating intent resolution for %s LLM spans...",
        len(llm_spans),
    )
    for i, span in enumerate(llm_spans):
        logger.debug(f"Processing LLM span {i+1}/{len(llm_spans)}")
        try:
            messages = SpanDataExtractor.extract_messages_from_span(span)
            
            # Build conversation safely
            conversation = []
            
            # Add system message if available
            if (
                messages.get("system")
                and isinstance(messages["system"], list)
                and len(messages["system"]) > 0
            ):
                system_msg = messages["system"][0]
                if isinstance(system_msg, dict) and "content" in system_msg:
                    conversation.append(
                        {"role": "system", "content": system_msg["content"]}
                    )
            
            # Add user message if available
            if (
                messages.get("user")
                and isinstance(messages["user"], list)
                and len(messages["user"]) > 0
            ):
                user_msg = messages["user"][0]
                if isinstance(user_msg, dict) and "content" in user_msg:
                    conversation.append(
                        {"role": "user", "content": user_msg["content"]}
                    )
            
            # Add assistant message if available
            if (
                messages.get("assistant")
                and isinstance(messages["assistant"], list)
                and len(messages["assistant"]) > 0
            ):
                assistant_msg = messages["assistant"][-1]
                if (
                    isinstance(assistant_msg, dict)
                    and "content" in assistant_msg
                ):
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg["content"],
                        }
                    )
            
            # Only evaluate if we have a valid conversation
            # Need at least user and assistant messages
            if len(conversation) >= 2:
                try:
                    result = intent_resolution_eval(conversation=conversation)
                    if result and isinstance(result, dict):
                        score = _pick_number(
                            result,
                            [
                                "intent_resolution",
                                "intent_resolution_score",
                                "score",
                            ],
                        )
                        score = _normalize_score_to_five(score)
                        if score > 0:  # Only use if we got a valid score
                            reasoning = _pick_text(
                                result,
                                [
                                    "intent_resolution_reason",
                                    "reason",
                                    "explanation",
                                ],
                            ) or "No reasoning provided"
                            
                            # Emit as OpenTelemetry event
                            emit_evaluation_event(
                                create_evaluation_event(
                                    name="intent_resolution",
                                    score=score,
                                    reasoning=reasoning,
                                    span_id=span["spanID"],
                                    trace_id=trace_id,
                                )
                            )
                            emit_evaluation_span(
                                name="intent_resolution",
                                score=score,
                                reasoning=reasoning,
                                span_id=span["spanID"],
                                trace_id=trace_id,
                            )
                            
                            all_results["intent_resolution"].append(
                                {
                                    "span_id": span["spanID"],
                                    "result": {
                                        "metric": "intent_resolution",
                                        "score": score,
                                        "reasoning": reasoning,
                                    },
                                }
                            )
                except Exception as e:
                    logger.debug(
                        "Intent evaluation failed for span %s: %s",
                        span['spanID'],
                        e,
                    )
            else:
                logger.debug(
                    (
                        "Skipping intent for span %s: "
                        "insufficient conversation (len=%s)"
                    ),
                    span.get("spanID"),
                    len(conversation),
                )
        except Exception as e:
            logger.debug(
                f"Error extracting messages for intent evaluation: {e}"
            )
    
    # 2. Tool Call Accuracy Evaluation - EVALUATE ALL TOOL SPANS
    logger.info(
        "Evaluating tool accuracy for %s tool spans...", len(tool_spans)
    )

    def _inherit_user_query_for_tool_span(sp: Dict[str, Any]) -> str:
        """Find a user query for a tool span by walking ancestors first,
        then by picking the nearest prior LLM span in time."""
        # 1) Walk ancestors via CHILD_OF chain
        visited = set()
        cur = sp
        while cur and cur.get("spanID") not in visited:
            visited.add(cur.get("spanID"))
            try:
                msgs = SpanDataExtractor.extract_messages_from_span(cur)
                if (
                    msgs.get("user")
                    and isinstance(msgs["user"], list)
                    and msgs["user"]
                ):
                    first = msgs["user"][0]
                    if isinstance(first, dict):
                        q = first.get("content", "")
                        if q:
                            return q
            except Exception:
                pass
            parent_id = _get_parent_id(cur)
            if not parent_id:
                break
            cur = span_by_id.get(parent_id)

        # 2) Fallback: nearest prior LLM span by startTime
        try:
            sp_start = sp.get("startTime") or 0
            # choose llm spans that start not after the tool span
            candidates = [
                lsp
                for lsp in llm_spans
                if (lsp.get("startTime") or 0) <= sp_start
            ]
            candidates.sort(
                key=lambda s: s.get("startTime") or 0, reverse=True
            )
            for lsp in candidates:
                msgs = SpanDataExtractor.extract_messages_from_span(lsp)
                if (
                    msgs.get("user")
                    and isinstance(msgs["user"], list)
                    and msgs["user"]
                ):
                    first = msgs["user"][0]
                    if isinstance(first, dict):
                        q = first.get("content", "")
                        if q:
                            return q
        except Exception:
            pass
        return ""
    for i, span in enumerate(tool_spans):
        logger.debug(f"Processing tool span {i+1}/{len(tool_spans)}")
        try:
            tool_calls = SpanDataExtractor.extract_tool_calls_from_span(span)
            messages = SpanDataExtractor.extract_messages_from_span(span)

            # Determine user query: prefer on the span; else inherit
            user_query = ""
            if messages.get("user") and len(messages["user"]) > 0:
                first = messages["user"][0]
                if isinstance(first, dict):
                    user_query = first.get("content", "")
            if not user_query:
                user_query = _inherit_user_query_for_tool_span(span)

            if tool_calls and user_query:
                # Format tool calls for the evaluator
                tools_used = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        tools_used.append({
                            "name": tc.get("name", "unknown"),
                            "arguments": tc.get("arguments", {}),
                            "id": tc.get("id", "")
                        })
                try:
                    # Use the actual ToolCallAccuracyEvaluator
                    result = tool_accuracy_eval(
                        query=user_query,
                        response=json.dumps(tools_used),
                    )

                    if result and isinstance(result, dict):
                        score = _pick_number(
                            result,
                            [
                                "tool_call_accuracy",
                                "tool_call_accuracy_score",
                                "accuracy",
                                "score",
                            ],
                        )
                        score = _normalize_score_to_five(score)
                        if score > 0:
                            reasoning = _pick_text(
                                result,
                                [
                                    "tool_call_accuracy_reason",
                                    "reason",
                                    "explanation",
                                ],
                            ) or "No reasoning provided"

                            # Emit as OpenTelemetry event
                            emit_evaluation_event(
                                create_evaluation_event(
                                    name="tool_call_accuracy",
                                    score=score,
                                    reasoning=reasoning,
                                    span_id=span["spanID"],
                                    trace_id=trace_id,
                                    metadata={
                                        "tools_used": [
                                            t["name"] for t in tools_used
                                        ]
                                    },
                                )
                            )
                            emit_evaluation_span(
                                name="tool_call_accuracy",
                                score=score,
                                reasoning=reasoning,
                                span_id=span["spanID"],
                                trace_id=trace_id,
                                metadata={
                                    "tools_used": [
                                        t["name"] for t in tools_used
                                    ]
                                },
                            )

                            all_results["tool_accuracy"].append(
                                {
                                    "span_id": span["spanID"],
                                    "result": {
                                        "metric": "tool_call_accuracy",
                                        "score": score,
                                        "reasoning": reasoning,
                                    },
                                }
                            )
                except Exception as e:
                    logger.debug(f"Tool accuracy evaluation failed: {e}")
            else:
                logger.debug(
                    (
                        "Skipping tool accuracy for span %s: "
                        "user_query(len=%s), tools=%s"
                    ),
                    span.get("spanID"),
                    len(user_query or ""),
                    len(tool_calls or []),
                )
        except Exception as e:
            logger.debug(f"Tool evaluation error: {e}")
    
    # 3. Response Quality Evaluation - evaluate all LLM and final
    # response spans
    # Fix: Can't use set with dicts, need to deduplicate based on spanID
    seen_span_ids = set()
    spans_to_evaluate = []
    for span in llm_spans + final_response_spans:
        if span["spanID"] not in seen_span_ids:
            seen_span_ids.add(span["spanID"])
            spans_to_evaluate.append(span)
    
    logger.info(
        "Evaluating response quality for %s spans...",
        len(spans_to_evaluate),
    )
    
    for i, span in enumerate(spans_to_evaluate):
        logger.debug(
            "Processing response quality for span %s/%s",
            i + 1,
            len(spans_to_evaluate),
        )
        try:
            messages = SpanDataExtractor.extract_messages_from_span(span)
            
            # Safely extract user query and assistant response
            user_query = ""
            if (
                messages.get("user")
                and isinstance(messages["user"], list)
                and len(messages["user"]) > 0
            ):
                user_msg = messages["user"][0]
                if isinstance(user_msg, dict):
                    user_query = user_msg.get("content", "")
            
            assistant_response = ""
            if (
                messages.get("assistant")
                and isinstance(messages["assistant"], list)
                and len(messages["assistant"]) > 0
            ):
                assistant_msg = messages["assistant"][-1]
                if isinstance(assistant_msg, dict):
                    assistant_response = assistant_msg.get("content", "")
            
            if assistant_response and user_query:
                results = []
                
                # Fluency evaluation
                try:
                    fluency_result = fluency_eval(response=assistant_response)
                    if fluency_result and isinstance(fluency_result, dict):
                        score = _pick_number(
                            fluency_result,
                            ["fluency", "fluency_score", "score"],
                        )
                        score = _normalize_score_to_five(score)
                        reasoning = _pick_text(
                            fluency_result,
                            ["fluency_reason", "reason", "explanation"],
                        ) or "No reasoning provided"
                        
                        if score > 0:
                            emit_evaluation_event(
                                create_evaluation_event(
                                    name="fluency",
                                    score=score,
                                    reasoning=reasoning,
                                    span_id=span["spanID"],
                                    trace_id=trace_id,
                                )
                            )
                            emit_evaluation_span(
                                name="fluency",
                                score=score,
                                reasoning=reasoning,
                                span_id=span["spanID"],
                                trace_id=trace_id,
                            )
                            results.append(
                                {
                                    "metric": "fluency",
                                    "score": score,
                                    "reasoning": reasoning,
                                }
                            )
                except Exception as e:
                    logger.debug(f"Fluency evaluation error: {e}")
                
                # Coherence evaluation
                try:
                    coherence_result = coherence_eval(response=assistant_response)
                    if coherence_result and isinstance(coherence_result, dict):
                        score = _pick_number(
                            coherence_result,
                            ["coherence", "coherence_score", "score"],
                        )
                        score = _normalize_score_to_five(score)
                        reasoning = _pick_text(
                            coherence_result,
                            ["coherence_reason", "reason", "explanation"],
                        ) or "No reasoning provided"
                        if score > 0:
                            emit_evaluation_event(
                                create_evaluation_event(
                                    name="coherence",
                                    score=score,
                                    reasoning=reasoning,
                                    span_id=span["spanID"],
                                    trace_id=trace_id,
                                )
                            )
                            emit_evaluation_span(
                                name="coherence",
                                score=score,
                                reasoning=reasoning,
                                span_id=span["spanID"],
                                trace_id=trace_id,
                            )
                            results.append(
                                {
                                    "metric": "coherence",
                                    "score": score,
                                    "reasoning": reasoning,
                                }
                            )
                except Exception as e:
                    logger.debug(f"Coherence evaluation error: {e}")
                
                # Relevance evaluation
                try:
                    relevance_result = relevance_eval(
                        query=user_query, response=assistant_response
                    )
                    if relevance_result and isinstance(relevance_result, dict):
                        score = _pick_number(
                            relevance_result,
                            ["relevance", "relevance_score", "score"],
                        )
                        score = _normalize_score_to_five(score)
                        reasoning = _pick_text(
                            relevance_result,
                            ["relevance_reason", "reason", "explanation"],
                        ) or "No reasoning provided"
                        
                        if score > 0:
                            emit_evaluation_event(
                                create_evaluation_event(
                                    name="relevance",
                                    score=score,
                                    reasoning=reasoning,
                                    span_id=span["spanID"],
                                    trace_id=trace_id,
                                )
                            )
                            emit_evaluation_span(
                                name="relevance",
                                score=score,
                                reasoning=reasoning,
                                span_id=span["spanID"],
                                trace_id=trace_id,
                            )
                            results.append(
                                {
                                    "metric": "relevance",
                                    "score": score,
                                    "reasoning": reasoning,
                                }
                            )
                except Exception as e:
                    logger.debug(f"Relevance evaluation error: {e}")
                
                # Groundedness evaluation (best-effort: may require context)
                try:
                    # Some SDK versions accept response-only; if context is required,
                    # this will raise and be logged.
                    grounded_result = groundedness_eval(response=assistant_response)
                    if grounded_result and isinstance(grounded_result, dict):
                        score = _pick_number(
                            grounded_result,
                            ["groundedness", "groundedness_score", "score"],
                        )
                        score = _normalize_score_to_five(score)
                        reasoning = _pick_text(
                            grounded_result,
                            ["groundedness_reason", "reason", "explanation"],
                        ) or "No reasoning provided"
                        if score > 0:
                            emit_evaluation_event(
                                create_evaluation_event(
                                    name="groundedness",
                                    score=score,
                                    reasoning=reasoning,
                                    span_id=span["spanID"],
                                    trace_id=trace_id,
                                )
                            )
                            emit_evaluation_span(
                                name="groundedness",
                                score=score,
                                reasoning=reasoning,
                                span_id=span["spanID"],
                                trace_id=trace_id,
                            )
                            results.append(
                                {
                                    "metric": "groundedness",
                                    "score": score,
                                    "reasoning": reasoning,
                                }
                            )
                except Exception as e:
                    logger.debug(f"Groundedness evaluation error: {e}")
            else:
                logger.debug(
                    (
                        "Skipping response quality for span %s: "
                        "user(len=%s), assistant(len=%s)"
                    ),
                    span.get("spanID"),
                    len(user_query or ""),
                    len(assistant_response or ""),
                )
            # Append accumulated results if any
            if results:
                all_results["response_quality"].append({
                    "span_id": span["spanID"],
                    "results": results
                })
        except Exception as e:
            logger.debug(f"Response quality evaluation error: {e}")
    
    logger.info(
        "Evaluation complete. Total evaluations performed: %s",
        sum(len(v) for v in all_results.values()),
    )
    return all_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Extract and evaluate traces from Application Insights'
    )
    parser.add_argument('--trace-id', help='Trace ID to extract')
    parser.add_argument(
        '--output',
        default='trace_with_evaluations_otel.json',
        help='Output file path',
    )
    parser.add_argument(
        '--time-range-hours',
        type=int,
        default=48,
        help='Time range in hours to search for traces',
    )
    parser.add_argument(
        '--input-json',
        help='Local OTEL JSON file to evaluate (bypass App Insights)',
    )
    args = parser.parse_args()
    
    # Configuration
    config = ApplicationInsightsConfig(
        resource_id=os.environ["APPLICATION_INSIGHTS_RESOURCE_ID"]
    )
    
    # Azure OpenAI configuration for evaluation
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"
        ),
        api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
        )
    )
    
    # Two modes: local JSON or App Insights
    if args.input_json:
        logger.info(f"Loading OTEL JSON from: {args.input_json}")
        try:
            with open(args.input_json, "r") as f:
                otel_data = json.load(f)
            # Try to infer trace_id
            if not args.trace_id:
                try:
                    args.trace_id = otel_data["data"][0]["traceID"]
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Failed to read input JSON: {e}")
            return
    else:
        if not args.trace_id:
            logger.error("--trace-id is required when not using --input-json")
            return
        # Connect to Application Insights
        connector = ApplicationInsightsConnector(config)

        # Extract traces
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=args.time_range_hours)

        logger.info(f"Extracting traces for ID: {args.trace_id}")
        logger.info(f"Time range: {start_time} to {end_time}")

        # Build the Kusto query
        kusto_query = f"""
        union dependencies, requests, traces
        | where timestamp between (datetime('{start_time.isoformat()}') ..
                                   datetime('{end_time.isoformat()}'))
        | where operation_Id == "{args.trace_id}"
        | project timestamp, itemType, message, severityLevel,
                  customDimensions, id, operation_ParentId, name,
                  url, success, resultCode, duration, target, type,
                  data, operation_Id, operation_Name
        | order by timestamp asc
        """

        # Use the correct method: _execute_query
        try:
            trace_records = connector._execute_query(
                kusto_query, timespan=timedelta(hours=args.time_range_hours)
            )
            # import pdb;pdb.set_trace()
            logger.info(f"Found {len(trace_records)} records")
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return

        if not trace_records:
            logger.warning("No trace records found. Exiting.")
            return

        # Convert to OpenTelemetry format
        otel_data = convert_to_otel_compliant_format(
            trace_records, args.trace_id
        )
    
    # Evaluate ALL spans - NO LIMITS
    logger.info("Starting comprehensive evaluation of ALL spans...")
    evaluation_results = evaluate_spans(otel_data, model_config, args.trace_id)
    
    # Add evaluation results to the OTEL data
    otel_data["evaluations"] = evaluation_results
    
    # Calculate statistics
    total_evaluations = sum(len(v) for v in evaluation_results.values())
    all_scores = []
    evaluation_breakdown = {}
    
    for eval_type, results in evaluation_results.items():
        evaluation_breakdown[eval_type] = len(results)
        for result in results:
            if "result" in result:
                all_scores.append(result["result"]["score"])
            elif "results" in result:
                for r in result["results"]:
                    all_scores.append(r["score"])
    
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(otel_data, f, indent=2)
    
    logger.info("Saved OTEL-compliant trace data to %s", args.output)
    
    print(f"""
{'='*80}
EXTRACTION AND EVALUATION COMPLETE
{'='*80}
Trace ID: {args.trace_id}
Total records: {len(trace_records)}
Total spans: {len(otel_data['data'][0]['spans'])}
OTEL Semantic Convention: v1.27.0
GenAI Semantic Convention: v1.0.0

Evaluation Results:
- Total evaluations: {total_evaluations}
- Average score: {avg_score:.1f}/5 ({avg_score*20:.0f}%)
- Evaluation breakdown:
{chr(10).join(
        f'  - {k}: {v} evaluations' for k, v in evaluation_breakdown.items()
    )}

Output file: {args.output}
""")


if __name__ == "__main__":
    main()