"""
Retrieve a specific root (or any) span by trace ID + span ID from Azure Application Insights,
emit a full JSON dump, optionally emit an operation details event (gen_ai.client.inference.operation.details),
run several Azure AI evaluation metrics on the span's GenAI input/output content, and
emit gen_ai.evaluation.result events following the latest GenAI semantic conventions.

Evaluators used:
- IntentResolutionEvaluator
- ToolCallAccuracyEvaluator
- TaskAdherenceEvaluator

Requirements:
  Environment:
    APPLICATION_INSIGHTS_CONNECTION_STRING (for emitting new events)
    APPLICATION_INSIGHTS_RESOURCE_ID (for querying – if using the custom connector)
    AZURE_OPENAI_API_KEY (+ endpoint/deployment vars) for evaluation model
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_DEPLOYMENT_NAME (default: gpt-4)
    AZURE_OPENAI_API_VERSION (default: 2024-08-01-preview)
    OTEL_EXPORTER_OTLP_ENDPOINT (optional additional OTLP export for emitted events)

Usage:
  python trace_root_span_evaluator.py --trace-id <trace_id> --span-id <span_id> \
    --output root_span_dump.json

Notes:
- The script focuses on the target span; it still downloads the entire trace to extract
  tool calls (child execute_tool spans) and tool definitions if present.
- Operation details event is opt-in via flag (on by default).
"""

import os
import json
import argparse
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import hashlib
import sys
import time

# Logging setup
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies").setLevel(logging.WARNING)
logging.getLogger("azure.monitor").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

# Azure Monitor setup (for emitting events)
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
except ImportError:
    configure_azure_monitor = None

# OpenTelemetry
from opentelemetry import trace
from opentelemetry._events import Event, get_event_logger
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Evaluators
from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    IntentResolutionEvaluator,
    ToolCallAccuracyEvaluator,
    TaskAdherenceEvaluator,
)

# Application Insights connector (project local)
try:
    from otel_gen_ai_hydrator.sources.application_insights import (
        ApplicationInsightsConnector,
        ApplicationInsightsConfig,
    )
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from otel_gen_ai_hydrator.sources.application_insights import (
        ApplicationInsightsConnector,
        ApplicationInsightsConfig,
    )

# Configure Azure Monitor if connection string present
_app_insights_conn = os.environ.get("APPLICATION_INSIGHTS_CONNECTION_STRING")
if configure_azure_monitor and _app_insights_conn and "=" in _app_insights_conn:
    try:
        configure_azure_monitor(
            connection_string=_app_insights_conn,
            logging_enabled=False,
        )
        logger.info("Azure Monitor configured for event export.")
    except Exception as e:
        logger.warning(f"Azure Monitor configuration failed: {e}")
else:
    logger.info("Azure Monitor not configured (missing / invalid connection string).")

# Optional OTLP exporter
otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
if otlp_endpoint:
    resource = Resource.create({"service.name": "root-span-evaluator", "service.version": "1.0.0"})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{otlp_endpoint}/v1/traces"))
    )
    trace.set_tracer_provider(provider)

event_logger = get_event_logger(__name__)


# ---------- Helpers ----------

def parse_timestamp(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            if ts.endswith("Z"):
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if "+" in ts:
                return datetime.fromisoformat(ts)
            return datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def get_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int64"
    if isinstance(value, float):
        return "float64"
    return "string"


def safe_json_load(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def json_default(o: Any):
    """JSON serializer for objects not serializable by default json code.

    Currently handles datetime (ISO 8601) and falls back to str() for others.
    """
    if isinstance(o, datetime):
        # Ensure timezone-aware datetimes are preserved
        if o.tzinfo is None:
            return o.replace(tzinfo=timezone.utc).isoformat()
        return o.isoformat()
    try:
        import decimal  # local import to avoid unnecessary dependency upfront
        if isinstance(o, decimal.Decimal):  # type: ignore[attr-defined]
            return float(o)
    except Exception:
        pass
    return str(o)


# ---------- Trace Retrieval & Parsing ----------

def retrieve_trace_records(
    trace_id: str,
    config: ApplicationInsightsConfig,
    time_range_hours: int = 48,
) -> List[Dict[str, Any]]:
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=time_range_hours)
    connector = ApplicationInsightsConnector(config)

    query = f"""
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
        rows = connector._execute_query(query, timespan=timedelta(hours=time_range_hours))
        logger.info(f"Retrieved {len(rows)} telemetry rows for trace {trace_id}")
        return rows
    except Exception as e:
        logger.error(f"Failed to query Application Insights: {e}")
        return []


def build_span_structures(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    spans: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        item_type = r.get("itemType", "").lower()
        if item_type not in ("request", "dependency"):
            continue

        raw_id = (r.get("id") or "").replace("-", "")
        span_id = raw_id[:16] if raw_id else hashlib.md5(
            f"{r.get('timestamp')}-{r.get('name')}".encode()
        ).hexdigest()[:16]

        custom_dims = r.get("customDimensions", {})
        if isinstance(custom_dims, str):
            custom_dims = safe_json_load(custom_dims)
            if not isinstance(custom_dims, dict):
                custom_dims = {}

        parent_id = (r.get("operation_ParentId") or "").replace("-", "")[:16] or None
        spans[span_id] = {
            "raw": r,
            "span_id": span_id,
            "parent_id": parent_id,
            "name": r.get("name"),
            "item_type": item_type,
            "custom_dimensions": custom_dims,
            "timestamp": r.get("timestamp"),
            "duration": r.get("duration"),
        }
    return spans


def extract_tool_calls(spans: Dict[str, Dict[str, Any]], root_span_id: str) -> List[Dict[str, Any]]:
    calls = []
    for s in spans.values():
        if s.get("parent_id") != root_span_id:
            continue
        cd = s.get("custom_dimensions", {})
        # Execute tool spans typically have gen_ai.tool.* attributes
        if any(k.startswith("gen_ai.tool.") for k in cd.keys()):
            call = {
                "type": "tool_call",
                "tool_call_id": cd.get("gen_ai.tool.call.id") or s["span_id"],
                "name": cd.get("gen_ai.tool.name"),
                "arguments": safe_json_load(cd.get("gen_ai.tool.call.arguments")) or {},
                "description": cd.get("gen_ai.tool.description"),
            }
            # Optionally attach result
            if "gen_ai.tool.call.result" in cd:
                call["result"] = safe_json_load(cd["gen_ai.tool.call.result"])
            calls.append(call)
    return calls


def extract_tool_definitions(root_cd: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    defs = root_cd.get("gen_ai.tool.definitions")
    if defs:
        defs = safe_json_load(defs)
        if isinstance(defs, list):
            return defs
    return None


def extract_messages(root_cd: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    input_msgs_raw = root_cd.get("gen_ai.input.messages")
    output_msgs_raw = root_cd.get("gen_ai.output.messages")

    def norm(msgs):
        msgs = safe_json_load(msgs)
        return msgs if isinstance(msgs, list) else []

    return norm(input_msgs_raw), norm(output_msgs_raw)


# ---------- Events (Spec Compliant) ----------

def create_operation_details_event(root_cd: Dict[str, Any], trace_id: str, span_id: str) -> Event:
    attrs = {}
    # Copy standard inference attributes if present
    copy_keys = [
        "gen_ai.provider.name",
        "gen_ai.operation.name",
        "gen_ai.request.model",
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
        "gen_ai.request.top_k",
        "gen_ai.request.max_tokens",
        "gen_ai.request.frequency_penalty",
        "gen_ai.request.presence_penalty",
        "gen_ai.request.stop_sequences",
        "gen_ai.conversation.id",
        "gen_ai.agent.id",
        "gen_ai.agent.name",
        "gen_ai.orchestrator.agent_definitions",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.output.type",
        "gen_ai.system_instructions",
    ]
    for k in copy_keys:
        if k in root_cd:
            attrs[k] = root_cd[k]

    # Convert IDs for event object (OTel internal expects ints)
    span_id_int = int(span_id, 16) if span_id else 0
    trace_id_int = int(trace_id.replace("-", ""), 16) if trace_id else 0

    return Event(
        name="gen_ai.client.inference.operation.details",
        attributes=attrs,
        body="GenAI operation details snapshot",
        span_id=span_id_int,
        trace_id=trace_id_int,
    )


def create_evaluation_event(
    *,
    metric: str,
    score: float,
    label: str,
    explanation: str,
    trace_id: str,
    span_id: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    response_id: Optional[str] = None,
    error_type: Optional[str] = None,
) -> Event:
    attrs = {
        "gen_ai.evaluation.name": metric,
        "gen_ai.evaluation.score.value": float(score),
        "gen_ai.evaluation.score.label": label,
        "gen_ai.evaluation.explanation": explanation[:2000],
    }
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if error_type:
        attrs["error.type"] = error_type

    span_id_int = int(span_id, 16) if span_id else 0
    trace_id_int = int(trace_id.replace("-", ""), 16) if trace_id else 0
    return Event(
        name="gen_ai.evaluation.result",
        attributes=attrs,
        body=f"Evaluation {metric}: {score:.2f}",
        span_id=span_id_int,
        trace_id=trace_id_int,
    )


# ---------- Evaluation Logic ----------

def normalize_score(score: float) -> float:
    # Many Azure evaluators return 0-5 already; if 0-1 scale, expand.
    if score <= 1.0:
        return score * 5.0
    return min(score, 5.0)


def derive_label(score: float, threshold: float = 3.0) -> str:
    return "pass" if score >= threshold else "fail"


def run_evaluations(
    *,
    model_config: AzureOpenAIModelConfiguration,
    query: str,
    response: str,
    tool_calls: List[Dict[str, Any]],
    tool_definitions: Optional[List[Dict[str, Any]]],
    trace_id: str,
    span_id: str,
    emit_events: bool = True,
    threshold: float = 3.0,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    # Instantiate evaluators
    intent_eval = IntentResolutionEvaluator(model_config=model_config, threshold=threshold)
    tool_eval = ToolCallAccuracyEvaluator(model_config=model_config, threshold=threshold)
    task_eval = TaskAdherenceEvaluator(model_config=model_config, threshold=threshold)

    # IntentResolution
    try:
        r = intent_eval(query=query, response=response)
        score = normalize_score(float(r.get("score", r.get("intent_resolution", 0.0))))
        explanation = r.get("reasoning") or r.get("explanation") or "n/a"
        res = {
            "metric": "intent_resolution",
            "score": score,
            "label": derive_label(score, threshold),
            "reasoning": explanation,
        }
        results.append(res)
        if emit_events:
            event_logger.emit(
                create_evaluation_event(
                    metric=res["metric"],
                    score=res["score"],
                    label=res["label"],
                    explanation=res["reasoning"],
                    trace_id=trace_id,
                    span_id=span_id,
                    input_tokens=r.get("usage", {}).get("input_tokens"),
                    output_tokens=r.get("usage", {}).get("output_tokens"),
                    response_id=r.get("response_id"),
                )
            )
    except Exception as e:
        logger.debug(f"Intent resolution evaluation failed: {e}")
        if emit_events:
            event_logger.emit(
                create_evaluation_event(
                    metric="intent_resolution",
                    score=0.0,
                    label="fail",
                    explanation=f"Evaluation error: {e}",
                    trace_id=trace_id,
                    span_id=span_id,
                    error_type=type(e).__name__,
                )
            )

    # ToolCallAccuracy (only if we have tool calls or definitions)
    if tool_calls or tool_definitions:
        try:
            # Align param names with evaluator signature (tool_calls, tool_definitions)
            r = tool_eval(
                query=query,
                tool_calls=tool_calls or [],
                tool_definitions=tool_definitions or [],
            )
            score = normalize_score(float(r.get("score", r.get("tool_call_accuracy", 0.0))))
            explanation = r.get("reasoning") or r.get("explanation") or "n/a"
            res = {
                "metric": "tool_call_accuracy",
                "score": score,
                "label": derive_label(score, threshold),
                "reasoning": explanation,
            }
            results.append(res)
            if emit_events:
                event_logger.emit(
                    create_evaluation_event(
                        metric=res["metric"],
                        score=res["score"],
                        label=res["label"],
                        explanation=res["reasoning"],
                        trace_id=trace_id,
                        span_id=span_id,
                        input_tokens=r.get("usage", {}).get("input_tokens"),
                        output_tokens=r.get("usage", {}).get("output_tokens"),
                        response_id=r.get("response_id"),
                    )
                )
        except Exception as e:
            logger.debug(f"Tool call accuracy evaluation failed: {e}")
            if emit_events:
                event_logger.emit(
                    create_evaluation_event(
                        metric="tool_call_accuracy",
                        score=0.0,
                        label="fail",
                        explanation=f"Evaluation error: {e}",
                        trace_id=trace_id,
                        span_id=span_id,
                        error_type=type(e).__name__,
                    )
                )

    # TaskAdherence
    try:
        r = task_eval(query=query, response=response)
        score = normalize_score(float(r.get("score", r.get("task_adherence", 0.0))))
        explanation = r.get("reasoning") or r.get("explanation") or "n/a"
        res = {
            "metric": "task_adherence",
            "score": score,
            "label": derive_label(score, threshold),
            "reasoning": explanation,
        }
        results.append(res)
        if emit_events:
            event_logger.emit(
                create_evaluation_event(
                    metric=res["metric"],
                    score=res["score"],
                    label=res["label"],
                    explanation=res["reasoning"],
                    trace_id=trace_id,
                    span_id=span_id,
                    input_tokens=r.get("usage", {}).get("input_tokens"),
                    output_tokens=r.get("usage", {}).get("output_tokens"),
                    response_id=r.get("response_id"),
                )
            )
    except Exception as e:
        logger.debug(f"Task adherence evaluation failed: {e}")
        if emit_events:
            event_logger.emit(
                create_evaluation_event(
                    metric="task_adherence",
                    score=0.0,
                    label="fail",
                    explanation=f"Evaluation error: {e}",
                    trace_id=trace_id,
                    span_id=span_id,
                    error_type=type(e).__name__,
                )
            )

    return results


# ---------- Verification of Evaluation Events ----------

def retrieve_evaluation_events(
    *,
    trace_id: str,
    span_id: str,
    config: ApplicationInsightsConfig,
    lookback_minutes: int = 30,
    retries: int = 5,
    delay_seconds: float = 2.0,
) -> List[Dict[str, Any]]:
    """Query Application Insights for emitted gen_ai.evaluation.result events.

    We search both traces and customEvents tables (events from OTel may surface in either
    depending on exporter mapping). Filtering by operation_Id (trace) and presence of
    gen_ai.evaluation.name in customDimensions. We retry a few times to allow ingestion delay.
    """
    connector = ApplicationInsightsConnector(
        config
    )
    collected: List[Dict[str, Any]] = []
    for attempt in range(1, retries + 1):
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=lookback_minutes)
        query = f"""
        union traces, customEvents
        | where timestamp between (datetime('{start_time.isoformat()}') .. datetime('{end_time.isoformat()}'))
        | where operation_Id == "{trace_id}"
        | where tostring(customDimensions) contains "gen_ai.evaluation.name"
        | project timestamp, itemType, name, message, customDimensions, operation_Id, operation_ParentId
        | order by timestamp asc
        """
        try:
            rows = connector._execute_query(query, timespan=timedelta(minutes=lookback_minutes))
        except Exception as e:
            logger.debug(f"Evaluation event query failed (attempt {attempt}): {e}")
            rows = []
        parsed: List[Dict[str, Any]] = []
        for r in rows:
            cd = r.get("customDimensions", {})
            if isinstance(cd, str):
                cd_loaded = safe_json_load(cd)
                if isinstance(cd_loaded, dict):
                    cd = cd_loaded
                else:
                    cd = {}
            # Only accept rows with evaluation key
            if any(k.startswith("gen_ai.evaluation.") for k in cd.keys()):
                parsed.append(
                    {
                        "timestamp": r.get("timestamp"),
                        "name": r.get("name") or r.get("message"),
                        "customDimensions": cd,
                        "metric": cd.get("gen_ai.evaluation.name"),
                        "score": cd.get("gen_ai.evaluation.score.value"),
                        "label": cd.get("gen_ai.evaluation.score.label"),
                        "explanation": cd.get("gen_ai.evaluation.explanation"),
                    }
                )
        if parsed:
            collected = parsed
            break
        if attempt < retries:
            time.sleep(delay_seconds)
    if not collected:
        logger.warning("No evaluation events located in Application Insights after retries.")
    else:
        logger.info(f"Retrieved {len(collected)} evaluation events from Application Insights.")
    return collected


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a specific GenAI root span by trace + span ID.")
    parser.add_argument("--trace-id", required=True, help="Trace ID (operation_Id)")
    parser.add_argument("--span-id", required=True, help="Target span ID (hex, first 16 chars)")
    parser.add_argument("--output", default=None, help="File to save full span JSON & eval results (default: <trace>_<span>.json)")
    parser.add_argument("--time-range-hours", type=int, default=48)
    parser.add_argument("--no-operation-event", action="store_true", help="Disable emission of operation details event")
    parser.add_argument("--no-eval-events", action="store_true", help="Disable emission of evaluation events")
    parser.add_argument("--threshold", type=float, default=3.0, help="Pass/fail threshold for evaluators")
    parser.add_argument("--verify-retries", type=int, default=5, help="Retries while searching for evaluation events")
    parser.add_argument("--verify-delay", type=float, default=2.0, help="Delay (s) between verification retries")
    parser.add_argument(
        "--verify-lookback-mins",
        type=int,
        default=30,
        help="Lookback window (minutes) when querying for evaluation events",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification query for evaluation events",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Query trace
    app_config = ApplicationInsightsConfig(
        resource_id=os.environ.get("APPLICATION_INSIGHTS_RESOURCE_ID")
    )
    records = retrieve_trace_records(
        args.trace_id, app_config, args.time_range_hours
    )
    if not records:
        logger.error("No telemetry records found for that trace ID.")
        return

    spans = build_span_structures(records)
    target_span = spans.get(args.span_id)
    if not target_span:
        logger.error(f"Span ID {args.span_id} not found in trace.")
        # Offer list
        logger.info(f"Available span IDs: {', '.join(spans.keys())}")
        return

    root_cd = target_span["custom_dimensions"] or {}
    input_messages, output_messages = extract_messages(root_cd)
    tool_calls = extract_tool_calls(spans, args.span_id)
    tool_definitions = extract_tool_definitions(root_cd)

    # Derive query & response
    query = ""
    if input_messages:
        # First user or first message content
        for m in input_messages:
            if m.get("role") == "user":
                parts = m.get("parts") or []
                for p in parts:
                    if p.get("type") == "text":
                        query = p.get("content", "")
                        break
                if query:
                    break
        if not query:
            # fallback: first message text
            first_parts = input_messages[0].get("parts") or []
            query = next(
                (
                    p.get("content", "")
                    for p in first_parts
                    if p.get("type") == "text"
                ),
                "",
            )

    response_text = ""
    if output_messages:
        # Use the last assistant message
        for msg in reversed(output_messages):
            if msg.get("role") in ("assistant", "ai"):
                parts = msg.get("parts") or []
                for p in parts:
                    if p.get("type") == "text":
                        response_text = p.get("content", "")
                        break
            if response_text:
                break

    logger.info(
        "Extracted query length=%d response length=%d tool_calls=%d",
        len(query),
        len(response_text),
        len(tool_calls),
    )

    # Build model config
    try:
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ.get(
                "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"
            ),
            api_version=os.environ.get(
                "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
            ),
        )
    except KeyError as e:
        logger.error(f"Missing required Azure OpenAI env var: {e}")
        return

    # Optionally emit operation details event
    if not args.no_operation_event:
        try:
            op_event = create_operation_details_event(
                root_cd, args.trace_id, args.span_id
            )
            event_logger.emit(op_event)
            logger.info(
                "Emitted gen_ai.client.inference.operation.details event."
            )
        except Exception as e:
            logger.warning(f"Failed to emit operation details event: {e}")

    # Run evaluations
    evaluations: List[Dict[str, Any]] = []
    if query and response_text:
        evaluations = run_evaluations(
            model_config=model_config,
            query=query,
            response=response_text,
            tool_calls=tool_calls,
            tool_definitions=tool_definitions,
            trace_id=args.trace_id,
            span_id=args.span_id,
            emit_events=not args.no_eval_events,
            threshold=args.threshold,
        )
    else:
        logger.warning(
            "Insufficient data (query/response) to run primary evaluations."
        )

    # Force flush if possible
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        try:
            provider.force_flush()
        except Exception:
            pass

    # Optionally verify evaluation events reached Application Insights
    evaluation_events: List[Dict[str, Any]] = []
    if evaluations and not args.no_eval_events and not args.no_verify:
        try:
            evaluation_events = retrieve_evaluation_events(
                trace_id=args.trace_id,
                span_id=args.span_id,
                config=app_config,
                lookback_minutes=args.verify_lookback_mins,
                retries=args.verify_retries,
                delay_seconds=args.verify_delay,
            )
        except Exception as e:
            logger.debug(f"Verification of evaluation events failed: {e}")

    # Determine output filename if not provided
    if not args.output:
        args.output = f"{args.trace_id}_{args.span_id}.json"

    # Assemble output JSON
    output_blob = {
        "trace_id": args.trace_id,
        "span_id": args.span_id,
        "span_name": target_span["name"],
        "item_type": target_span["item_type"],
        "timestamp": target_span["timestamp"],
        "duration": target_span["duration"],
        "parent_id": target_span["parent_id"],
        "custom_dimensions": root_cd,
        "input_messages": input_messages,
        "output_messages": output_messages,
        "tool_calls": tool_calls,
        "tool_definitions": tool_definitions,
        "query_extracted": query,
        "response_extracted": response_text,
        "evaluations": evaluations,
        "evaluation_events_found": evaluation_events,
        "evaluation_events_count": len(evaluation_events),
        "evaluation_summary": {
            "count": len(evaluations),
            "metrics": [e["metric"] for e in evaluations],
            "avg_score": (
                sum(e["score"] for e in evaluations) / len(evaluations)
            )
            if evaluations
            else 0.0,
            "pass_rate": (
                sum(
                    1 for e in evaluations if e["score"] >= args.threshold
                )
                / len(evaluations)
            )
            if evaluations
            else 0.0,
            "events_ingested": bool(evaluation_events)
            if evaluations
            else False,
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_blob, f, indent=2, default=json_default)

    # Print full JSON to stdout (as requested)
    print(json.dumps(output_blob, indent=2, default=json_default))

    logger.info(f"Saved span + evaluation data to {args.output}")


if __name__ == "__main__":
    main()
