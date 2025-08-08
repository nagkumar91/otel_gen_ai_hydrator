"""
Unified CLI to extract a trace, evaluate all relevant spans, and emit
evaluation results to the correct backend (Azure Monitor or Jaeger).

Usage examples:
- Azure App Insights source (push events + spans to Azure):
  python evaluate_trace.py --source azure --trace-id <trace_id>

- Jaeger source from a local OTEL JSON (push eval spans/events to Jaeger):
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces \
  python evaluate_trace.py --source jaeger --input-json path/to/otel.json

- Evaluate a local JSON without emitting (dry-run):
  python evaluate_trace.py --source json --input-json path/to/otel.json
"""
from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import argparse

# Evaluators
from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    ToolCallAccuracyEvaluator,
    IntentResolutionEvaluator,
    FluencyEvaluator,
    RelevanceEvaluator,
)

logger = logging.getLogger("evaluate_trace")


def _load_otel_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _build_model_config() -> AzureOpenAIModelConfiguration:
    return AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ.get(
            "AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"
        ),
        api_version=os.environ.get(
            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
        ),
    )


def _evaluate_and_emit_for_azure(
    otel_data: Dict[str, Any], trace_id: str
) -> Dict[str, Any]:
    """Use the existing evaluator pipeline which already emits Azure
    events and spans."""
    import trace_to_json as t2j  # type: ignore

    model_config = _build_model_config()
    results = t2j.evaluate_spans(otel_data, model_config, trace_id)
    return results


def _configure_jaeger_tracer() -> None:
    """Configure an OTLP exporter to Jaeger for emitting evaluation
    spans and events. Uses OTEL_EXPORTER_OTLP_ENDPOINT (default
    http://localhost:4318/v1/traces).
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME

    endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces"
    )
    resource = Resource.create(
        {SERVICE_NAME: os.environ.get("OTEL_SERVICE_NAME", "eval-runner")}
    )
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def _emit_eval_span_and_event(
    *,
    name: str,
    score: float,
    reasoning: str,
    span_id: str,
    trace_id: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit a span with gen_ai.evaluation.* attributes, link to the
    original span, and attach a small event body."""
    from opentelemetry import trace
    from opentelemetry.trace import Link, SpanContext, TraceFlags, TraceState

    tracer = trace.get_tracer(__name__)

    # Try to build a link to the original span
    links = []
    try:
        trace_id_int = int(trace_id, 16)
        span_id_int = int(span_id, 16)
        parent_ctx = SpanContext(
            trace_id=trace_id_int,
            span_id=span_id_int,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
            trace_state=TraceState.get_default(),
        )
        links = [Link(parent_ctx)]
    except Exception:
        links = []

    with tracer.start_as_current_span(
        f"gen_ai.evaluation.{name}", links=links
    ) as sp:
        sp.set_attribute("gen_ai.evaluation.name", name)
        sp.set_attribute("gen_ai.evaluation.score", score)
        sp.set_attribute(
            "gen_ai.evaluation.result", "pass" if score >= 3.0 else "fail"
        )
        sp.set_attribute("gen_ai.evaluation.trace_id", trace_id)
        sp.set_attribute("gen_ai.evaluation.span_id", span_id)
        if metadata:
            for k, v in metadata.items():
                key = (
                    k
                    if k.startswith("gen_ai.evaluation.")
                    else f"gen_ai.evaluation.{k}"
                )
                sp.set_attribute(key, v)
        # Add a small event payload for visibility
        sp.add_event(
            name=f"gen_ai.evaluation.{name}",
            attributes={
                "score": score,
                "reasoning": (
                    reasoning[:1000]
                    if isinstance(reasoning, str)
                    else str(reasoning)
                ),
            },
        )


def _evaluate_for_jaeger(
    otel_data: Dict[str, Any], trace_id: str
) -> Dict[str, Any]:
    """Evaluate spans and emit results to Jaeger as spans with events."""
    from span_extraction import (
        extract_messages_from_span,
        extract_tool_calls_from_span,
    )  # type: ignore

    model_config = _build_model_config()
    tool_eval = ToolCallAccuracyEvaluator(model_config=model_config)
    intent_eval = IntentResolutionEvaluator(model_config=model_config)
    fluency_eval = FluencyEvaluator(model_config=model_config)
    relevance_eval = RelevanceEvaluator(model_config=model_config)

    spans = otel_data.get("data", [{}])[0].get("spans", [])

    llm_spans: List[Dict[str, Any]] = []
    tool_spans: List[Dict[str, Any]] = []
    final_response_spans: List[Dict[str, Any]] = []

    for sp in spans:
        op = (sp.get("operationName") or "").lower()
        tags = {
            t.get("key"): t.get("value")
            for t in sp.get("tags", [])
            if isinstance(t, dict)
        }
        gen_op = tags.get("gen_ai.operation.name")
        if "chat.completions" in op or "gpt" in op or gen_op == "chat":
            llm_spans.append(sp)
            if (
                extract_tool_calls_from_span(sp)
                or tags.get("gen_ai.tool.name")
            ):
                tool_spans.append(sp)

        msgs = extract_messages_from_span(sp)
        if msgs.get("assistant"):
            last = msgs["assistant"][-1].get("content", "")
            if last and (
                len(last) > 500
                or "complete" in last.lower()
                or "recommendation" in last.lower()
            ):
                final_response_spans.append(sp)

    results: Dict[str, Any] = {
        "intent_resolution": [],
        "tool_accuracy": [],
        "response_quality": [],
        "completeness": [],
    }

    # Intent resolution
    for sp in llm_spans:
        msgs = extract_messages_from_span(sp)
        convo: List[Dict[str, str]] = []
        if msgs.get("system"):
            sys = msgs["system"][0]
            if isinstance(sys, dict) and sys.get("content"):
                convo.append({"role": "system", "content": sys["content"]})
        if msgs.get("user"):
            usr = msgs["user"][0]
            if isinstance(usr, dict) and usr.get("content"):
                convo.append({"role": "user", "content": usr["content"]})
        if msgs.get("assistant"):
            ass = msgs["assistant"][-1]
            if isinstance(ass, dict) and ass.get("content"):
                convo.append({"role": "assistant", "content": ass["content"]})

        if len(convo) >= 2:
            try:
                r = intent_eval(conversation=convo)
                score = (
                    float(r.get("intent_resolution_score", 0.0))
                    if isinstance(r, dict)
                    else 0.0
                )
                if score > 0:
                    score *= 5
                    reason = (
                        r.get("intent_resolution_reason", "")
                        if isinstance(r, dict)
                        else ""
                    )
                    _emit_eval_span_and_event(
                        name="intent_resolution",
                        score=score,
                        reasoning=reason,
                        span_id=sp["spanID"],
                        trace_id=trace_id,
                    )
                    results["intent_resolution"].append({
                        "span_id": sp["spanID"],
                        "result": {
                            "metric": "intent_resolution",
                            "score": score,
                            "reasoning": reason,
                        },
                    })
            except Exception as e:
                logger.debug("Intent evaluation error: %s", e)

    # Tool call accuracy
    for sp in tool_spans:
        try:
            tcs = extract_tool_calls_from_span(sp)
            msgs = extract_messages_from_span(sp)
            q = (
                msgs.get("user", [{}])[0].get("content", "")
                if msgs.get("user")
                else ""
            )
            tools_used = [
                {
                    "name": tc.get("name", "unknown"),
                    "arguments": tc.get("arguments", {}),
                    "id": tc.get("id", ""),
                }
                for tc in tcs
                if isinstance(tc, dict)
            ]
            if q and tools_used:
                r = tool_eval(query=q, response=json.dumps(tools_used))
                score = (
                    float(r.get("tool_call_accuracy", 0.0))
                    if isinstance(r, dict)
                    else 0.0
                )
                if score > 0:
                    score *= 5
                    reason = (
                        r.get("tool_call_accuracy_reason", "")
                        if isinstance(r, dict)
                        else ""
                    )
                    _emit_eval_span_and_event(
                        name="tool_call_accuracy",
                        score=score,
                        reasoning=reason,
                        span_id=sp["spanID"],
                        trace_id=trace_id,
                        metadata={
                            "tools_used": [t["name"] for t in tools_used]
                        },
                    )
                    results["tool_accuracy"].append({
                        "span_id": sp["spanID"],
                        "result": {
                            "metric": "tool_call_accuracy",
                            "score": score,
                            "reasoning": reason,
                        },
                    })
        except Exception as e:
            logger.debug("Tool accuracy evaluation error: %s", e)

    # Response quality (fluency + relevance)
    seen = set()
    to_eval: List[Dict[str, Any]] = []
    for sp in llm_spans + final_response_spans:
        if sp["spanID"] not in seen:
            seen.add(sp["spanID"])
            to_eval.append(sp)

    for sp in to_eval:
        msgs = extract_messages_from_span(sp)
        q = (
            msgs.get("user", [{}])[0].get("content", "")
            if msgs.get("user")
            else ""
        )
        a = (
            msgs.get("assistant", [{}])[-1].get("content", "")
            if msgs.get("assistant")
            else ""
        )
        if not (q and a):
            continue
        # Fluency
        try:
            fr = fluency_eval(response=a)
            fscore = (
                float(fr.get("fluency", 0)) if isinstance(fr, dict) else 0
            )
            if fscore > 0:
                freason = (
                    fr.get("fluency_reason", "")
                    if isinstance(fr, dict)
                    else ""
                )
                _emit_eval_span_and_event(
                    name="fluency",
                    score=fscore,
                    reasoning=freason,
                    span_id=sp["spanID"],
                    trace_id=trace_id,
                )
                results["response_quality"].append({
                    "span_id": sp["spanID"],
                    "results": [
                        {
                            "metric": "fluency",
                            "score": fscore,
                            "reasoning": freason,
                        }
                    ],
                })
        except Exception as e:
            logger.debug("Fluency evaluation error: %s", e)
        # Relevance
        try:
            rr = relevance_eval(query=q, response=a)
            rscore = (
                float(rr.get("relevance", 0)) if isinstance(rr, dict) else 0
            )
            if rscore > 0:
                rreason = (
                    rr.get("relevance_reason", "")
                    if isinstance(rr, dict)
                    else ""
                )
                _emit_eval_span_and_event(
                    name="relevance",
                    score=rscore,
                    reasoning=rreason,
                    span_id=sp["spanID"],
                    trace_id=trace_id,
                )
                # Merge with previous fluency if existed for this span
                appended = False
                for item in results["response_quality"]:
                    if item["span_id"] == sp["spanID"]:
                        item.setdefault("results", []).append(
                            {
                                "metric": "relevance",
                                "score": rscore,
                                "reasoning": rreason,
                            }
                        )
                        appended = True
                        break
                if not appended:
                    results["response_quality"].append({
                        "span_id": sp["spanID"],
                        "results": [
                            {
                                "metric": "relevance",
                                "score": rscore,
                                "reasoning": rreason,
                            }
                        ],
                    })
        except Exception as e:
            logger.debug("Relevance evaluation error: %s", e)

    return results


def _fetch_from_azure(trace_id: str, hours: int) -> Dict[str, Any]:
    from otel_gen_ai_hydrator.sources.application_insights import (
        ApplicationInsightsConnector,
        ApplicationInsightsConfig,
    )
    from trace_to_json import convert_to_otel_compliant_format  # type: ignore

    resource_id = os.environ["APPLICATION_INSIGHTS_RESOURCE_ID"]
    config = ApplicationInsightsConfig(resource_id=resource_id)
    connector = ApplicationInsightsConnector(config)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=hours)
    kusto_query = f"""
    union dependencies, requests, traces
    | where timestamp between (
        datetime('{start_time.isoformat()}') ..
        datetime('{end_time.isoformat()}')
      )
    | where operation_Id == "{trace_id}"
    | project timestamp, itemType, message, severityLevel,
              customDimensions, id, operation_ParentId, name,
              url, success, resultCode, duration, target, type,
              data, operation_Id, operation_Name
    | order by timestamp asc
    """
    records = connector._execute_query(
        kusto_query, timespan=timedelta(hours=hours)
    )  # noqa: SLF001
    if not records:
        raise RuntimeError(
            "No records found for the provided trace_id and window."
        )
    return convert_to_otel_compliant_format(records, trace_id)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trace and emit results to Azure Monitor or Jaeger"
        )
    )
    parser.add_argument(
        "--source", choices=["azure", "jaeger", "json"], required=True
    )
    parser.add_argument(
        "--trace-id", help="Trace ID when source=azure or jaeger"
    )
    parser.add_argument(
        "--input-json",
        help="Local OTEL JSON file (for source=json or jaeger)",
    )
    parser.add_argument("--time-range-hours", type=int, default=48)
    parser.add_argument(
        "--output", help="Write enriched OTEL JSON to this path"
    )
    args = parser.parse_args()

    # Acquire OTEL JSON either from Azure or local JSON
    if args.source == "azure":
        if not args.trace_id:
            raise SystemExit("--trace-id is required for source=azure")
        otel_data = _fetch_from_azure(args.trace_id, args.time_range_hours)
        # Evaluate and emit to Azure Monitor (events + spans)
        results = _evaluate_and_emit_for_azure(otel_data, args.trace_id)
    else:
        if not args.input_json:
            raise SystemExit("--input-json is required for source=jaeger|json")
        otel_data = _load_otel_json(args.input_json)
        # If explicit trace id not provided, try reading it from file
        if not args.trace_id:
            try:
                args.trace_id = otel_data["data"][0]["traceID"]
            except Exception:
                args.trace_id = ""
        if args.source == "jaeger":
            _configure_jaeger_tracer()
            results = _evaluate_for_jaeger(otel_data, args.trace_id)
        else:
            # Dry-run: compute using Azure pipeline without monitor
            # Avoid Azure emissions by not setting connection string
            os.environ.setdefault("APPLICATION_INSIGHTS_CONNECTION_STRING", "")
            results = _evaluate_and_emit_for_azure(otel_data, args.trace_id)

    # Attach results and optionally write output
    otel_data["evaluations"] = results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(otel_data, f, indent=2)
        logger.info("Wrote enriched OTEL JSON to %s", args.output)

    # Quick summary
    total = sum(len(v) for v in results.values())
    logger.info("Evaluation complete. Total evaluations: %s", total)


if __name__ == "__main__":
    main()
