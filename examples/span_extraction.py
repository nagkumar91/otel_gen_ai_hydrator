"""
Utilities to robustly extract messages and tool calls from OTEL GenAI spans.

This module is domain-agnostic and attempts to handle common patterns we see in
LangChain/LangGraph + OTEL exporters. It favors the GenAI semantic conventions
when present and falls back to tags/log fields.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

Json = Dict[str, Any]


def _safe_json_loads(val: Any) -> Any:
    try:
        import json

        if isinstance(val, str):
            return json.loads(val)
    except Exception:
        pass
    return val


def _fields_list_to_dict(fields: Any) -> Dict[str, Any]:
    """Convert span event/log fields list[{key,value}] to dict safely."""
    if isinstance(fields, dict):
        return fields
    out: Dict[str, Any] = {}
    if isinstance(fields, list):
        for item in fields:
            if not isinstance(item, dict):
                continue
            k = item.get("key") or item.get("name")
            v = item.get("value")
            out[k] = _safe_json_loads(v)
    return out


def _infer_fields_from_values_list(values: List[Any]) -> Dict[str, Any]:
    """Heuristically reconstruct a fields dict when the event fields are
    emitted as a list of positional 'value' entries without explicit keys.

    We look for sentinel event names like 'gen_ai.content.prompt', the next
    JSON-like string is treated as the content. Supports completion and
    tool.call similarly.
    """
    out: Dict[str, Any] = {}
    if not isinstance(values, list):
        return out

    def _looks_json_str(s: Any) -> bool:
        if not isinstance(s, str):
            return False
        st = s.strip()
        return st.startswith("{") or st.startswith("[")

    # Map event name if present
    for i, val in enumerate(values):
        if not isinstance(val, str):
            continue
        if val in {
            "gen_ai.content.prompt",
            "gen_ai.content.message",
            "gen_ai.content.input",
        }:
            out["event.name"] = val
            # find the next json-ish value as content
            for j in range(i + 1, len(values)):
                if _looks_json_str(values[j]) or isinstance(values[j], str):
                    out["gen_ai.event.content"] = values[j]
                    break
            break
        if val in {
            "gen_ai.content.completion",
            "gen_ai.content.response",
            "gen_ai.content.output",
        }:
            out["event.name"] = val
            for j in range(i + 1, len(values)):
                if _looks_json_str(values[j]) or isinstance(values[j], str):
                    out["gen_ai.event.content"] = values[j]
                    break
            break
        if val in {"gen_ai.tool.call"}:
            out["event.name"] = val
            for j in range(i + 1, len(values)):
                if _looks_json_str(values[j]) or isinstance(values[j], str):
                    out["gen_ai.tool_calls"] = values[j]
                    break
            break
    return out


def extract_messages_from_span(span: Json) -> Dict[str, List[Dict[str, str]]]:
    """Extract user/assistant/system messages from a span.

    Returns a dict with possible keys: "user", "assistant", "system" mapping to
    a list of {"role", "content"} messages.
    """
    messages: Dict[str, List[Dict[str, str]]] = {
        "user": [],
        "assistant": [],
        "system": [],
    }

    def _normalize_role(role: Optional[str]) -> str:
        if not role:
            return "user"
        r = str(role).strip().lower()
        if r in {"user", "human", "customer", "client", "end_user"}:
            return "user"
        if r in {"assistant", "ai", "model", "bot"}:
            return "assistant"
        if r in {"system", "system_message", "context"}:
            return "system"
        # Default to user for unknown roles to keep convo evaluable
        return "user"

    attrs = span.get("attributes") or span.get("tags") or {}
    # Normalize attrs when provided as a list of {key,value}
    if isinstance(attrs, list):
        attrs = _fields_list_to_dict(attrs)

    # GenAI semconv primary sources (attributes/tags)
    prompt = (
        attrs.get("gen_ai.prompt")
        or attrs.get("gen_ai.input_messages")
        or attrs.get("llm.input_messages")
        or attrs.get("input")
        or attrs.get("messages")
    )
    completion = (
        attrs.get("gen_ai.completion")
        or attrs.get("gen_ai.output_messages")
        or attrs.get("llm.output_messages")
        or attrs.get("output")
        or attrs.get("response")
    )

    if prompt:
        # prompt can be a string or a list of messages
        p = _safe_json_loads(prompt)
        if isinstance(p, str):
            messages["user"].append({"role": "user", "content": p})
        elif isinstance(p, list):
            for m in p:
                if isinstance(m, dict):
                    role = _normalize_role(
                        m.get("role") or m.get("type") or "user"
                    )
                    content = m.get("content") or m.get("text") or ""
                    if content:
                        messages.setdefault(role, []).append(
                            {"role": role, "content": content}
                        )
        elif isinstance(p, dict):
            # Sometimes nested structures
            if p.get("messages") and isinstance(p["messages"], list):
                for m in p["messages"]:
                    if isinstance(m, dict):
                        role = _normalize_role(m.get("role") or "user")
                        content = m.get("content") or m.get("text") or ""
                        if content:
                            messages.setdefault(role, []).append(
                                {"role": role, "content": content}
                            )

    if completion:
        c = _safe_json_loads(completion)
        if isinstance(c, str):
            messages["assistant"].append({"role": "assistant", "content": c})
        elif isinstance(c, list):
            for m in c:
                if isinstance(m, dict):
                    role = _normalize_role(
                        m.get("role") or m.get("type") or "assistant"
                    )
                    content = m.get("content") or m.get("text") or ""
                    if content:
                        messages.setdefault(role, []).append(
                            {"role": role, "content": content}
                        )
        elif isinstance(c, dict):
            # OpenAI-like response with choices
            if isinstance(c.get("choices"), list) and c["choices"]:
                ch = c["choices"][0]
                if isinstance(ch, dict):
                    msg = ch.get("message") or {}
                    content = (
                        msg.get("content")
                        if isinstance(msg, dict)
                        else None
                    )
                    if isinstance(content, str) and content:
                        messages.setdefault("assistant", []).append(
                            {"role": "assistant", "content": content}
                        )

    # Fall back to span events (logs)
    for ev in span.get("events", []) or span.get("logs", []) or []:
        name = ev.get("name") or ev.get("event")
        raw_fields = ev.get("attributes") or ev.get("fields")
        fields = _fields_list_to_dict(raw_fields)
        # If we couldn't parse any keys, try positional heuristic
        if not fields and isinstance(raw_fields, list):
            vals = []
            for it in raw_fields:
                if isinstance(it, dict) and "value" in it:
                    vals.append(it.get("value"))
                elif not isinstance(it, dict):
                    vals.append(it)
            fields = _infer_fields_from_values_list(vals)
        if not fields:
            continue

        # Some pipelines put the event type under 'event.name'
        event_name = fields.get("event.name") or fields.get("name")

    # Common patterns: messages captured under gen_ai.event.content
    # or generic message fields
        # 1) GenAI semconv names directly in fields
        if fields.get("gen_ai.event.name") in {"prompt", "message", "input"}:
            content = (
                fields.get("gen_ai.event.content")
                or fields.get("gen_ai.prompt")
                or fields.get("gen_ai.input_messages")
                or fields.get("llm.input_messages")
                or fields.get("message")
                or fields.get("content")
            )
            role = _normalize_role(
                fields.get("gen_ai.event.role") or fields.get("role") or "user"
            )
            content = _safe_json_loads(content)
            if isinstance(content, str) and content:
                messages.setdefault(role, []).append(
                    {"role": role, "content": content}
                )
        # 2) Our converter uses event.name like 'gen_ai.content.prompt'
        elif event_name in {
            "gen_ai.content.prompt",
            "gen_ai.content.message",
            "gen_ai.content.input",
        }:
            content = (
                fields.get("gen_ai.prompt")
                or fields.get("gen_ai.event.content")
                or fields.get("gen_ai.input_messages")
                or fields.get("llm.input_messages")
                or fields.get("message")
                or fields.get("content")
            )
            role = _normalize_role(
                fields.get("gen_ai.event.role")
                or fields.get("role")
                or "user"
            )
            content = _safe_json_loads(content)
            if isinstance(content, str) and content:
                messages.setdefault(role, []).append(
                    {"role": role, "content": content}
                )
            elif isinstance(content, dict):
                c_role = _normalize_role(content.get("role") or role)
                c_text = content.get("content") or content.get("text") or ""
                if c_text:
                    messages.setdefault(c_role, []).append(
                        {"role": c_role, "content": c_text}
                    )
        elif name in {"prompt", "message", "input"}:
            content = fields.get("message") or fields.get("content")
            role = _normalize_role(fields.get("role") or "user")
            content = _safe_json_loads(content)
            if isinstance(content, str) and content:
                messages.setdefault(role, []).append(
                    {"role": role, "content": content}
                )

        if fields.get("gen_ai.event.name") in {
            "completion",
            "response",
            "output",
        }:
            content = (
                fields.get("gen_ai.event.content")
                or fields.get("gen_ai.completion")
                or fields.get("gen_ai.output_messages")
                or fields.get("llm.output_messages")
                or fields.get("message")
                or fields.get("content")
            )
            role = _normalize_role(
                fields.get("gen_ai.event.role")
                or fields.get("role")
                or "assistant"
            )
            content = _safe_json_loads(content)
            if isinstance(content, str) and content:
                messages.setdefault(role, []).append(
                    {"role": role, "content": content}
                )
        # 2) Our converter uses event.name like 'gen_ai.content.completion'
        elif event_name in {
            "gen_ai.content.completion",
            "gen_ai.content.response",
            "gen_ai.content.output",
        }:
            content = (
                fields.get("gen_ai.event.content")
                or fields.get("gen_ai.completion")
                or fields.get("gen_ai.output_messages")
                or fields.get("llm.output_messages")
                or fields.get("message")
                or fields.get("content")
            )
            role = _normalize_role(
                fields.get("gen_ai.event.role")
                or fields.get("role")
                or "assistant"
            )
            content = _safe_json_loads(content)
            if isinstance(content, str) and content:
                messages.setdefault(role, []).append(
                    {"role": role, "content": content}
                )
            elif isinstance(content, dict):
                # Handle OpenAI-like {role, content}
                c_role = _normalize_role(content.get("role") or role)
                c_text = content.get("content") or content.get("text") or ""
                if c_text:
                    messages.setdefault(c_role, []).append(
                        {"role": c_role, "content": c_text}
                    )
        elif name in {"completion", "response", "output"}:
            content = fields.get("message") or fields.get("content")
            role = _normalize_role(fields.get("role") or "assistant")
            content = _safe_json_loads(content)
            if isinstance(content, str) and content:
                messages.setdefault(role, []).append(
                    {"role": role, "content": content}
                )

        # 3) Generic fallback: if fields expose prompt/completion directly
        # without an event name, extract conservatively.
        if not any(messages.get(r) for r in ("user", "assistant")):
            # User-ish content
            user_src = (
                fields.get("gen_ai.prompt")
                or fields.get("gen_ai.input_messages")
                or fields.get("llm.input_messages")
                or fields.get("input")
                or None
            )
            if user_src:
                u = _safe_json_loads(user_src)
                if isinstance(u, str) and u:
                    messages.setdefault("user", []).append(
                        {"role": "user", "content": u}
                    )
                elif isinstance(u, list):
                    for m in u:
                        if isinstance(m, dict):
                            role = _normalize_role(m.get("role") or "user")
                            content = (
                                m.get("content") or m.get("text") or ""
                            )
                            if content:
                                messages.setdefault(role, []).append(
                                    {"role": role, "content": content}
                                )
            # Assistant-ish content
            asrc = (
                fields.get("gen_ai.completion")
                or fields.get("gen_ai.output_messages")
                or fields.get("llm.output_messages")
                or fields.get("output")
                or fields.get("response")
                or None
            )
            if asrc:
                a = _safe_json_loads(asrc)
                if isinstance(a, str) and a:
                    messages.setdefault("assistant", []).append(
                        {"role": "assistant", "content": a}
                    )
                elif isinstance(a, list):
                    for m in a:
                        if isinstance(m, dict):
                            role = _normalize_role(
                                m.get("role") or "assistant"
                            )
                            content = (
                                m.get("content") or m.get("text") or ""
                            )
                            if content:
                                messages.setdefault(role, []).append(
                                    {"role": role, "content": content}
                                )

    # Consolidate roles to canonical keys and update inner role fields
    normalized: Dict[str, List[Dict[str, str]]] = {
        "user": [],
        "assistant": [],
        "system": [],
    }
    for r, lst in messages.items():
        if not lst:
            continue
        nr = _normalize_role(r)
        for m in lst:
            if isinstance(m, dict):
                m["role"] = nr
                normalized.setdefault(nr, []).append(m)

    # Remove empty roles
    return {k: v for k, v in normalized.items() if v}


def extract_tool_calls_from_span(span: Json) -> List[Dict[str, Any]]:
    """Extract tool calls from a span.

    Returns a list of tool call dicts with fields like name, arguments, result.
    """
    tools: List[Dict[str, Any]] = []
    attrs = span.get("attributes") or span.get("tags") or {}
    if isinstance(attrs, list):
        attrs = _fields_list_to_dict(attrs)

    # Primary: gen_ai.tool_calls often a JSON string or list
    raw_calls = (
        attrs.get("gen_ai.tool_calls")
        or attrs.get("llm.tools")
        or attrs.get("gen_ai.tools")
    )
    if raw_calls:
        val = _safe_json_loads(raw_calls)
        if isinstance(val, list):
            for tc in val:
                if not isinstance(tc, dict):
                    continue
                name = tc.get("name") or tc.get("function", {}).get("name")
                args = (
                    tc.get("arguments")
                    or tc.get("function", {}).get("arguments")
                )
                res = tc.get("result") or tc.get("output")
                tools.append({"name": name, "arguments": args, "result": res})
        elif isinstance(val, dict):
            # Sometimes a single call dict
            name = val.get("name") or val.get("function", {}).get("name")
            args = (
                val.get("arguments")
                or val.get("function", {}).get("arguments")
            )
            res = val.get("result") or val.get("output")
            tools.append({"name": name, "arguments": args, "result": res})

    # Fallback: events/logs with tool information
    for ev in span.get("events", []) or span.get("logs", []) or []:
        raw_fields = ev.get("attributes") or ev.get("fields")
        fields = _fields_list_to_dict(raw_fields)
        if not fields and isinstance(raw_fields, list):
            vals = []
            for it in raw_fields:
                if isinstance(it, dict) and "value" in it:
                    vals.append(it.get("value"))
                elif not isinstance(it, dict):
                    vals.append(it)
            fields = _infer_fields_from_values_list(vals)
        if not fields:
            continue
        event_name = fields.get("event.name") or fields.get("name")
        if fields.get("gen_ai.event.name") in {
            "tool_call",
            "tool",
            "function_call",
        } or event_name in {"gen_ai.tool.call"}:
            name = (
                fields.get("gen_ai.tool.name")
                or fields.get("tool.name")
                or fields.get("function.name")
            )
            args = (
                fields.get("gen_ai.tool.arguments")
                or fields.get("tool.arguments")
                or fields.get("function.arguments")
            )
            res = (
                fields.get("gen_ai.tool.result")
                or fields.get("tool.result")
                or fields.get("function.result")
            )
            if not name and fields.get("gen_ai.tool_calls"):
                # Try to parse a structured list under gen_ai.tool_calls
                tc_val = _safe_json_loads(fields.get("gen_ai.tool_calls"))
                if isinstance(tc_val, list):
                    for tc in tc_val:
                        if isinstance(tc, dict):
                            tools.append(
                                {
                                    "name": tc.get("name")
                                    or tc.get("function", {}).get("name"),
                                    "arguments": tc.get("arguments")
                                    or tc.get("function", {}).get(
                                        "arguments"
                                    ),
                                    "result": tc.get("result")
                                    or tc.get("output"),
                                }
                            )
                    continue
            tools.append({"name": name, "arguments": args, "result": res})

        # Generic fallbacks
        if fields.get("tool_name") or fields.get("function_name"):
            name = fields.get("tool_name") or fields.get("function_name")
            args = fields.get("tool_args") or fields.get("function_args")
            res = fields.get("tool_result") or fields.get("function_result")
            tools.append({"name": name, "arguments": args, "result": res})

    # Simple tag-only fallback for spans that only mark a tool by name
    if not tools and attrs.get("gen_ai.tool.name"):
        tools.append({
            "name": attrs.get("gen_ai.tool.name"),
            "arguments": attrs.get("gen_ai.tool.arguments"),
            "result": attrs.get("gen_ai.tool.result"),
        })

    return tools


def last_assistant_message(
    messages: Dict[str, List[Dict[str, str]]]
) -> Optional[str]:
    """Return the content of the last assistant message if present."""
    assistants = messages.get("assistant") or []
    if not assistants:
        return None
    return assistants[-1].get("content")


if __name__ == "__main__":
    # Lightweight CLI to extract messages and tool calls from a trace
    import argparse
    import json
    import os
    from datetime import datetime, timedelta, timezone
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    log = logging.getLogger("span_extraction_cli")

    parser = argparse.ArgumentParser(
        description=(
            "Extract messages and tool calls from OTEL GenAI spans. "
            "Use --input-json to load a local OTEL JSON, or --trace-id to "
            "pull from Application Insights."
        )
    )
    parser.add_argument("--input-json", help="Path to OTEL JSON file")
    parser.add_argument("--trace-id", help="Trace ID in App Insights")
    parser.add_argument(
        "--time-range-hours",
        type=int,
        default=48,
        help="Time range to search for the trace (hours)",
    )
    parser.add_argument(
        "--output",
        help="Write extracted summary JSON to this path",
    )
    args = parser.parse_args()

    otel_data: Dict[str, Any] = {}

    if args.input_json:
        log.info("Loading OTEL JSON from %s", args.input_json)
        try:
            with open(args.input_json, "r") as f:
                otel_data = json.load(f)
            if not args.trace_id:
                try:
                    args.trace_id = otel_data["data"][0]["traceID"]
                except Exception:
                    pass
        except Exception as e:
            log.error("Failed to read input JSON: %s", e)
            raise SystemExit(1)
    else:
        if not args.trace_id:
            log.error("--trace-id is required when not using --input-json")
            raise SystemExit(2)

        # Lazy import to avoid hard dependency when only using --input-json
        try:
            from otel_gen_ai_hydrator.sources.application_insights import (
                ApplicationInsightsConnector,
                ApplicationInsightsConfig,
            )
            try:
                # Prefer relative import to reuse the same conversion code
                from .trace_to_json import convert_to_otel_compliant_format
            except Exception:
                # Fallback for direct script runs
                from trace_to_json import convert_to_otel_compliant_format
        except Exception as e:
            log.error("Missing AI dependencies for App Insights: %s", e)
            raise SystemExit(3)

        resource_id = os.environ.get("APPLICATION_INSIGHTS_RESOURCE_ID")
        if not resource_id:
            log.error("APPLICATION_INSIGHTS_RESOURCE_ID is not set in env")
            raise SystemExit(4)

        config = ApplicationInsightsConfig(resource_id=resource_id)
        connector = ApplicationInsightsConnector(config)

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=args.time_range_hours)

        log.info("Querying App Insights for trace %s", args.trace_id)
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

        try:
            records = connector._execute_query(  # noqa: SLF001
                kusto_query,
                timespan=timedelta(hours=args.time_range_hours),
            )
        except Exception as e:  # pragma: no cover - live dependency
            log.error("Failed to query App Insights: %s", e)
            raise SystemExit(5)

        if not records:
            log.warning("No records found for the trace. Exiting.")
            raise SystemExit(0)

        otel_data = convert_to_otel_compliant_format(records, args.trace_id)

    # Summarize spans
    try:
        spans = otel_data["data"][0]["spans"]
    except Exception:
        log.error("Input is not an OTEL JSON with data[0].spans")
        raise SystemExit(6)

    summary: List[Dict[str, Any]] = []
    for sp in spans:
        msgs = extract_messages_from_span(sp)
        tools = extract_tool_calls_from_span(sp)
        summary.append(
            {
                "span_id": sp.get("spanID"),
                "operation": sp.get("operationName"),
                "messages": msgs,
                "tool_calls": tools,
            }
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {"trace_id": args.trace_id, "summary": summary}, f, indent=2
            )
        log.info("Wrote extraction summary to %s", args.output)
    else:
        # Print a concise human-readable summary
        print("Spans:")
        for item in summary:
            print("-", item["span_id"], ":", item["operation"])
            ucnt = len(item["messages"].get("user", []))
            acnt = len(item["messages"].get("assistant", []))
            tcnt = len(item["tool_calls"] or [])
            print(
                "  messages:", ucnt, "user,", acnt, "assistant;",
                "tools:", tcnt,
            )
