"""
Script to query OpenTelemetry traces from Jaeger, evaluate them, and store results as linked spans.
Works directly with OTEL traces for both LangGraph and single-agent systems.
"""

import json
import os
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import hashlib
import uuid

from dotenv import load_dotenv

# Azure AI Evaluation imports
from azure.ai.evaluation import (
    AzureOpenAIModelConfiguration,
    ToolCallAccuracyEvaluator,
    IntentResolutionEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
)

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind, Link
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JaegerTraceExtractor:
    """Extract traces from Jaeger API."""
    
    def __init__(self, jaeger_url: str = "http://localhost:16686"):
        self.jaeger_url = jaeger_url
        self.api_base = f"{jaeger_url}/api"
        
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get a single trace by ID."""
        url = f"{self.api_base}/traces/{trace_id}"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get trace {trace_id}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching trace: {e}")
            return None
    
    def search_traces(self, service: str = None, operation: str = None, 
                     tags: Dict[str, str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for traces with filters."""
        url = f"{self.api_base}/traces"
        params = {"limit": limit}
        
        if service:
            params["service"] = service
        if operation:
            params["operation"] = operation
        if tags:
            for key, value in tags.items():
                params[f"tags[{key}]"] = value
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json().get("data", [])
            else:
                logger.error(f"Failed to search traces: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error searching traces: {e}")
            return []


class OTELSpanDataExtractor:
    """Extract data from OTEL spans for evaluation."""
    
    @staticmethod
    def extract_messages_from_span(span: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Extract messages from OTEL span logs and tags."""
        messages = {
            "system": [],
            "user": [],
            "assistant": [],
            "tool": []
        }
        
        # Debug: print span structure
        # logger.debug(f"Span operation: {span.get('operationName')}")
        # logger.debug(f"Span tags: {[tag['key'] for tag in span.get('tags', [])]}")
        # logger.debug(f"Span logs: {len(span.get('logs', []))}")
        
        # Extract from logs
        for log in span.get("logs", []):
            fields = {f["key"]: f["value"] for f in log.get("fields", [])}
            
            # Check for GenAI events - note the field is just "event" not "event.name"
            event_name = fields.get("event", "")
            
            if event_name == "gen_ai.content.prompt":
                prompt_data = fields.get("gen_ai.prompt", "")
                if prompt_data:
                    try:
                        prompt = json.loads(prompt_data) if isinstance(prompt_data, str) else prompt_data
                        role = prompt.get("role", "")
                        content = prompt.get("content", "")
                        
                        # Parse content that contains multiple messages
                        if "System:" in content and "Human:" in content:
                            parts = content.split("\n")
                            current_role = None
                            current_content = []
                            
                            for part in parts:
                                if part.startswith("System:"):
                                    if current_role and current_content:
                                        messages[current_role].append({"role": current_role, "content": "\n".join(current_content).strip()})
                                    current_role = "system"
                                    current_content = [part.replace("System:", "").strip()]
                                elif part.startswith("Human:"):
                                    if current_role and current_content:
                                        messages[current_role].append({"role": current_role, "content": "\n".join(current_content).strip()})
                                    current_role = "user"
                                    current_content = [part.replace("Human:", "").strip()]
                                elif part.startswith("AI:"):
                                    if current_role and current_content:
                                        messages[current_role].append({"role": current_role, "content": "\n".join(current_content).strip()})
                                    current_role = "assistant"
                                    current_content = [part.replace("AI:", "").strip()]
                                elif current_content is not None:
                                    current_content.append(part)
                            
                            if current_role and current_content:
                                messages[current_role].append({"role": current_role, "content": "\n".join(current_content).strip()})
                        else:
                            # Simple format
                            if role == "system":
                                messages["system"].append({"role": role, "content": content})
                            elif role in ["user", "human"]:
                                messages["user"].append({"role": "user", "content": content})
                    except Exception as e:
                        logger.debug(f"Failed to parse prompt: {e}")
                        
            elif event_name == "gen_ai.content.completion":
                event_content = fields.get("gen_ai.event.content", "")
                if event_content:
                    try:
                        content_data = json.loads(event_content) if isinstance(event_content, str) else event_content
                        content = content_data.get("content", "")
                        if content:
                            messages["assistant"].append({"role": "assistant", "content": content})
                    except:
                        pass
        
        # Check tags for additional data
        tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
        
        # Check for final plan in tags
        if "travel.plan.final" in tags:
            messages["assistant"].append({"role": "assistant", "content": tags["travel.plan.final"]})
        
        # Check chain.input.messages for conversation history
        if "chain.input.messages" in tags:
            try:
                input_messages = json.loads(tags["chain.input.messages"])
                if isinstance(input_messages, list):
                    for msg in input_messages:
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            if role and content:
                                if role == "ai":
                                    messages["assistant"].append({"role": "assistant", "content": content})
                                elif role == "human":
                                    messages["user"].append({"role": "user", "content": content})
                        elif isinstance(msg, list) and len(msg) == 2:
                            # Handle [role, content] format
                            role, content = msg[0], msg[1]
                            if role == "ai":
                                messages["assistant"].append({"role": "assistant", "content": content})
                            elif role == "human":
                                messages["user"].append({"role": "user", "content": content})
            except Exception as e:
                logger.debug(f"Failed to parse chain.input.messages: {e}")
        
        return messages
    
    @staticmethod
    def extract_tool_calls_from_span(span: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from OTEL span."""
        tool_calls = []
        
        # Check if this is a tool execution span
        operation_name = span.get("operationName", "")
        if operation_name.startswith("execute_tool"):
            # Extract tool name from operation name
            tool_name = operation_name.replace("execute_tool", "").strip()
            
            # Get tool details from tags
            tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
            
            tool_call = {
                "name": tags.get("gen_ai.tool.name", tool_name),
                "id": tags.get("gen_ai.tool.invocation.id", ""),
                "arguments": tags.get("gen_ai.tool.invocation.arguments", ""),
                "type": "execution"
            }
            tool_calls.append(tool_call)
        
        # Check logs for tool calls
        for log in span.get("logs", []):
            fields = {f["key"]: f["value"] for f in log.get("fields", [])}
            
            # Check for tool_calls field
            if "tool_calls" in fields:
                try:
                    calls = json.loads(fields["tool_calls"]) if isinstance(fields["tool_calls"], str) else fields["tool_calls"]
                    if isinstance(calls, list):
                        tool_calls.extend(calls)
                except:
                    pass
            
            # Check for gen_ai.tool.call event
            if fields.get("event") == "gen_ai.tool.call":
                tool_call = {
                    "name": fields.get("tool_name", ""),
                    "id": fields.get("tool_id", ""),
                    "type": "call"
                }
                if tool_call["name"]:
                    tool_calls.append(tool_call)
        
        return tool_calls
    
    @staticmethod
    def find_final_response_content(span: Dict[str, Any]) -> Optional[str]:
        """Find final travel plan content in span."""
        # Check tags first
        tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
        
        if "travel.plan.final" in tags:
            return tags["travel.plan.final"]
        
        # Check chain.outputs for final plan
        if "chain.outputs" in tags:
            try:
                outputs = json.loads(tags["chain.outputs"])
                if isinstance(outputs, dict) and "messages" in outputs:
                    # Look through messages for final plan
                    for msg in outputs["messages"]:
                        if "FINAL TRAVEL PLAN" in str(msg):
                            # Extract the content
                            if "content=" in msg:
                                content_start = msg.find("content='") + 9
                                content_end = msg.find("' additional_kwargs")
                                if content_end > content_start:
                                    return msg[content_start:content_end]
            except:
                pass
        
        # Then check messages
        messages = OTELSpanDataExtractor.extract_messages_from_span(span)
        
        for msg in messages["assistant"]:
            content = msg.get("content", "")
            if any(keyword in content.upper() for keyword in ["FINAL TRAVEL PLAN", "FINAL PLAN", "TRAVEL ITINERARY"]):
                return content
        
        return None


class OTELTraceEvaluator:
    """Evaluate OTEL traces and create evaluation spans."""
    
    def __init__(self, jaeger_url: str = "http://localhost:16686", 
                 otlp_endpoint: str = "http://localhost:4318/v1/traces",
                 debug: bool = False):
        self.extractor = JaegerTraceExtractor(jaeger_url)
        self.debug = debug
        
        # Set debug logging if requested
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Azure OpenAI configuration
        self.model_config = AzureOpenAIModelConfiguration(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        )
        
        # Initialize evaluators
        self.tool_accuracy_eval = ToolCallAccuracyEvaluator(model_config=self.model_config)
        self.intent_resolution_eval = IntentResolutionEvaluator(model_config=self.model_config)
        self.fluency_eval = FluencyEvaluator(model_config=self.model_config)
        self.groundedness_eval = GroundednessEvaluator(model_config=self.model_config)
        self.relevance_eval = RelevanceEvaluator(model_config=self.model_config)
        
        # Setup OpenTelemetry for evaluation spans
        resource = Resource.create({
            SERVICE_NAME: "otel-trace-evaluator",
            "service.version": "1.0.0",
            "deployment.environment": "evaluation"
        })
        
        provider = TracerProvider(resource=resource)
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        trace.set_tracer_provider(provider)
        
        self.tracer = trace.get_tracer(__name__)
        
    def evaluate_trace(self, trace_id: str) -> Dict[str, Any]:
        """Evaluate a trace and create evaluation spans."""
        logger.info(f"Evaluating trace: {trace_id}")
        
        # Get trace from Jaeger
        trace_data = self.extractor.get_trace(trace_id)
        if not trace_data:
            logger.error(f"Could not fetch trace {trace_id}")
            return {}
        
        # Extract spans
        spans = []
        if "data" in trace_data:
            for trace_item in trace_data["data"]:
                spans.extend(trace_item.get("spans", []))
        
        logger.info(f"Found {len(spans)} spans in trace")
        
        # Debug: print first few spans
        # if self.debug:
        #     for i, span in enumerate(spans[:5]):
        #         logger.debug(f"\nSpan {i}: {span.get('operationName')}")
        #         tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
        #         logger.debug(f"Tags: {list(tags.keys())}")
        #         if span.get("logs"):
        #             logger.debug(f"Logs: {len(span['logs'])}")
        
        # Categorize spans
        categorized = self._categorize_spans(spans)
        
        # Perform evaluations
        evaluation_results = {
            "trace_id": trace_id,
            "span_count": len(spans),
            "evaluations": {}
        }
        
        # Create root evaluation span
        with self.tracer.start_as_current_span(
            f"evaluate_trace:{trace_id}",
            kind=SpanKind.CLIENT,
            attributes={
                "evaluation.trace_id": trace_id,
                "evaluation.span_count": len(spans),
                "evaluation.timestamp": datetime.now(timezone.utc).isoformat()
            }
        ) as eval_root_span:
            
            # 1. Evaluate Intent Resolution
            if categorized["llm_spans"]:
                intent_results = self._evaluate_intent_resolution(
                    categorized["llm_spans"], trace_id, eval_root_span
                )
                evaluation_results["evaluations"]["intent_resolution"] = intent_results
            
            # 2. Evaluate Tool Usage
            if categorized["tool_execution_spans"]:
                tool_results = self._evaluate_tool_usage(
                    categorized["tool_execution_spans"], trace_id, eval_root_span
                )
                evaluation_results["evaluations"]["tool_usage"] = tool_results
            
            # 3. Evaluate Response Quality
            if categorized["llm_spans"]:
                quality_results = self._evaluate_response_quality(
                    categorized["llm_spans"], trace_id, eval_root_span
                )
                evaluation_results["evaluations"]["response_quality"] = quality_results
            
            # 4. Evaluate Final Plan Completeness
            if categorized["final_plan_spans"]:
                plan_results = self._evaluate_plan_completeness(
                    categorized["final_plan_spans"], trace_id, eval_root_span
                )
                evaluation_results["evaluations"]["plan_completeness"] = plan_results
            
            # Add summary
            summary = self._create_evaluation_summary(evaluation_results["evaluations"])
            evaluation_results["summary"] = summary
            
            # Add summary to root span
            eval_root_span.set_attribute("evaluation.total_evaluations", summary["total_evaluations"])
            eval_root_span.set_attribute("evaluation.average_score", summary["average_score"])
            eval_root_span.set_attribute("evaluation.percentage", summary["percentage"])
        
        return evaluation_results
    
    def _categorize_spans(self, spans: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize spans by type."""
        categorized = {
            "llm_spans": [],
            "tool_spans": [],
            "final_plan_spans": [],
            "tool_execution_spans": []
        }
        
        for span in spans:
            operation_name = span.get("operationName", "")
            tags = {tag["key"]: tag["value"] for tag in span.get("tags", [])}
            
            # Debug
            # if self.debug:
            #     logger.debug(f"Checking span: {operation_name}")
            
            # Check for LLM spans - exact match from trace
            if "chat.completions" in operation_name:
                categorized["llm_spans"].append(span)
                
                # Check if it has tool calls in logs
                for log in span.get("logs", []):
                    fields = {f["key"]: f["value"] for f in log.get("fields", [])}
                    if "tool_calls" in fields:
                        categorized["tool_spans"].append(span)
                        break
            
            # Check for tool execution spans - exact match from trace
            if operation_name.startswith("execute_tool"):
                categorized["tool_execution_spans"].append(span)
            
            # Check for final plan spans
            if "travel.plan.final" in tags:
                categorized["final_plan_spans"].append(span)
            elif "chain.outputs" in tags and "FINAL TRAVEL PLAN" in tags["chain.outputs"]:
                categorized["final_plan_spans"].append(span)
        
        # Remove duplicates
        for key in categorized:
            seen = set()
            unique = []
            for span in categorized[key]:
                span_id = span["spanID"]
                if span_id not in seen:
                    seen.add(span_id)
                    unique.append(span)
            categorized[key] = unique
        
        logger.info(f"Categorized spans - LLM: {len(categorized['llm_spans'])}, "
                   f"Tool calls in LLM: {len(categorized['tool_spans'])}, "
                   f"Tool Execution: {len(categorized['tool_execution_spans'])}, "
                   f"Final: {len(categorized['final_plan_spans'])}")
        
        return categorized
    
    def _evaluate_intent_resolution(self, llm_spans: List[Dict[str, Any]], 
                                  trace_id: str, parent_span) -> List[Dict[str, Any]]:
        """Evaluate intent resolution for LLM spans."""
        results = []
        
        # Limit evaluations but ensure we get some
        spans_to_eval = llm_spans[:3] if llm_spans else []
        
        for span in spans_to_eval:
            span_id = span["spanID"]
            
            # Generate a unique evaluation span ID
            eval_span_id = str(uuid.uuid4()).replace("-", "")[:16]
            
            # Create evaluation span linked to original
            with self.tracer.start_as_current_span(
                f"evaluate:intent_resolution:{span_id[:8]}",
                kind=SpanKind.CLIENT,
                attributes={
                    "evaluation.type": "intent_resolution",
                    "evaluation.original_span_id": span_id,
                    "evaluation.original_trace_id": trace_id
                }
            ) as eval_span:
                
                try:
                    messages = OTELSpanDataExtractor.extract_messages_from_span(span)
                    
                    # Build conversation
                    conversation = []
                    
                    # Add messages in order
                    if messages["system"]:
                        conversation.append({"role": "system", "content": messages["system"][0]["content"]})
                    
                    if messages["user"]:
                        conversation.append({"role": "user", "content": messages["user"][0]["content"]})
                    
                    if messages["assistant"]:
                        conversation.append({"role": "assistant", "content": messages["assistant"][0]["content"]})
                    
                    # Need at least 2 messages for evaluation
                    if len(conversation) >= 2:
                        # IntentResolutionEvaluator expects conversation in a specific format
                        conversation_input = {"messages": conversation}
                        result = self.intent_resolution_eval(conversation=conversation_input)
                        score = float(result.get("intent_resolution_score", 0.0)) * 5
                        reasoning = result.get("intent_resolution_reason", "")
                        
                        eval_span.set_attribute("evaluation.score", score)
                        eval_span.set_attribute("evaluation.reasoning", reasoning[:1000])  # Limit length
                        eval_span.set_attribute("evaluation.result", "pass" if score >= 3.0 else "fail")
                        
                        results.append({
                            "span_id": span_id,
                            "score": score,
                            "reasoning": reasoning,
                            "evaluation_span_id": eval_span_id
                        })
                    else:
                        logger.debug(f"Not enough messages for intent evaluation in span {span_id[:8]}")
                        
                except Exception as e:
                    logger.error(f"Intent evaluation error: {e}", exc_info=True)
                    eval_span.set_status(Status(StatusCode.ERROR, str(e)))
                    eval_span.record_exception(e)
        
        return results
    
    def _evaluate_tool_usage(self, tool_execution_spans: List[Dict[str, Any]], 
                           trace_id: str, parent_span) -> List[Dict[str, Any]]:
        """Evaluate tool usage accuracy."""
        results = []
        
        # Group by unique tools
        unique_tools = {}
        for span in tool_execution_spans:
            tool_calls = OTELSpanDataExtractor.extract_tool_calls_from_span(span)
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                if tool_name and tool_name not in unique_tools:
                    unique_tools[tool_name] = span
        
        # Evaluate up to 5 unique tools
        for tool_name, span in list(unique_tools.items())[:5]:
            span_id = span["spanID"]
            eval_span_id = str(uuid.uuid4()).replace("-", "")[:16]
            
            with self.tracer.start_as_current_span(
                f"evaluate:tool_usage:{tool_name}",
                kind=SpanKind.CLIENT,
                attributes={
                    "evaluation.type": "tool_usage",
                    "evaluation.original_span_id": span_id,
                    "evaluation.original_trace_id": trace_id,
                    "evaluation.tool_name": tool_name
                }
            ) as eval_span:
                
                try:
                    # Evaluate tool relevance for travel planning
                    travel_tools = [
                        'search_flights', 'search_hotels', 'book_flight', 'book_hotel',
                        'get_weather', 'search_activities', 'calculate_cost', 'get_schedule',
                        'get_match_schedule', 'premier_league', 'match_schedule'
                    ]
                    
                    is_relevant = any(tt in tool_name.lower() for tt in travel_tools)
                    score = 5.0 if is_relevant else 2.0
                    reasoning = f"Tool '{tool_name}' is {'relevant' if is_relevant else 'not directly relevant'} to travel planning"
                    
                    eval_span.set_attribute("evaluation.score", score)
                    eval_span.set_attribute("evaluation.reasoning", reasoning)
                    eval_span.set_attribute("evaluation.tool_relevant", is_relevant)
                    
                    results.append({
                        "span_id": span_id,
                        "tool_name": tool_name,
                        "score": score,
                        "reasoning": reasoning,
                        "evaluation_span_id": eval_span_id
                    })
                    
                except Exception as e:
                    logger.error(f"Tool usage evaluation error: {e}", exc_info=True)
                    eval_span.set_status(Status(StatusCode.ERROR, str(e)))
                    eval_span.record_exception(e)
        
        return results
    
    def _evaluate_response_quality(self, spans: List[Dict[str, Any]], 
                                 trace_id: str, parent_span) -> List[Dict[str, Any]]:
        """Evaluate response quality metrics."""
        results = []
        
        # Select spans with substantial content
        spans_to_eval = []
        for span in spans:
            messages = OTELSpanDataExtractor.extract_messages_from_span(span)
            if messages["assistant"] and messages["user"]:
                # Check if content is substantial
                assistant_content = messages["assistant"][0]["content"] if messages["assistant"] else ""
                if len(assistant_content) > 50:
                    spans_to_eval.append(span)
        
        # Evaluate up to 3 spans
        for span in spans_to_eval[:3]:
            span_id = span["spanID"]
            eval_span_id = str(uuid.uuid4()).replace("-", "")[:16]
            
            with self.tracer.start_as_current_span(
                f"evaluate:response_quality:{span_id[:8]}",
                kind=SpanKind.CLIENT,
                attributes={
                    "evaluation.type": "response_quality",
                    "evaluation.original_span_id": span_id,
                    "evaluation.original_trace_id": trace_id
                }
            ) as eval_span:
                
                try:
                    messages = OTELSpanDataExtractor.extract_messages_from_span(span)
                    
                    response = messages["assistant"][0]["content"]
                    query = messages["user"][0]["content"]
                    
                    # Fluency evaluation
                    fluency_result = self.fluency_eval(response=response)
                    fluency_score = float(fluency_result.get("fluency", 0))
                    
                    # Relevance evaluation
                    relevance_result = self.relevance_eval(query=query, response=response)
                    relevance_score = float(relevance_result.get("relevance", 0))
                    
                    avg_score = (fluency_score + relevance_score) / 2
                    
                    eval_span.set_attribute("evaluation.fluency_score", fluency_score)
                    eval_span.set_attribute("evaluation.relevance_score", relevance_score)
                    eval_span.set_attribute("evaluation.average_score", avg_score)
                    
                    results.append({
                        "span_id": span_id,
                        "fluency_score": fluency_score,
                        "relevance_score": relevance_score,
                        "average_score": avg_score,
                        "evaluation_span_id": eval_span_id
                    })
                    
                except Exception as e:
                    logger.error(f"Response quality evaluation error: {e}", exc_info=True)
                    eval_span.set_status(Status(StatusCode.ERROR, str(e)))
                    eval_span.record_exception(e)
        
        return results
    
    def _evaluate_plan_completeness(self, final_spans: List[Dict[str, Any]], 
                                  trace_id: str, parent_span) -> List[Dict[str, Any]]:
        """Evaluate final plan completeness."""
        results = []
        
        for span in final_spans[:1]:  # Usually only one final plan
            span_id = span["spanID"]
            eval_span_id = str(uuid.uuid4()).replace("-", "")[:16]
            
            with self.tracer.start_as_current_span(
                f"evaluate:plan_completeness:{span_id[:8]}",
                kind=SpanKind.CLIENT,
                attributes={
                    "evaluation.type": "plan_completeness",
                    "evaluation.original_span_id": span_id,
                    "evaluation.original_trace_id": trace_id
                }
            ) as eval_span:
                
                try:
                    final_content = OTELSpanDataExtractor.find_final_response_content(span)
                    
                    if final_content and len(final_content) > 50:
                        # Check for key elements in a travel plan
                        elements = {
                            'flights': any(word in final_content.lower() for word in ['flight', 'airline', 'departure', 'arrival']),
                            'accommodation': any(word in final_content.lower() for word in ['hotel', 'accommodation', 'stay', 'lodging']),
                            'dates': any(word in final_content.lower() for word in ['date', 'august', 'september', 'day', 'week']),
                            'pricing': any(char in final_content for char in ['$', '£', '€']) or 'cost' in final_content.lower(),
                            'itinerary': any(word in final_content.lower() for word in ['day', 'itinerary', 'schedule', 'plan']),
                            'activities': any(word in final_content.lower() for word in ['activity', 'visit', 'tour', 'match', 'game'])
                        }
                        
                        elements_present = sum(elements.values())
                        score = (elements_present / len(elements)) * 5
                        
                        eval_span.set_attribute("evaluation.score", score)
                        eval_span.set_attribute("evaluation.elements_present", elements_present)
                        eval_span.set_attribute("evaluation.elements_checked", json.dumps(elements))
                        eval_span.set_attribute("evaluation.content_length", len(final_content))
                        
                        results.append({
                            "span_id": span_id,
                            "score": score,
                            "elements_present": elements_present,
                            "elements": elements,
                            "evaluation_span_id": eval_span_id
                        })
                        
                except Exception as e:
                    logger.error(f"Plan completeness evaluation error: {e}", exc_info=True)
                    eval_span.set_status(Status(StatusCode.ERROR, str(e)))
                    eval_span.record_exception(e)
        
        return results
    
    def _create_evaluation_summary(self, evaluations: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Create summary of all evaluations."""
        total_score = 0
        total_count = 0
        
        for category, results in evaluations.items():
            for result in results:
                if "score" in result:
                    total_score += result["score"]
                    total_count += 1
                elif "average_score" in result:
                    total_score += result["average_score"]
                    total_count += 1
        
        avg_score = total_score / total_count if total_count > 0 else 0
        
        return {
            "total_evaluations": total_count,
            "average_score": avg_score,
            "percentage": (avg_score / 5) * 100,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def main():
    """Main function to evaluate OTEL traces."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate OpenTelemetry traces from Jaeger')
    parser.add_argument('trace_id', help='Trace ID to evaluate')
    parser.add_argument('--jaeger-url', default='http://localhost:16686', 
                       help='Jaeger URL (default: http://localhost:16686)')
    parser.add_argument('--otlp-endpoint', default='http://localhost:4318/v1/traces',
                       help='OTLP endpoint for evaluation spans')
    parser.add_argument('--output', help='Output file for results (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = OTELTraceEvaluator(
        jaeger_url=args.jaeger_url,
        otlp_endpoint=args.otlp_endpoint,
        debug=args.debug
    )
    
    # Evaluate trace
    results = evaluator.evaluate_trace(args.trace_id)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS FOR TRACE: {args.trace_id}")
    print(f"{'='*80}")
    
    print(f"\nSpan Count: {results['span_count']}")
    
    if "summary" in results:
        summary = results["summary"]
        print(f"\nEvaluation Summary:")
        print(f"- Total Evaluations: {summary['total_evaluations']}")
        print(f"- Average Score: {summary['average_score']:.2f}/5")
        print(f"- Percentage: {summary['percentage']:.1f}%")
    
    print(f"\nDetailed Results:")
    for category, evals in results["evaluations"].items():
        if evals:
            print(f"\n{category.replace('_', ' ').upper()}:")
            for eval_result in evals:
                print(f"  - Span {eval_result.get('span_id', 'unknown')[:8]}...")
                if "tool_name" in eval_result:
                    print(f"    Tool: {eval_result['tool_name']}")
                if "score" in eval_result:
                    print(f"    Score: {eval_result['score']:.2f}/5")
                elif "average_score" in eval_result:
                    print(f"    Average Score: {eval_result['average_score']:.2f}/5")
                    if "fluency_score" in eval_result:
                        print(f"    - Fluency: {eval_result['fluency_score']:.2f}")
                    if "relevance_score" in eval_result:
                        print(f"    - Relevance: {eval_result['relevance_score']:.2f}")
                if "reasoning" in eval_result:
                    print(f"    Reasoning: {eval_result['reasoning'][:100]}...")
                if "elements_present" in eval_result:
                    print(f"    Elements Present: {eval_result['elements_present']}/6")
                if "evaluation_span_id" in eval_result:
                    print(f"    Evaluation Span: {eval_result['evaluation_span_id']}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    print(f"\n{'='*80}")
    print("Evaluation spans have been sent to OTLP endpoint.")
    print("You can view them in Jaeger under the 'otel-trace-evaluator' service.")


if __name__ == "__main__":
    main()