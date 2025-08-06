"""
Enhanced script that extracts traces, evaluates them, and exports everything including evaluation results.
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

# Import evaluation functionality
from evaluate_span_langgraph_agents import (
    ComprehensiveSpanEvaluator,
    SpanDataExtractor,
    create_evaluation_event,
    emit_evaluation_event
)

from azure.ai.evaluation import AzureOpenAIModelConfiguration

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_and_evaluate_trace(
    trace_id: str,
    output_file: str = "trace_with_evaluations.json",
    time_range_hours: int = 24,
    evaluate: bool = True
):
    """
    Extract traces, optionally evaluate them, and save everything to JSON.
    
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
        
        # Import the conversion function from trace_to_json
        from trace_to_json import convert_to_jaeger_format
        
        # Convert to Jaeger format
        jaeger_data = convert_to_jaeger_format(trace_records, trace_id)
        jaeger_data["total"] = len(jaeger_data["data"][0]["spans"])
        
        # Add evaluation results if requested
        if evaluate:
            logger.info("Starting evaluation...")
            evaluation_results = evaluate_spans(jaeger_data, trace_id, connector)
            
            # Add evaluation results to the JSON structure
            jaeger_data["evaluations"] = evaluation_results
            
            # Also add evaluation summary
            jaeger_data["evaluation_summary"] = create_evaluation_summary(evaluation_results)
        
        # Add metadata
        jaeger_data["metadata"] = {
            "extraction_time": datetime.now(timezone.utc).isoformat(),
            "trace_id": trace_id,
            "total_records": len(trace_records),
            "total_spans": len(jaeger_data["data"][0]["spans"]),
            "evaluated": evaluate
        }
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(jaeger_data, f, indent=2, default=str)
        
        logger.info(f"Successfully saved trace data to {output_file}")
        
        # Print summary
        print(f"\nExtraction and evaluation complete!")
        print(f"- Trace ID: {trace_id}")
        print(f"- Total records: {len(trace_records)}")
        print(f"- Total spans: {len(jaeger_data['data'][0]['spans'])}")
        if evaluate and "evaluation_summary" in jaeger_data:
            summary = jaeger_data["evaluation_summary"]
            print(f"- Evaluations performed: {summary['total_evaluations']}")
            print(f"- Average score: {summary['average_score']:.1f}/5 ({summary['percentage']:.0f}%)")
        print(f"- Output file: {output_file}")
        
        return jaeger_data
        
    except Exception as e:
        logger.error(f"Error in extraction/evaluation: {e}")
        raise


def evaluate_spans(jaeger_data: Dict[str, Any], trace_id: str, connector) -> Dict[str, Any]:
    """Evaluate spans and return results."""
    # Azure OpenAI model config
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    )
    
    evaluator = ComprehensiveSpanEvaluator(model_config)
    spans = jaeger_data["data"][0]["spans"]
    
    # Categorize spans
    llm_spans = []
    tool_spans = []
    final_plan_spans = []
    
    for span in spans:
        operation_name = span.get("operationName", "").lower()
        
        # Identify LLM spans
        if "chat.completions" in operation_name or "gpt" in operation_name:
            llm_spans.append(span)
            
            # Check for tool calls in logs
            has_tool_calls = False
            for log in span.get("logs", []):
                fields = {f.get("key", ""): f.get("value", "") for f in log.get("fields", [])}
                if any("tool" in str(v).lower() for v in fields.values()):
                    has_tool_calls = True
                    break
            
            if has_tool_calls:
                tool_spans.append(span)
        
        # Check for execute_tool spans
        if "execute_tool" in operation_name:
            # Find parent LLM span
            refs = span.get("references", [])
            for ref in refs:
                parent_span_id = ref.get("spanID")
                parent_span = next((s for s in spans if s["spanID"] == parent_span_id), None)
                if parent_span and parent_span not in tool_spans:
                    tool_spans.append(parent_span)
        
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
    
    # Perform evaluations with improved error handling
    
    # 1. Intent Resolution (with better conversation handling)
    for span in llm_spans[:3]:
        try:
            result = evaluate_intent_with_conversation(span, evaluator, trace_id)
            if result:
                all_results["intent_resolution"].append({
                    "span_id": span["spanID"],
                    "result": result
                })
        except Exception as e:
            logger.error(f"Intent evaluation error for span {span['spanID']}: {e}")
    
    # 2. Tool Call Accuracy (with improved tool detection)
    for span in tool_spans[:3]:
        try:
            result = evaluate_tools_improved(span, spans, evaluator, trace_id)
            if result:
                all_results["tool_accuracy"].append({
                    "span_id": span["spanID"],
                    "result": result
                })
        except Exception as e:
            logger.error(f"Tool evaluation error for span {span['spanID']}: {e}")
    
    # 3. Response Quality (without coherence to avoid errors)
    for span in llm_spans[:2] + final_plan_spans[:1]:
        try:
            results = evaluate_response_quality_improved(span, evaluator, trace_id)
            if results:
                all_results["response_quality"].append({
                    "span_id": span["spanID"],
                    "results": results
                })
        except Exception as e:
            logger.error(f"Response quality evaluation error for span {span['spanID']}: {e}")
    
    # 4. Final Plan Completeness
    for span in final_plan_spans:
        try:
            result = evaluator.evaluate_final_plan_completeness(span, trace_id)
            if result:
                all_results["plan_completeness"].append({
                    "span_id": span["spanID"],
                    "result": result
                })
        except Exception as e:
            logger.error(f"Plan completeness evaluation error for span {span['spanID']}: {e}")
    
    return all_results


def evaluate_intent_with_conversation(span: Dict[str, Any], evaluator, trace_id: str) -> Optional[Dict[str, Any]]:
    """Evaluate intent with proper conversation format."""
    span_id = span["spanID"]
    messages = SpanDataExtractor.extract_messages_from_span(span)
    
    # Build conversation in the expected format
    conversation = []
    
    # Add system message if exists
    if messages["system"]:
        conversation.append({
            "role": "system",
            "content": messages["system"][0].get("content", "")
        })
    
    # Add user message
    if messages["user"]:
        conversation.append({
            "role": "user", 
            "content": messages["user"][0].get("content", "")
        })
    
    # Add assistant response
    if messages["assistant"]:
        conversation.append({
            "role": "assistant",
            "content": messages["assistant"][-1].get("content", "")
        })
    
    if len(conversation) < 2:  # Need at least user and assistant
        return None
    
    try:
        # Try with conversation format
        result = evaluator.intent_resolution_eval(
            conversation=conversation
        )
        
        score = float(result.get("intent_resolution_score", 0.0))
        reasoning = result.get("intent_resolution_reason", "No reasoning provided")
        
        # Convert 0-1 score to 1-5 scale
        score_5_scale = score * 5
        
        emit_evaluation_event(
            create_evaluation_event(
                name="IntentResolution",
                score=score_5_scale,
                reasoning=reasoning,
                span_id=span_id,
                trace_id=trace_id
            )
        )
        
        return {
            "metric": "intent_resolution",
            "score": score_5_scale,
            "original_score": score,
            "reasoning": reasoning
        }
        
    except Exception as e:
        logger.error(f"Intent resolution evaluation failed: {e}")
        # Fallback to original method
        return evaluator.evaluate_intent_resolution(span, trace_id)


def evaluate_tools_improved(span: Dict[str, Any], all_spans: List[Dict[str, Any]], evaluator, trace_id: str) -> Optional[Dict[str, Any]]:
    """Evaluate tool usage with improved detection."""
    span_id = span["spanID"]
    
    # Find all tool execution spans that are children of this span
    tool_executions = []
    
    for s in all_spans:
        if "execute_tool" in s.get("operationName", "").lower():
            refs = s.get("references", [])
            for ref in refs:
                if ref.get("spanID") == span_id:
                    tool_name = s.get("operationName", "").replace("execute_tool", "").strip()
                    tool_executions.append({
                        "name": tool_name,
                        "span": s
                    })
                    break
    
    if not tool_executions:
        return None
    
    # Extract user query from the parent span
    messages = SpanDataExtractor.extract_messages_from_span(span)
    user_query = ""
    if messages["user"]:
        user_query = messages["user"][0].get("content", "")
    
    if not user_query:
        return None
    
    # Simple evaluation based on tool relevance
    travel_tools = ['get_match_schedule', 'search_flights', 'search_hotels', 'search_rental_cars']
    tools_used = [te["name"] for te in tool_executions]
    relevant_tools = sum(1 for tool in tools_used if any(tt in tool.lower() for tt in travel_tools))
    
    score = (relevant_tools / len(tools_used)) * 5 if tools_used else 0
    reasoning = f"Used {relevant_tools} travel-related tools out of {len(tools_used)} total: {', '.join(tools_used)}"
    
    emit_evaluation_event(
        create_evaluation_event(
            name="ToolUsageRelevance",
            score=score,
            reasoning=reasoning,
            span_id=span_id,
            trace_id=trace_id,
            metadata={
                "tools_used": ", ".join(tools_used),
                "num_tools": len(tools_used)
            }
        )
    )
    
    return {
        "metric": "tool_usage_relevance",
        "score": score,
        "reasoning": reasoning,
        "tools_evaluated": tools_used
    }


def evaluate_response_quality_improved(span: Dict[str, Any], evaluator, trace_id: str) -> List[Dict[str, Any]]:
    """Evaluate response quality without coherence to avoid errors."""
    span_id = span["spanID"]
    messages = SpanDataExtractor.extract_messages_from_span(span)
    
    response = ""
    if messages["assistant"]:
        response = messages["assistant"][-1].get("content", "")
    
    if not response:
        return []
    
    context = ""
    if messages["system"]:
        context = messages["system"][0].get("content", "")
    
    user_query = ""
    if messages["user"]:
        user_query = messages["user"][0].get("content", "")
    
    results = []
    
    # Only evaluate metrics that don't require special formats
    
    # 1. Fluency
    try:
        fluency_result = evaluator.fluency_eval(response=response)
        score = float(fluency_result.get("fluency", 0))
        reasoning = fluency_result.get("fluency_reason", "")
        
        emit_evaluation_event(
            create_evaluation_event(
                name="Fluency",
                score=score,
                reasoning=reasoning,
                span_id=span_id,
                trace_id=trace_id
            )
        )
        results.append({"metric": "fluency", "score": score, "reasoning": reasoning})
    except Exception as e:
        logger.error(f"Fluency evaluation failed: {e}")
    
    # 2. Relevance (if user query exists)
    if user_query:
        try:
            relevance_result = evaluator.relevance_eval(
                query=user_query,
                response=response
            )
            score = float(relevance_result.get("relevance", 0))
            reasoning = relevance_result.get("relevance_reason", "")
            
            emit_evaluation_event(
                create_evaluation_event(
                    name="Relevance",
                    score=score,
                    reasoning=reasoning,
                    span_id=span_id,
                    trace_id=trace_id
                )
            )
            results.append({"metric": "relevance", "score": score, "reasoning": reasoning})
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
    
    # 3. Groundedness (if context exists)
    if context:
        try:
            groundedness_result = evaluator.groundedness_eval(
                response=response,
                context=context
            )
            score = float(groundedness_result.get("groundedness", 0))
            reasoning = groundedness_result.get("groundedness_reason", "")
            
            emit_evaluation_event(
                create_evaluation_event(
                    name="Groundedness",
                    score=score,
                    reasoning=reasoning,
                    span_id=span_id,
                    trace_id=trace_id
                )
            )
            results.append({"metric": "groundedness", "score": score, "reasoning": reasoning})
        except Exception as e:
            logger.error(f"Groundedness evaluation failed: {e}")
    
    return results


def create_evaluation_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of all evaluations."""
    total_score = 0
    total_count = 0
    evaluation_counts = defaultdict(int)
    
    for category, items in evaluation_results.items():
        for item in items:
            if "result" in item:  # Single result
                result = item["result"]
                total_score += result["score"]
                total_count += 1
                evaluation_counts[category] += 1
            elif "results" in item:  # Multiple results
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


def main():
    """Main function to extract and evaluate traces."""
    # Configuration
    trace_id = "a33d66627a40c88e7ec94860fa967893"
    
    # Extract, evaluate, and save
    extract_and_evaluate_trace(
        trace_id=trace_id,
        output_file="trace_with_evaluations.json",
        time_range_hours=48,  # Increased time range
        evaluate=True
    )


if __name__ == "__main__":
    main()