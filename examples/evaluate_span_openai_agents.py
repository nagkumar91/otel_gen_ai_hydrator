from datetime import timedelta
import json, os, asyncio, logging
from otel_gen_ai_hydrator.models.events import GenAIAssistantMessageEvent, GenAIChoiceEvent, GenAISystemMessageEvent, GenAIToolMessageEvent, GenAIUserMessageEvent
from otel_gen_ai_hydrator.span_hydrator import SpanHydrator
from otel_gen_ai_hydrator.sources.application_insights import (
    ApplicationInsightsConnector, 
    ApplicationInsightsConfig
)
from opentelemetry._events import Event, get_event_logger
from azure.monitor.opentelemetry import configure_azure_monitor

from azure.ai.evaluation import ToolCallAccuracyEvaluator, AzureOpenAIModelConfiguration, IntentResolutionEvaluator
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

configure_azure_monitor(
    connection_string=os.environ["APPLICATION_INSIGHTS_CONNECTION_STRING"],
)

# Initialize event logger for emitting evaluation events
event_logger = get_event_logger(__name__)

def create_evaluation_event(*,name: str, score: float, reasoning: str, span_id: int, trace_id: int):
    """
    Create an OpenTelemetry event for AI evaluation results.
    
    Args:
        name: Name of the evaluation metric (e.g., "Relevance", "Violence")
        score: Numerical score from the evaluation
        reasoning: Text explanation of the evaluation result
        span_id: OpenTelemetry span ID to associate the event with
        trace_id: OpenTelemetry trace ID to associate the event with
    
    Returns:
        Event: OpenTelemetry event object containing evaluation data
    """
    event = Event(
        name=f'gen_ai.evaluation.{name}',  # Event name following OpenTelemetry convention
        attributes={
            "gen_ai.evaluation.name": name,
            "gen_ai.evaluation.score": score,
            "gen_ai.evaluation.reasoning": reasoning,
            "gen_ai.evaluation.result": "pass/fail" if score >= 0.5 else "fail",  # Example logic for result

        },
        body=f'Evaluation for {name}',  # Needed due to a bug in Azure Monitor Exporter
        span_id=span_id,  # Link to specific span
        trace_id=trace_id,
          # Link to specific trace
    )

    return event

def create_and_log_evaluation_event(*, name: str, score: float, reasoning: str, span_id: int, trace_id: int):
    """
    Create and log an OpenTelemetry event for AI evaluation results.
    
    Args:
        name: Name of the evaluation metric (e.g., "Relevance", "Violence")
        score: Numerical score from the evaluation
        reasoning: Text explanation of the evaluation result
        span_id: OpenTelemetry span ID to associate the event with
        trace_id: OpenTelemetry trace ID to associate the event with
    """
    event = create_evaluation_event(name=name, score=score, reasoning=reasoning, span_id=span_id, trace_id=trace_id)
    event_logger.emit(event)  # Log the event using OpenTelemetry's event logger

def get_output_messages_from_events(span):

    event = span.events[-1]
    if isinstance(event, dict):
        return event
    return event.model_dump_json()

def get_input_messages_from_events(span):
    input_messages = []
    for event in span.events[: len(span.events) - 1]:
        if isinstance(event, dict):
            input_messages.append(event)
        else: 
            input_messages.append(event.model_dump_json())
    return input_messages

model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["MODEL_DEPLOYMENT_NAME"],
    )

# Ankit's App Insights resource ID
app_config = ApplicationInsightsConfig(
    resource_id="/subscriptions/b17253fa-f327-42d6-9686-f3e553e24763/resourceGroups/anksing-vanilla-eval/providers/microsoft.insights/components/anksing-app-insights",
)

app_insights_connector = ApplicationInsightsConnector(app_config)
span_hydrator = SpanHydrator(app_insights_connector)

invoke_agent_span_id = "c2b5b8fb035f4625"  # Example span ID for OpenAI Agents

invoke_agent_span = span_hydrator.get_span_by_id(invoke_agent_span_id, time_range=timedelta(days=30))

tool_definitions = invoke_agent_span.attributes.get("gen_ai.agent.tools", [])


child_spans = span_hydrator.get_child_spans(invoke_agent_span_id, time_range=timedelta(days=30))

def get_query(span):
    if span.span_type in ["chat", "chat.streaming_completions"]:
        if span.attributes.get("gen_ai.input.messages"):
            return json.loads(span.attributes.get("gen_ai.input.messages"))
        else:
            return get_input_messages_from_events(span)

tool_calls = []
query = get_query(child_spans[0])
response = []

# Generate response from child spans
for idx, child_span in enumerate(child_spans):
    span_type = child_span.span_type
    if span_type in ["chat", "chat.streaming_completions"]:
        if child_span.attributes.get("gen_ai.output.messages"):
            response.extend(
                json.loads(child_span.attributes.get("gen_ai.output.messages"))
            )
        else:
            response.extend(get_output_messages_from_events(child_span))

    if child_span.span_type == "execute_tool":
        tool_calls.append(
            json.loads(child_span.attributes.get("gen_ai.tool.input", child_span.attributes.get("gen_ai.tool.call.arguments")))
        )
        if child_span.attributes.get("gen_ai.tool.output"):
            response.append(
                json.loads(child_span.attributes.get("gen_ai.tool.output"))
            )


tool_call_accuracy_evaluator = ToolCallAccuracyEvaluator(
    model_config=model_config,
)

intentent_resolution_evaluattor = IntentResolutionEvaluator(model_config=model_config)

async def main():
    result = await intentent_resolution_evaluattor._flow(
        query=query, 
        response=response, 
        tool_definitions=tool_definitions
    )
    print(json.dumps(result, indent=4))

    # create_and_log_evaluation_event(
    #     name="IntentResolution",
    #     score=result.get("score", 0.0),
    #     reasoning=result.get("explanation", ""),
    #     span_id=int(invoke_agent_span_id, 16),
    #     trace_id=int(invoke_agent_span.trace_id, 16)
    # )
    result = await tool_call_accuracy_evaluator._flow(
        query=query, 
        tool_calls=tool_calls, 
        tool_definitions=tool_definitions
    )
    print(json.dumps(result, indent=4))

    create_and_log_evaluation_event(
        name="ToolCallAccuracy",
        score=result.get("tool_calls_success_level", 0.0),
        reasoning=result.get("chain_of_thought", ""),
        span_id=int(invoke_agent_span_id, 16),
        trace_id=int(invoke_agent_span.trace_id, 16)
    )

if __name__ == "__main__":
    asyncio.run(main())



