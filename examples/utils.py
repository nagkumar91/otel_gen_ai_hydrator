import os
from opentelemetry._events import Event, get_event_logger
from azure.monitor.opentelemetry import configure_azure_monitor

# Configure Azure Monitor for OpenTelemetry telemetry collection
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