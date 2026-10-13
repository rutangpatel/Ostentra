from typing import Annotated, Literal
from typing_extensions import NotRequired, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class Requirement(TypedDict):
    id: str
    text: str
    covered: bool

class InterviewState(TypedDict):
    role: str
    job_description: str
    resume_summary: NotRequired[str]
    requirements: NotRequired[list[Requirement]]
    messages: Annotated[list[AnyMessage], add_messages]
    phase: NotRequired[Literal["setup", "in_progress", "completed"]]
    turn_count: NotRequired[int] # Just for future
