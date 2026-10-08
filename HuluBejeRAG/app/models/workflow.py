from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowStep:
    """
    One executable/logical step inside a workflow.
    """

    step_id: str

    number: str

    title: str

    description: str = ""

    page_number: int | None = None

    region_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class WorkflowTransition:
    """
    Represents movement from one workflow node to another.
    """

    transition_id: str

    source_step_id: str

    target_step_id: str

    condition: str | None = None

    label: str | None = None

    page_number: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class WorkflowDecision:
    """
    Represents a decision/branch in a workflow.
    """

    decision_id: str

    step_id: str

    question: str

    options: list[str] = field(
        default_factory=list
    )

    page_number: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class Workflow:
    """
    Complete structured workflow.

    The workflow is intentionally graph-oriented.

    This allows us to represent:

        A -> B -> C

    as well as:

        A -> B
             ├── YES -> C
             └── NO  -> D
    """

    workflow_id: str

    name: str

    document_id: str

    page_numbers: list[int] = field(
        default_factory=list
    )

    source_region_ids: list[str] = field(
        default_factory=list
    )

    steps: list[WorkflowStep] = field(
        default_factory=list
    )

    decisions: list[WorkflowDecision] = field(
        default_factory=list
    )

    transitions: list[WorkflowTransition] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)