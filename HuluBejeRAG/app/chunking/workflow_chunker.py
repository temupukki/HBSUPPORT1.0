from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.workflow import (
    Workflow,
    WorkflowDecision,
    WorkflowStep,
    WorkflowTransition,
)


@dataclass
class WorkflowChunk:
    """
    Retrieval-ready representation of part of a workflow.

    A WorkflowChunk keeps both:
    - natural-language text for embedding/retrieval
    - graph metadata for later reasoning and citation
    """

    chunk_id: str
    text: str

    workflow_id: str
    workflow_name: str

    step_ids: list[str] = field(default_factory=list)
    decision_ids: list[str] = field(default_factory=list)
    transition_ids: list[str] = field(default_factory=list)

    page_numbers: list[int] = field(default_factory=list)

    chunk_type: str = "workflow"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def character_count(self) -> int:
        return len(self.text)

    @property
    def step_count(self) -> int:
        return len(self.step_ids)


@dataclass
class WorkflowChunkConfig:
    """
    Configuration for workflow-aware chunking.
    """

    # Target size for normal chunks.
    target_characters: int = 1200

    # Hard upper limit.
    max_characters: int = 1800

    # Minimum size before we consider a chunk complete.
    min_characters: int = 450

    # Number of neighboring steps whose context may be included.
    context_steps: int = 1

    # Maximum number of steps in one chunk.
    max_steps_per_chunk: int = 4

    # Whether to create a dedicated chunk around decisions.
    include_decision_chunks: bool = True

    # Whether to create a compact complete-workflow overview.
    include_workflow_overview: bool = True

    # Avoid producing tiny duplicate branch chunks.
    min_branch_chunk_characters: int = 300


class WorkflowChunker:
    """
    Converts a Workflow graph into retrieval-ready chunks.

    Design goals:

    1. Preserve workflow order.
    2. Preserve decisions and branches.
    3. Preserve transition conditions such as Yes/No.
    4. Keep neighboring context.
    5. Avoid enumerating every possible workflow path.
    6. Produce useful natural-language text for embeddings.
    7. Preserve graph IDs in metadata.

    Input:

        Workflow
            ├── WorkflowStep
            ├── WorkflowDecision
            └── WorkflowTransition

    Output:

        list[WorkflowChunk]
    """

    def __init__(
        self,
        config: WorkflowChunkConfig | None = None,
    ):
        self.config = (
            config
            or WorkflowChunkConfig()
        )

    # ================================================================
    # PUBLIC API
    # ================================================================

    def chunk(
        self,
        workflow: Workflow,
    ) -> list[WorkflowChunk]:
        """
        Convert a workflow graph into retrieval-ready chunks.
        """

        if not workflow.steps:
            return []

        chunks: list[WorkflowChunk] = []

        steps = self._ordered_steps(
            workflow
        )

        transitions_by_source = (
            self._transitions_by_source(
                workflow.transitions
            )
        )

        transitions_by_target = (
            self._transitions_by_target(
                workflow.transitions
            )
        )

        decisions_by_step = (
            self._decisions_by_step(
                workflow.decisions
            )
        )

        # ------------------------------------------------------------
        # 1. Complete workflow overview
        # ------------------------------------------------------------

        if (
            self.config.include_workflow_overview
            and len(steps) > 1
        ):
            overview = (
                self._build_workflow_overview(
                    workflow,
                    steps,
                    transitions_by_source,
                    decisions_by_step,
                )
            )

            if overview is not None:
                chunks.append(
                    overview
                )

        # ------------------------------------------------------------
        # 2. Main sequential workflow chunks
        # ------------------------------------------------------------

        main_chunks = (
            self._build_sequential_chunks(
                workflow,
                steps,
                transitions_by_source,
                transitions_by_target,
                decisions_by_step,
            )
        )

        chunks.extend(
            main_chunks
        )

        # ------------------------------------------------------------
        # 3. Dedicated decision/branch chunks
        # ------------------------------------------------------------

        if self.config.include_decision_chunks:
            decision_chunks = (
                self._build_decision_chunks(
                    workflow,
                    steps,
                    decisions_by_step,
                    transitions_by_source,
                )
            )

            chunks.extend(
                decision_chunks
            )

        # ------------------------------------------------------------
        # 4. Remove duplicate / useless chunks
        # ------------------------------------------------------------

        chunks = self._deduplicate_chunks(
            chunks
        )

        chunks = self._remove_tiny_chunks(
            chunks
        )

        # ------------------------------------------------------------
        # 5. Final metadata
        # ------------------------------------------------------------

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk.metadata[
                "chunk_index"
            ] = index

            chunk.metadata[
                "total_chunks"
            ] = len(chunks)

            chunk.metadata[
                "workflow_step_count"
            ] = len(workflow.steps)

            chunk.metadata[
                "workflow_decision_count"
            ] = len(
                workflow.decisions
            )

            chunk.metadata[
                "workflow_transition_count"
            ] = len(
                workflow.transitions
            )

        return chunks

    # ================================================================
    # ORDERING
    # ================================================================

    def _ordered_steps(
        self,
        workflow: Workflow,
    ) -> list[WorkflowStep]:
        """
        Preserve the graph extractor's numbering when possible.

        If numbering is not usable, fall back to page/step ordering.
        """

        def sort_key(
            step: WorkflowStep,
        ) -> tuple[Any, ...]:
            number = self._numeric_number(
                step.number
            )

            page = (
                step.page_number
                if step.page_number is not None
                else 999999
            )

            return (
                number,
                page,
                step.step_id,
            )

        return sorted(
            workflow.steps,
            key=sort_key,
        )

    def _numeric_number(
        self,
        value: str,
    ) -> float:
        try:
            return float(
                str(value)
            )
        except (
            ValueError,
            TypeError,
        ):
            return 999999.0

    # ================================================================
    # GRAPH INDEXES
    # ================================================================

    def _transitions_by_source(
        self,
        transitions: list[WorkflowTransition],
    ) -> dict[
        str,
        list[WorkflowTransition],
    ]:
        result: dict[
            str,
            list[WorkflowTransition],
        ] = {}

        for transition in transitions:
            result.setdefault(
                transition.source_step_id,
                [],
            ).append(
                transition
            )

        return result

    def _transitions_by_target(
        self,
        transitions: list[WorkflowTransition],
    ) -> dict[
        str,
        list[WorkflowTransition],
    ]:
        result: dict[
            str,
            list[WorkflowTransition],
        ] = {}

        for transition in transitions:
            result.setdefault(
                transition.target_step_id,
                [],
            ).append(
                transition
            )

        return result

    def _decisions_by_step(
        self,
        decisions: list[WorkflowDecision],
    ) -> dict[
        str,
        list[WorkflowDecision],
    ]:
        result: dict[
            str,
            list[WorkflowDecision],
        ] = {}

        for decision in decisions:
            result.setdefault(
                decision.step_id,
                [],
            ).append(
                decision
            )

        return result

    # ================================================================
    # WORKFLOW OVERVIEW
    # ================================================================

    def _build_workflow_overview(
        self,
        workflow: Workflow,
        steps: list[WorkflowStep],
        transitions_by_source: dict[
            str,
            list[WorkflowTransition],
        ],
        decisions_by_step: dict[
            str,
            list[WorkflowDecision],
        ],
    ) -> WorkflowChunk | None:
        """
        Build one compact representation of the entire workflow.

        This is extremely useful for broad questions such as:

        "How does restaurant delivery work?"

        It deliberately does not expand every branch.
        """

        lines: list[str] = []

        lines.append(
            f"Workflow: {workflow.name}"
        )

        lines.append(
            "Overall process:"
        )

        for index, step in enumerate(
            steps,
            start=1,
        ):
            lines.append(
                f"{index}. {step.title}"
            )

            decisions = (
                decisions_by_step.get(
                    step.step_id,
                    [],
                )
            )

            if decisions:
                for decision in decisions:
                    options = (
                        ", ".join(
                            decision.options
                        )
                        if decision.options
                        else "multiple outcomes"
                    )

                    lines.append(
                        f"   Decision: "
                        f"{decision.question} "
                        f"Options: {options}"
                    )

            outgoing = (
                transitions_by_source.get(
                    step.step_id,
                    [],
                )
            )

            if len(outgoing) > 1:
                for transition in outgoing:
                    target = (
                        self._find_step(
                            steps,
                            transition.target_step_id,
                        )
                    )

                    if target:
                        condition = (
                            transition.condition
                            or transition.label
                            or "branch"
                        )

                        lines.append(
                            f"   {condition} → "
                            f"{target.title}"
                        )

        text = "\n".join(
            lines
        ).strip()

        if not text:
            return None

        return WorkflowChunk(
            chunk_id=(
                f"{workflow.workflow_id}"
                "_overview"
            ),
            text=text,
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            step_ids=[
                step.step_id
                for step in steps
            ],
            decision_ids=[
                decision.decision_id
                for decision in workflow.decisions
            ],
            transition_ids=[
                transition.transition_id
                for transition in workflow.transitions
            ],
            page_numbers=self._page_numbers(
                workflow,
                steps,
            ),
            chunk_type="workflow_overview",
            metadata={
                "content_type": "workflow",
                "chunk_role": "overview",
                "complete_workflow": True,
            },
        )

    # ================================================================
    # SEQUENTIAL CHUNKS
    # ================================================================

    def _build_sequential_chunks(
        self,
        workflow: Workflow,
        steps: list[WorkflowStep],
        transitions_by_source: dict[
            str,
            list[WorkflowTransition],
        ],
        transitions_by_target: dict[
            str,
            list[WorkflowTransition],
        ],
        decisions_by_step: dict[
            str,
            list[WorkflowDecision],
        ],
    ) -> list[WorkflowChunk]:
        """
        Build chunks around logical runs of steps.

        We use the graph's main step ordering but do not flatten branches.
        """

        chunks: list[WorkflowChunk] = []

        current_steps: list[
            WorkflowStep
        ] = []

        for step in steps:
            if not current_steps:
                current_steps.append(
                    step
                )
                continue

            previous = current_steps[-1]

            # A branching point starts a new semantic chunk.
            previous_outgoing = (
                transitions_by_source.get(
                    previous.step_id,
                    [],
                )
            )

            current_has_decision = bool(
                decisions_by_step.get(
                    step.step_id,
                    [],
                )
            )

            previous_has_decision = bool(
                decisions_by_step.get(
                    previous.step_id,
                    [],
                )
            )

            is_branch_boundary = (
                len(previous_outgoing) > 1
                or current_has_decision
                or previous_has_decision
            )

            proposed = (
                current_steps
                + [step]
            )

            proposed_text = (
                self._render_step_sequence(
                    workflow,
                    proposed,
                    transitions_by_source,
                    decisions_by_step,
                )
            )

            too_many_steps = (
                len(proposed)
                > self.config.max_steps_per_chunk
            )

            too_large = (
                len(proposed_text)
                > self.config.max_characters
            )

            if (
                is_branch_boundary
                or too_many_steps
                or too_large
            ):
                chunks.append(
                    self._make_sequence_chunk(
                        workflow,
                        current_steps,
                        transitions_by_source,
                        transitions_by_target,
                        decisions_by_step,
                        len(chunks),
                    )
                )

                # Include one neighboring step as context
                # when possible.
                current_steps = [
                    previous,
                    step,
                ]

                if len(current_steps) > 2:
                    current_steps = (
                        current_steps[-2:]
                    )

            else:
                current_steps.append(
                    step
                )

        if current_steps:
            chunks.append(
                self._make_sequence_chunk(
                    workflow,
                    current_steps,
                    transitions_by_source,
                    transitions_by_target,
                    decisions_by_step,
                    len(chunks),
                )
            )

        return chunks

    def _make_sequence_chunk(
        self,
        workflow: Workflow,
        steps: list[WorkflowStep],
        transitions_by_source: dict[
            str,
            list[WorkflowTransition],
        ],
        transitions_by_target: dict[
            str,
            list[WorkflowTransition],
        ],
        decisions_by_step: dict[
            str,
            list[WorkflowDecision],
        ],
        index: int,
    ) -> WorkflowChunk:
        text = self._render_step_sequence(
            workflow,
            steps,
            transitions_by_source,
            decisions_by_step,
        )

        transition_ids: list[str] = []
        decision_ids: list[str] = []

        step_ids = [
            step.step_id
            for step in steps
        ]

        for step in steps:
            for transition in (
                transitions_by_source.get(
                    step.step_id,
                    [],
                )
            ):
                if (
                    transition.target_step_id
                    in step_ids
                ):
                    transition_ids.append(
                        transition.transition_id
                    )

            for decision in (
                decisions_by_step.get(
                    step.step_id,
                    [],
                )
            ):
                decision_ids.append(
                    decision.decision_id
                )

        return WorkflowChunk(
            chunk_id=(
                f"{workflow.workflow_id}"
                f"_sequence_{index + 1}"
            ),
            text=text,
            workflow_id=workflow.workflow_id,
            workflow_name=workflow.name,
            step_ids=step_ids,
            decision_ids=self._unique(
                decision_ids
            ),
            transition_ids=self._unique(
                transition_ids
            ),
            page_numbers=self._pages_from_steps(
                steps
            ),
            chunk_type="workflow_sequence",
            metadata={
                "content_type": "workflow",
                "chunk_role": "sequence",
                "step_range": [
                    steps[0].number,
                    steps[-1].number,
                ],
            },
        )

    def _render_step_sequence(
        self,
        workflow: Workflow,
        steps: list[WorkflowStep],
        transitions_by_source: dict[
            str,
            list[WorkflowTransition],
        ],
        decisions_by_step: dict[
            str,
            list[WorkflowDecision],
        ],
    ) -> str:
        lines: list[str] = []

        lines.append(
            f"Workflow: {workflow.name}"
        )

        lines.append(
            "Process:"
        )

        for index, step in enumerate(
            steps,
            start=1,
        ):
            lines.append(
                f"{index}. {step.title}"
            )

            outgoing = (
                transitions_by_source.get(
                    step.step_id,
                    [],
                )
            )

            if not outgoing:
                continue

            # A single normal transition.
            if len(outgoing) == 1:
                transition = outgoing[0]

                target = (
                    self._find_step(
                        workflow.steps,
                        transition.target_step_id,
                    )
                )

                if target:
                    condition = (
                        transition.condition
                        or transition.label
                    )

                    if condition:
                        lines.append(
                            f"   {condition} → "
                            f"{target.title}"
                        )
                    else:
                        lines.append(
                            f"   Next → "
                            f"{target.title}"
                        )

            # Multiple transitions = branch.
            else:
                lines.append(
                    "   Branches:"
                )

                for transition in outgoing:
                    target = (
                        self._find_step(
                            workflow.steps,
                            transition.target_step_id,
                        )
                    )

                    if not target:
                        continue

                    condition = (
                        transition.condition
                        or transition.label
                        or "Other outcome"
                    )

                    lines.append(
                        f"   - {condition} → "
                        f"{target.title}"
                    )

            decisions = (
                decisions_by_step.get(
                    step.step_id,
                    [],
                )
            )

            for decision in decisions:
                if decision.options:
                    lines.append(
                        "   Decision options: "
                        + ", ".join(
                            decision.options
                        )
                    )

        return "\n".join(
            lines
        ).strip()

    # ================================================================
    # DECISION CHUNKS
    # ================================================================

    def _build_decision_chunks(
        self,
        workflow: Workflow,
        steps: list[WorkflowStep],
        decisions_by_step: dict[
            str,
            list[WorkflowDecision],
        ],
        transitions_by_source: dict[
            str,
            list[WorkflowTransition],
        ],
    ) -> list[WorkflowChunk]:
        """
        Build compact chunks centered around branching decisions.

        This is useful for questions such as:

        "What happens if the order is accepted?"

        or:

        "What are the possible outcomes?"
        """

        chunks: list[WorkflowChunk] = []

        for decision in workflow.decisions:
            step = self._find_step(
                steps,
                decision.step_id,
            )

            if step is None:
                continue

            outgoing = (
                transitions_by_source.get(
                    step.step_id,
                    [],
                )
            )

            if not outgoing:
                continue

            lines: list[str] = []

            lines.append(
                f"Workflow: {workflow.name}"
            )

            lines.append(
                f"Decision: "
                f"{decision.question}"
            )

            if decision.options:
                lines.append(
                    "Possible outcomes: "
                    + ", ".join(
                        decision.options
                    )
                )

            lines.append(
                f"Current step: "
                f"{step.title}"
            )

            lines.append(
                "Outcomes:"
            )

            transition_ids: list[str] = []

            for transition in outgoing:
                target = self._find_step(
                    steps,
                    transition.target_step_id,
                )

                if target is None:
                    continue

                condition = (
                    transition.condition
                    or transition.label
                    or "Next outcome"
                )

                lines.append(
                    f"- {condition}: "
                    f"{target.title}"
                )

                transition_ids.append(
                    transition.transition_id
                )

            text = "\n".join(
                lines
            ).strip()

            if (
                len(text)
                < self.config.min_branch_chunk_characters
            ):
                # Still useful if it contains a real decision.
                text = (
                    text
                    + "\n"
                    + "This decision determines "
                    "the next workflow step."
                )

            chunks.append(
                WorkflowChunk(
                    chunk_id=(
                        f"{workflow.workflow_id}"
                        f"_decision_"
                        f"{decision.decision_id}"
                    ),
                    text=text,
                    workflow_id=workflow.workflow_id,
                    workflow_name=workflow.name,
                    step_ids=[
                        step.step_id
                    ],
                    decision_ids=[
                        decision.decision_id
                    ],
                    transition_ids=transition_ids,
                    page_numbers=(
                        [step.page_number]
                        if step.page_number
                        is not None
                        else []
                    ),
                    chunk_type="workflow_decision",
                    metadata={
                        "content_type": "workflow",
                        "chunk_role": "decision",
                        "decision_question": (
                            decision.question
                        ),
                        "options": decision.options,
                    },
                )
            )

        return chunks

    # ================================================================
    # DEDUPLICATION
    # ================================================================

    def _deduplicate_chunks(
        self,
        chunks: list[WorkflowChunk],
    ) -> list[WorkflowChunk]:
        """
        Remove exact/near duplicate chunks.

        We keep the richer chunk when two chunks describe essentially
        the same step set.
        """

        result: list[
            WorkflowChunk
        ] = []

        for chunk in chunks:
            duplicate_index: int | None = None

            for index, existing in enumerate(
                result
            ):
                same_steps = (
                    set(chunk.step_ids)
                    == set(
                        existing.step_ids
                    )
                )

                same_type = (
                    chunk.chunk_type
                    == existing.chunk_type
                )

                if (
                    same_steps
                    and same_type
                ):
                    duplicate_index = index
                    break

                if self._text_similarity(
                    chunk.text,
                    existing.text,
                ) >= 0.90:
                    duplicate_index = index
                    break

            if duplicate_index is None:
                result.append(
                    chunk
                )
                continue

            existing = result[
                duplicate_index
            ]

            if (
                len(chunk.text)
                > len(existing.text)
            ):
                result[
                    duplicate_index
                ] = chunk

        return result

    def _remove_tiny_chunks(
        self,
        chunks: list[WorkflowChunk],
    ) -> list[WorkflowChunk]:
        result: list[
            WorkflowChunk
        ] = []

        for chunk in chunks:
            minimum = (
                self.config.min_branch_chunk_characters
                if chunk.chunk_type
                == "workflow_decision"
                else self.config.min_characters
            )

            # Always preserve an overview.
            if (
                chunk.chunk_type
                == "workflow_overview"
            ):
                result.append(
                    chunk
                )
                continue

            if len(chunk.text) >= minimum:
                result.append(
                    chunk
                )

        return result

    # ================================================================
    # UTILITY
    # ================================================================

    def _find_step(
        self,
        steps: list[WorkflowStep],
        step_id: str,
    ) -> WorkflowStep | None:
        for step in steps:
            if step.step_id == step_id:
                return step

        return None

    def _pages_from_steps(
        self,
        steps: list[WorkflowStep],
    ) -> list[int]:
        pages = [
            step.page_number
            for step in steps
            if step.page_number
            is not None
        ]

        return self._unique(
            pages
        )

    def _page_numbers(
        self,
        workflow: Workflow,
        steps: list[WorkflowStep],
    ) -> list[int]:
        pages = list(
            workflow.page_numbers
        )

        pages.extend(
            self._pages_from_steps(
                steps
            )
        )

        return self._unique(
            pages
        )

    def _unique(
        self,
        values: list[Any],
    ) -> list[Any]:
        result: list[Any] = []

        for value in values:
            if value not in result:
                result.append(
                    value
                )

        return result

    def _text_similarity(
        self,
        first: str,
        second: str,
    ) -> float:
        first_words = set(
            self._words(first)
        )

        second_words = set(
            self._words(second)
        )

        if not first_words or not second_words:
            return 0.0

        intersection = (
            first_words
            & second_words
        )

        union = (
            first_words
            | second_words
        )

        return len(intersection) / len(
            union
        )

    def _words(
        self,
        text: str,
    ) -> list[str]:
        return [
            word.lower()
            for word in text.split()
            if word.strip()
        ]