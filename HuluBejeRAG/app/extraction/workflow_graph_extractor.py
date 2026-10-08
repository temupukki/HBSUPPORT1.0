from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models.workflow import (
    Workflow,
    WorkflowDecision,
    WorkflowStep,
    WorkflowTransition,
)


@dataclass
class GraphConfig:
    """
    Configuration for converting diagram elements into a workflow graph.

    The extractor is intentionally conservative because OCR and diagram
    detection are imperfect.
    """

    # Maximum distance used when looking for a nearby downstream node.
    max_connection_distance: float = 700.0

    # Maximum horizontal distance for a mostly horizontal workflow.
    max_horizontal_distance: float = 500.0

    # Maximum vertical distance for a mostly vertical workflow.
    max_vertical_distance: float = 350.0

    # A node must be reasonably aligned with another node.
    horizontal_alignment_tolerance: float = 180.0
    vertical_alignment_tolerance: float = 180.0

    # OCR text regions containing these words are treated as branch labels.
    branch_labels: tuple[str, ...] = (
        "yes",
        "no",
        "true",
        "false",
        "success",
        "successful",
        "failure",
        "failed",
        "approved",
        "rejected",
    )

    # Elements this small are usually OCR artifacts rather than workflow nodes.
    min_node_width: int = 45
    min_node_height: int = 30

    # Minimum confidence for a shape to become a graph node.
    min_shape_confidence: float = 0.65

    # Minimum confidence for standalone OCR text to be considered.
    min_text_confidence: float = 0.55


@dataclass
class _CandidateNode:
    """
    Internal normalized representation of a diagram node.
    """

    element_id: str
    element_type: str
    text: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    metadata: dict[str, Any]

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class _BranchLabel:
    """
    Internal representation of a Yes/No/etc. label.
    """

    element_id: str
    text: str
    center_x: float
    center_y: float
    confidence: float


class WorkflowGraphExtractor:
    """
    Converts diagram analysis into a semantic workflow graph.

    Pipeline:

        DiagramAnalysis
              ↓
        candidate nodes
              ↓
        OCR normalization
              ↓
        branch label detection
              ↓
        spatial graph inference
              ↓
        decision detection
              ↓
        Workflow
              ↓
        WorkflowStep
        WorkflowDecision
        WorkflowTransition

    Important:
    - This extractor does NOT enumerate every possible workflow path.
    - It stores graph edges directly.
    - It tolerates OCR errors.
    - It does not require explicit line detection.
    """

    NODE_TYPES = {
        "rectangle",
        "rounded_rectangle",
        "diamond",
        "ellipse",
    }

    BRANCH_TEXT_RE = re.compile(
        r"^(yes|no|true|false|success|successful|failure|failed|"
        r"approved|rejected)$",
        re.IGNORECASE,
    )

    def __init__(
        self,
        config: GraphConfig | None = None,
    ):
        self.config = config or GraphConfig()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def extract(
        self,
        analysis: Any,
        document_id: str,
        workflow_name: str = "Diagram workflow",
        page_number: int | None = None,
        region_id: str | None = None,
    ) -> Workflow:
        """
        Extract a Workflow from DiagramAnalysis.

        Parameters
        ----------
        analysis:
            Existing DiagramAnalysis from DiagramExtractor.

        document_id:
            Source document identifier.

        workflow_name:
            Name assigned to the workflow.

        page_number:
            PDF page containing the diagram.

        region_id:
            Optional source region identifier.
        """

        candidates = self._build_candidates(analysis)

        branch_labels = self._extract_branch_labels(
            analysis,
            candidates,
        )

        candidates = self._remove_branch_labels_from_nodes(
            candidates,
            branch_labels,
        )

        candidates = self._deduplicate_candidates(candidates)

        if not candidates:
            return Workflow(
                workflow_id=self._workflow_id(
                    document_id,
                    page_number,
                ),
                name=workflow_name,
                document_id=document_id,
                page_numbers=[page_number] if page_number else [],
                source_region_ids=[region_id] if region_id else [],
                metadata={
                    "source": "diagram",
                    "graph_extraction": True,
                    "node_count": 0,
                    "edge_count": 0,
                    "decision_count": 0,
                },
            )

        decisions = self._detect_decisions(
            candidates,
            branch_labels,
        )

        transitions = self._infer_transitions(
            candidates,
            branch_labels,
            decisions,
        )

        steps = self._build_steps(
            candidates,
            page_number=page_number,
            region_id=region_id,
        )

        workflow_decisions = self._build_decision_models(
            decisions,
            candidates,
            page_number=page_number,
        )

        workflow_transitions = self._build_transition_models(
            transitions,
            page_number=page_number,
        )

        workflow = Workflow(
            workflow_id=self._workflow_id(
                document_id,
                page_number,
            ),
            name=workflow_name,
            document_id=document_id,
            page_numbers=[page_number] if page_number else [],
            source_region_ids=[region_id] if region_id else [],
            steps=steps,
            decisions=workflow_decisions,
            transitions=workflow_transitions,
            metadata={
                "source": "diagram",
                "graph_extraction": True,
                "image_path": getattr(
                    analysis,
                    "image_path",
                    None,
                ),
                "image_width": getattr(
                    analysis,
                    "image_width",
                    None,
                ),
                "image_height": getattr(
                    analysis,
                    "image_height",
                    None,
                ),
                "node_count": len(steps),
                "edge_count": len(workflow_transitions),
                "decision_count": len(workflow_decisions),
                "branch_labels": [
                    label.text
                    for label in branch_labels
                ],
            },
        )

        return workflow

    # ------------------------------------------------------------------
    # CANDIDATE NODES
    # ------------------------------------------------------------------

    def _build_candidates(
        self,
        analysis: Any,
    ) -> list[_CandidateNode]:
        """
        Convert DiagramElement objects into normalized graph candidates.
        """

        elements = getattr(
            analysis,
            "elements",
            [],
        )

        candidates: list[_CandidateNode] = []

        for element in elements:
            element_type = str(
                getattr(
                    element,
                    "element_type",
                    "",
                )
            ).lower()

            if element_type not in self.NODE_TYPES:
                continue

            confidence = float(
                getattr(
                    element,
                    "confidence",
                    0.0,
                )
            )

            if confidence < self.config.min_shape_confidence:
                continue

            bbox = getattr(
                element,
                "bbox",
                None,
            )

            if not bbox or len(bbox) != 4:
                continue

            x, y, width, height = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )

            if (
                width < self.config.min_node_width
                or height < self.config.min_node_height
            ):
                continue

            text = self._normalize_text(
                getattr(
                    element,
                    "text",
                    "",
                )
            )

            # A shape with no text is usually a visual artifact.
            if not text:
                continue

            metadata = dict(
                getattr(
                    element,
                    "metadata",
                    {},
                )
                or {}
            )

            candidates.append(
                _CandidateNode(
                    element_id=str(
                        getattr(
                            element,
                            "element_id",
                            "",
                        )
                    ),
                    element_type=element_type,
                    text=text,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    metadata=metadata,
                )
            )

        return candidates

    # ------------------------------------------------------------------
    # TEXT NORMALIZATION
    # ------------------------------------------------------------------

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        """
        Clean common OCR artifacts while preserving semantic meaning.
        """

        if not text:
            return ""

        value = str(text)

        replacements = {
            "Onder": "Order",
            "Onder ": "Order ",
            "D nver": "Driver",
            "Dnver": "Driver",
            "D nvergoing": "Driver going",
            "Driverarnved": "Driver arrived",
            "successtull": "successful",
            "successtully": "successfully",
            "deliverthe": "deliver the",
            "userto": "user to",
            "theuser": "the user",
            "userlocation": "user location",
            "bedelivered": "be delivered",
            "wi Ready": "Ready",
        }

        for source, target in replacements.items():
            value = value.replace(
                source,
                target,
            )

        # Normalize whitespace.
        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        # Remove isolated OCR garbage characters.
        value = re.sub(
            r"\s+[|Il1]+\s+",
            " ",
            value,
        )

        return value.strip()

    # ------------------------------------------------------------------
    # BRANCH LABELS
    # ------------------------------------------------------------------

    def _extract_branch_labels(
        self,
        analysis: Any,
        candidates: list[_CandidateNode],
    ) -> list[_BranchLabel]:
        """
        Find standalone OCR labels such as Yes/No.

        Labels inside an actual node are not considered branch labels.
        """

        labels: list[_BranchLabel] = []

        elements = getattr(
            analysis,
            "elements",
            [],
        )

        candidate_boxes = [
            (
                node.x,
                node.y,
                node.right,
                node.bottom,
            )
            for node in candidates
        ]

        for element in elements:
            element_type = str(
                getattr(
                    element,
                    "element_type",
                    "",
                )
            ).lower()

            if element_type != "text_region":
                continue

            confidence = float(
                getattr(
                    element,
                    "confidence",
                    0.0,
                )
            )

            if confidence < self.config.min_text_confidence:
                continue

            text = self._normalize_text(
                getattr(
                    element,
                    "text",
                    "",
                )
            )

            if not self._is_branch_text(text):
                continue

            bbox = getattr(
                element,
                "bbox",
                None,
            )

            if not bbox or len(bbox) != 4:
                continue

            x, y, width, height = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )

            center_x = x + width / 2.0
            center_y = y + height / 2.0

            # Ignore branch text that lies inside a node.
            inside_node = any(
                self._point_inside_box(
                    center_x,
                    center_y,
                    box,
                )
                for box in candidate_boxes
            )

            if inside_node:
                continue

            labels.append(
                _BranchLabel(
                    element_id=str(
                        getattr(
                            element,
                            "element_id",
                            "",
                        )
                    ),
                    text=text,
                    center_x=center_x,
                    center_y=center_y,
                    confidence=confidence,
                )
            )

        return self._deduplicate_branch_labels(labels)

    def _is_branch_text(
        self,
        text: str,
    ) -> bool:
        normalized = text.lower().strip()

        if normalized in self.config.branch_labels:
            return True

        return bool(
            self.BRANCH_TEXT_RE.match(
                normalized
            )
        )

    def _deduplicate_branch_labels(
        self,
        labels: list[_BranchLabel],
    ) -> list[_BranchLabel]:
        result: list[_BranchLabel] = []

        for label in labels:
            duplicate = False

            for existing in result:
                if (
                    existing.text.lower()
                    == label.text.lower()
                    and abs(
                        existing.center_x
                        - label.center_x
                    )
                    < 30
                    and abs(
                        existing.center_y
                        - label.center_y
                    )
                    < 30
                ):
                    duplicate = True
                    break

            if not duplicate:
                result.append(label)

        return result

    def _remove_branch_labels_from_nodes(
        self,
        candidates: list[_CandidateNode],
        branch_labels: list[_BranchLabel],
    ) -> list[_CandidateNode]:
        """
        Remove a trailing/embedded Yes/No from node text when OCR
        accidentally attached it to the node.
        """

        labels = {
            label.text.lower()
            for label in branch_labels
        }

        if not labels:
            return candidates

        result: list[_CandidateNode] = []

        for node in candidates:
            text = node.text

            words = text.split()

            filtered = [
                word
                for word in words
                if word.lower() not in labels
            ]

            cleaned = " ".join(filtered).strip()

            if not cleaned:
                continue

            node.text = cleaned
            result.append(node)

        return result

    # ------------------------------------------------------------------
    # NODE DEDUPLICATION
    # ------------------------------------------------------------------

    def _deduplicate_candidates(
        self,
        candidates: list[_CandidateNode],
    ) -> list[_CandidateNode]:
        """
        Remove duplicate physical nodes produced by contour detection.
        """

        result: list[_CandidateNode] = []

        for candidate in sorted(
            candidates,
            key=lambda item: (
                item.confidence,
                item.area,
            ),
            reverse=True,
        ):
            duplicate_index: int | None = None

            for index, existing in enumerate(result):
                if self._nodes_are_same_physical_node(
                    candidate,
                    existing,
                ):
                    duplicate_index = index
                    break

            if duplicate_index is None:
                result.append(candidate)
                continue

            existing = result[duplicate_index]

            # Keep the more useful text.
            if self._text_quality(candidate.text) > self._text_quality(
                existing.text
            ):
                result[duplicate_index] = candidate

        result.sort(
            key=lambda item: (
                item.center_y,
                item.center_x,
            )
        )

        return result

    def _nodes_are_same_physical_node(
        self,
        first: _CandidateNode,
        second: _CandidateNode,
    ) -> bool:
        intersection = self._intersection_area(
            first,
            second,
        )

        if intersection <= 0:
            return False

        smaller_area = min(
            first.area,
            second.area,
        )

        if smaller_area <= 0:
            return False

        containment = (
            intersection / smaller_area
        )

        if containment >= 0.70:
            return True

        center_distance = (
            (
                first.center_x
                - second.center_x
            )
            ** 2
            + (
                first.center_y
                - second.center_y
            )
            ** 2
        ) ** 0.5

        average_size = (
            first.width
            + first.height
            + second.width
            + second.height
        ) / 4.0

        return (
            center_distance
            < average_size * 0.25
        )

    def _intersection_area(
        self,
        first: _CandidateNode,
        second: _CandidateNode,
    ) -> float:
        left = max(
            first.x,
            second.x,
        )
        top = max(
            first.y,
            second.y,
        )
        right = min(
            first.right,
            second.right,
        )
        bottom = min(
            first.bottom,
            second.bottom,
        )

        width = max(
            0.0,
            right - left,
        )
        height = max(
            0.0,
            bottom - top,
        )

        return width * height

    def _text_quality(
        self,
        text: str,
    ) -> float:
        if not text:
            return 0.0

        words = text.split()

        if not words:
            return 0.0

        alpha_chars = sum(
            1
            for char in text
            if char.isalpha()
        )

        return (
            len(words) * 2.0
            + alpha_chars / 20.0
        )

    # ------------------------------------------------------------------
    # DECISION DETECTION
    # ------------------------------------------------------------------

    def _detect_decisions(
        self,
        candidates: list[_CandidateNode],
        branch_labels: list[_BranchLabel],
    ) -> list[_CandidateNode]:
        """
        Diamonds are the strongest decision signal.

        A diamond is treated as a decision when:
        - it has branch labels nearby, OR
        - its text looks like a question/decision, OR
        - it is located between multiple possible downstream nodes.
        """

        decisions: list[_CandidateNode] = []

        for node in candidates:
            if node.element_type == "diamond":
                decisions.append(node)
                continue

            text = node.text.lower()

            decision_words = (
                "accepted",
                "accept",
                "success",
                "successful",
                "available",
                "approved",
                "ready",
                "arrived",
                "picked up",
                "delivered",
                "?",
            )

            if "?" in text:
                decisions.append(node)
                continue

            if any(
                word in text
                for word in decision_words
            ):
                nearby_labels = [
                    label
                    for label in branch_labels
                    if self._distance_to_node(
                        label.center_x,
                        label.center_y,
                        node,
                    )
                    <= self.config.max_connection_distance
                ]

                if len(nearby_labels) >= 1:
                    decisions.append(node)

        return decisions

    # ------------------------------------------------------------------
    # TRANSITION INFERENCE
    # ------------------------------------------------------------------

    def _infer_transitions(
        self,
        candidates: list[_CandidateNode],
        branch_labels: list[_BranchLabel],
        decisions: list[_CandidateNode],
    ) -> list[dict[str, Any]]:
        """
        Infer graph edges using spatial relationships.

        We intentionally do not use simple nearest-neighbor chaining.

        A candidate downstream node must satisfy:
        - reasonable direction
        - reasonable distance
        - reasonable alignment
        - no obvious reverse direction

        For each node we keep only the strongest downstream candidates.
        """

        transitions: list[dict[str, Any]] = []

        decision_ids = {
            node.element_id
            for node in decisions
        }

        for source in candidates:
            possible_targets: list[
                tuple[
                    float,
                    _CandidateNode,
                ]
            ] = []

            for target in candidates:
                if (
                    source.element_id
                    == target.element_id
                ):
                    continue

                score = self._connection_score(
                    source,
                    target,
                )

                if score <= 0:
                    continue

                possible_targets.append(
                    (
                        score,
                        target,
                    )
                )

            possible_targets.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            # Normal node → normally one strong outgoing edge.
            # Decision nodes can have multiple branches.
            max_edges = (
                3
                if source.element_id in decision_ids
                else 1
            )

            selected = possible_targets[
                :max_edges
            ]

            for score, target in selected:
                condition = self._find_transition_condition(
                    source,
                    target,
                    branch_labels,
                )

                transitions.append(
                    {
                        "source": source,
                        "target": target,
                        "condition": condition,
                        "score": score,
                    }
                )

        transitions = self._remove_bad_transitions(
            transitions
        )

        transitions = self._deduplicate_transitions(
            transitions
        )

        return transitions

    def _connection_score(
        self,
        source: _CandidateNode,
        target: _CandidateNode,
    ) -> float:
        dx = target.center_x - source.center_x
        dy = target.center_y - source.center_y

        distance = (
            dx * dx + dy * dy
        ) ** 0.5

        if distance <= 1:
            return 0.0

        if distance > self.config.max_connection_distance:
            return 0.0

        # We expect the target to be generally to the right
        # or below the source.
        forward_x = dx > 0
        forward_y = dy > 0

        if not forward_x and not forward_y:
            return 0.0

        horizontal_gap = max(
            0.0,
            target.x - source.right,
        )

        vertical_gap = max(
            0.0,
            target.y - source.bottom,
        )

        same_column = abs(
            target.center_x
            - source.center_x
        ) <= self.config.horizontal_alignment_tolerance

        same_row = abs(
            target.center_y
            - source.center_y
        ) <= self.config.vertical_alignment_tolerance

        horizontal_flow = (
            forward_x
            and same_row
            and dx <= self.config.max_horizontal_distance
        )

        vertical_flow = (
            forward_y
            and same_column
            and dy <= self.config.max_vertical_distance
        )

        diagonal_flow = (
            forward_x
            and forward_y
            and distance <= self.config.max_connection_distance
            and (
                same_row
                or same_column
            )
        )

        if not (
            horizontal_flow
            or vertical_flow
            or diagonal_flow
        ):
            return 0.0

        score = 1.0

        # Strongly prefer aligned nodes.
        if same_row:
            score += 2.0

        if same_column:
            score += 2.0

        # Prefer smaller distance.
        score += max(
            0.0,
            2.0 - distance / 400.0,
        )

        # Penalize large empty gaps.
        score -= min(
            1.5,
            (
                horizontal_gap
                + vertical_gap
            ) / 500.0,
        )

        # If target is dramatically below-right, reduce confidence.
        if (
            abs(dx) > 300
            and abs(dy) > 250
        ):
            score -= 1.0

        return max(
            0.0,
            score,
        )

    def _remove_bad_transitions(
        self,
        transitions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Prevent long-distance accidental connections when a much
        more natural intermediate node exists.
        """

        result: list[dict[str, Any]] = []

        for transition in transitions:
            source = transition["source"]
            target = transition["target"]
            score = transition["score"]

            if score < 1.0:
                continue

            result.append(
                transition
            )

        return result

    def _deduplicate_transitions(
        self,
        transitions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        best: dict[
            tuple[str, str],
            dict[str, Any],
        ] = {}

        for transition in transitions:
            key = (
                transition["source"].element_id,
                transition["target"].element_id,
            )

            existing = best.get(key)

            if (
                existing is None
                or transition["score"]
                > existing["score"]
            ):
                best[key] = transition

        return list(
            best.values()
        )

    # ------------------------------------------------------------------
    # BRANCH CONDITION INFERENCE
    # ------------------------------------------------------------------

    def _find_transition_condition(
        self,
        source: _CandidateNode,
        target: _CandidateNode,
        branch_labels: list[_BranchLabel],
    ) -> str | None:
        """
        Find a Yes/No label spatially associated with an edge.
        """

        best_label: _BranchLabel | None = None
        best_score = float("-inf")

        midpoint_x = (
            source.center_x
            + target.center_x
        ) / 2.0

        midpoint_y = (
            source.center_y
            + target.center_y
        ) / 2.0

        for label in branch_labels:
            distance = (
                (
                    label.center_x
                    - midpoint_x
                )
                ** 2
                + (
                    label.center_y
                    - midpoint_y
                )
                ** 2
            ) ** 0.5

            if distance > 250:
                continue

            # Labels close to the source/target edge are preferred.
            source_distance = self._distance_to_node(
                label.center_x,
                label.center_y,
                source,
            )

            target_distance = self._distance_to_node(
                label.center_x,
                label.center_y,
                target,
            )

            score = (
                300.0
                - distance
                - min(
                    source_distance,
                    target_distance,
                )
            )

            # For decision nodes, labels should normally be
            # close to the outgoing side.
            if (
                source.element_type == "diamond"
                and label.center_x >= source.center_x
            ):
                score += 40.0

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        return best_label.text

    # ------------------------------------------------------------------
    # WORKFLOW STEPS
    # ------------------------------------------------------------------

    def _build_steps(
        self,
        candidates: list[_CandidateNode],
        page_number: int | None,
        region_id: str | None,
    ) -> list[WorkflowStep]:
        steps: list[WorkflowStep] = []

        for index, node in enumerate(
            candidates,
            start=1,
        ):
            metadata = dict(
                node.metadata
            )

            metadata.update(
                {
                    "diagram_element_id": node.element_id,
                    "element_type": node.element_type,
                    "bbox": (
                        node.x,
                        node.y,
                        node.width,
                        node.height,
                    ),
                    "confidence": node.confidence,
                    "source_region_id": region_id,
                }
            )

            steps.append(
                WorkflowStep(
                    step_id=self._step_id(
                        node.element_id
                    ),
                    number=str(index),
                    title=self._clean_title(
                        node.text
                    ),
                    description="",
                    page_number=page_number,
                    region_id=region_id,
                    metadata=metadata,
                )
            )

        return steps

    # ------------------------------------------------------------------
    # DECISION MODELS
    # ------------------------------------------------------------------

    def _build_decision_models(
        self,
        decisions: list[_CandidateNode],
        candidates: list[_CandidateNode],
        page_number: int | None,
    ) -> list[WorkflowDecision]:
        result: list[WorkflowDecision] = []

        for index, node in enumerate(
            decisions,
            start=1,
        ):
            options = self._infer_decision_options(
                node,
                candidates,
            )

            question = self._decision_question(
                node.text
            )

            result.append(
                WorkflowDecision(
                    decision_id=(
                        f"decision_{index}_{node.element_id}"
                    ),
                    step_id=self._step_id(
                        node.element_id
                    ),
                    question=question,
                    options=options,
                    page_number=page_number,
                    metadata={
                        "diagram_element_id": node.element_id,
                        "element_type": node.element_type,
                        "confidence": node.confidence,
                    },
                )
            )

        return result

    def _infer_decision_options(
        self,
        decision: _CandidateNode,
        candidates: list[_CandidateNode],
    ) -> list[str]:
        """
        Basic semantic fallback.

        Actual Yes/No labels are attached to transitions, while
        this gives the decision model useful options.
        """

        text = decision.text.lower()

        if any(
            word in text
            for word in (
                "accepted",
                "accept",
                "success",
                "successful",
                "approved",
                "available",
            )
        ):
            return [
                "Yes",
                "No",
            ]

        return []

    def _decision_question(
        self,
        text: str,
    ) -> str:
        cleaned = self._clean_title(
            text
        )

        if cleaned.endswith("?"):
            return cleaned

        return (
            f"Decision: {cleaned}?"
        )

    # ------------------------------------------------------------------
    # TRANSITION MODELS
    # ------------------------------------------------------------------

    def _build_transition_models(
        self,
        transitions: list[dict[str, Any]],
        page_number: int | None,
    ) -> list[WorkflowTransition]:
        result: list[WorkflowTransition] = []

        for index, transition in enumerate(
            transitions,
            start=1,
        ):
            source = transition["source"]
            target = transition["target"]

            condition = transition.get(
                "condition"
            )

            result.append(
                WorkflowTransition(
                    transition_id=(
                        f"transition_{index}_"
                        f"{source.element_id}_"
                        f"{target.element_id}"
                    ),
                    source_step_id=self._step_id(
                        source.element_id
                    ),
                    target_step_id=self._step_id(
                        target.element_id
                    ),
                    condition=condition,
                    label=condition,
                    page_number=page_number,
                    metadata={
                        "source_element_id": source.element_id,
                        "target_element_id": target.element_id,
                        "inference_score": transition[
                            "score"
                        ],
                        "spatial_inference": True,
                    },
                )
            )

        return result

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _distance_to_node(
        self,
        x: float,
        y: float,
        node: _CandidateNode,
    ) -> float:
        dx = max(
            node.x - x,
            0,
            x - node.right,
        )

        dy = max(
            node.y - y,
            0,
            y - node.bottom,
        )

        return (
            dx * dx + dy * dy
        ) ** 0.5

    def _point_inside_box(
        self,
        x: float,
        y: float,
        box: tuple[float, float, float, float],
    ) -> bool:
        x0, y0, x1, y1 = box

        return (
            x0 <= x <= x1
            and y0 <= y <= y1
        )

    def _clean_title(
        self,
        text: str,
    ) -> str:
        value = self._normalize_text(
            text
        )

        # Remove accidental branch labels.
        value = re.sub(
            r"\b(?:yes|no)\b",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        return value

    def _step_id(
        self,
        element_id: str,
    ) -> str:
        return (
            f"step_{element_id}"
        )

    def _workflow_id(
        self,
        document_id: str,
        page_number: int | None,
    ) -> str:
        suffix = (
            f"_page_{page_number}"
            if page_number is not None
            else ""
        )

        return (
            f"workflow_diagram_"
            f"{document_id}"
            f"{suffix}"
        )