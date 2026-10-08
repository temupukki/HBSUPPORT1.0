import re
from typing import Optional

from app.models.region import Region
from app.models.workflow import (
    Workflow,
    WorkflowDecision,
    WorkflowStep,
    WorkflowTransition,
)


class WorkflowExtractor:
    """
    Extracts text-defined workflows from document regions.

    Important:
    - A numbered section is NOT automatically a workflow.
    - A workflow must have strong process/workflow evidence.
    - Ordinary sections such as "Performance metrics" are ignored.
    - Job responsibilities and role descriptions are ignored.
    - Only direct child steps of the workflow section are extracted.
    - Sequential transitions are inferred from step ordering.
    - Decision detection is intentionally conservative.
    - Complex visual workflows will be handled later by VisualExtractor.
    """

    # ---------------------------------------------------------------
    # Numbered section patterns
    # ---------------------------------------------------------------

    STEP_PATTERN = re.compile(
        r"^(?P<number>\d+(?:\.\d+)+)\s+(?P<content>.+)$"
    )

    SECTION_PATTERN = re.compile(
        r"^(?P<number>\d+)\s+(?P<title>.+)$"
    )

    # ---------------------------------------------------------------
    # Strong workflow/process title keywords
    # ---------------------------------------------------------------

    WORKFLOW_TITLE_KEYWORDS = {
        "procedure",
        "process",
        "workflow",
        "flow",
        "fulfillment",
        "registration",
        "approval",
        "payment",
        "delivery",
        "return",
        "refund",
        "order processing",
        "order fulfillment",
        "handling procedure",
        "operating procedure",
    }

    def extract_from_regions(
        self,
        regions: list[Region],
    ) -> list[Workflow]:
        """
        Extract workflows from all eligible regions.
        """

        workflows: list[Workflow] = []

        for region in regions:
            workflow = self.extract_from_region(region)

            if workflow is not None:
                workflows.append(workflow)

        return workflows

    def extract_from_region(
        self,
        region: Region,
    ) -> Optional[Workflow]:
        """
        Extract one text-defined workflow from a region.
        """

        # -----------------------------------------------------------
        # 1. Verify that the region is actually a workflow.
        # -----------------------------------------------------------

        if not self._is_workflow_region(region):
            return None

        lines = self._clean_lines(region.text)

        if not lines:
            return None

        # -----------------------------------------------------------
        # 2. Find the workflow's top-level section.
        # -----------------------------------------------------------

        title_info = self._find_workflow_title(region)

        if title_info is None:
            return None

        workflow_number, workflow_name = title_info

        # -----------------------------------------------------------
        # 3. Extract only direct child steps.
        # -----------------------------------------------------------

        steps = self._extract_steps(
            lines=lines,
            page_number=region.page_number,
            region_id=region.region_id,
            document_id=region.document_id,
            parent_number=workflow_number,
        )

        # A workflow needs at least two steps.
        if len(steps) < 2:
            return None

        # -----------------------------------------------------------
        # 4. Build workflow ID.
        # -----------------------------------------------------------

        workflow_id = (
            f"{region.document_id}"
            f"_workflow_page_{region.page_number}"
            f"_{region.region_id}"
        )

        # -----------------------------------------------------------
        # 5. Create workflow.
        # -----------------------------------------------------------

        workflow = Workflow(
            workflow_id=workflow_id,
            name=workflow_name,
            document_id=region.document_id,
            page_numbers=[region.page_number],
            source_region_ids=[region.region_id],
            steps=steps,
            decisions=[],
            transitions=[],
            metadata={
                "source": "text",
                "region_type": region.region_type,
                "confidence": region.confidence,
                "extraction_method": "numbered_workflow_section",
                "parent_section": workflow_number,
            },
        )

        # -----------------------------------------------------------
        # 6. Infer sequential transitions.
        # -----------------------------------------------------------

        workflow.transitions = (
            self._build_sequential_transitions(workflow)
        )

        # -----------------------------------------------------------
        # 7. Detect explicit textual decisions.
        # -----------------------------------------------------------

        workflow.decisions = (
            self._extract_decisions(workflow)
        )

        return workflow

    # ===============================================================
    # WORKFLOW DETECTION
    # ===============================================================

    def _is_workflow_region(
        self,
        region: Region,
    ) -> bool:
        """
        Determine whether a region actually represents a workflow.

        Strong rule:

        A structured section must have a top-level section title
        containing a workflow/process keyword.

        Example accepted:

            7. Order fulfillment procedure for self-pickup

        Example rejected:

            12. Performance metrics

        Example rejected:

            10.3 Driver roles and duties
        """

        # RegionDetector can explicitly classify a workflow.
        if region.region_type == "workflow_definition":
            return True

        # Other region types are not handled as text workflows.
        if region.region_type != "structured_section":
            return False

        lines = self._clean_lines(region.text)

        if not lines:
            return False

        first_line = lines[0]

        match = self.SECTION_PATTERN.match(first_line)

        if not match:
            return False

        section_title = match.group("title").strip()

        if not section_title:
            return False

        return self._contains_workflow_keyword(
            section_title.lower()
        )

    # ===============================================================
    # WORKFLOW TITLE
    # ===============================================================

    def _find_workflow_title(
        self,
        region: Region,
    ) -> Optional[tuple[str, str]]:
        """
        Find the top-level workflow heading.

        Example:

            7. Order fulfillment procedure for self-pickup

        Returns:

            ("7", "Order fulfillment procedure for self-pickup")
        """

        lines = self._clean_lines(region.text)

        if not lines:
            return None

        first_line = lines[0]

        match = self.SECTION_PATTERN.match(first_line)

        if not match:
            return None

        section_number = match.group("number").strip()
        section_title = match.group("title").strip()

        if not self._contains_workflow_keyword(
            section_title.lower()
        ):
            return None

        return section_number, section_title

    # ===============================================================
    # STEP EXTRACTION
    # ===============================================================

    def _extract_steps(
        self,
        lines: list[str],
        page_number: int,
        region_id: str,
        document_id: str,
        parent_number: str,
    ) -> list[WorkflowStep]:
        """
        Extract only direct children of the workflow section.

        Example:

            Parent = 7

            7.1  -> accepted
            7.2  -> accepted
            7.3  -> accepted

        But:

            7.1.1 -> rejected
            8.1   -> rejected
            10.3  -> rejected
            6.3.6 -> rejected
        """

        steps: list[WorkflowStep] = []

        for line in lines:
            match = self.STEP_PATTERN.match(line)

            if not match:
                continue

            number = match.group("number").strip()
            content = match.group("content").strip()

            if not content:
                continue

            # -------------------------------------------------------
            # Only direct children belong to this workflow.
            # -------------------------------------------------------

            if not self._is_direct_child(
                number=number,
                parent_number=parent_number,
            ):
                continue

            title, description = (
                self._split_step_content(content)
            )

            step_id = (
                f"{document_id}"
                f"_page_{page_number}"
                f"_step_{number}"
            )

            step = WorkflowStep(
                step_id=step_id,
                number=number,
                title=title,
                description=description,
                page_number=page_number,
                region_id=region_id,
                metadata={
                    "source": "numbered_text",
                    "parent_section": parent_number,
                },
            )

            steps.append(step)

        return steps

    @staticmethod
    def _is_direct_child(
        number: str,
        parent_number: str,
    ) -> bool:
        """
        Verify that a numbered item is a direct child.

        Example:

            parent = 7

            7.1   -> True
            7.2   -> True
            7.5   -> True

            7.1.1 -> False
            8.1   -> False
            10.3  -> False
            6.3.6 -> False
        """

        prefix = f"{parent_number}."

        if not number.startswith(prefix):
            return False

        remainder = number[len(prefix):]

        if not remainder:
            return False

        # Only one numeric component after parent.
        return remainder.isdigit()

    @staticmethod
    def _split_step_content(
        content: str,
    ) -> tuple[str, str]:
        """
        Split a step into title and description.

        Example:

            Delivery Time: Average time from order placement to delivery.

        becomes:

            title:
                Delivery Time

            description:
                Average time from order placement to delivery.
        """

        if ":" in content:
            title, description = content.split(
                ":",
                1,
            )

            title = title.strip()
            description = description.strip()

            if title:
                return title, description

        return content.strip(), ""

    # ===============================================================
    # TRANSITIONS
    # ===============================================================

    def _build_sequential_transitions(
        self,
        workflow: Workflow,
    ) -> list[WorkflowTransition]:
        """
        Build inferred sequential transitions.

        Example:

            7.2 -> 7.3
            7.3 -> 7.4
            7.4 -> 7.5

        These are inferred from document ordering.

        They are NOT explicit diagram arrows.
        """

        transitions: list[WorkflowTransition] = []

        for index in range(
            len(workflow.steps) - 1
        ):
            source = workflow.steps[index]
            target = workflow.steps[index + 1]

            transition_id = (
                f"{workflow.workflow_id}"
                f"_transition_{index + 1}"
            )

            transition = WorkflowTransition(
                transition_id=transition_id,
                source_step_id=source.step_id,
                target_step_id=target.step_id,
                condition=None,
                label="next",
                page_number=source.page_number,
                metadata={
                    "inferred": True,
                    "method": "sequential_order",
                },
            )

            transitions.append(transition)

        return transitions

    # ===============================================================
    # DECISIONS
    # ===============================================================

    def _extract_decisions(
        self,
        workflow: Workflow,
    ) -> list[WorkflowDecision]:
        """
        Detect explicit textual decisions.

        This is intentionally conservative.

        Detected examples:

            Is the payment successful?

            If payment fails...

            Whether the item is available...

        NOT detected:

            The order is accepted.

            Order is approved.

            Restaurant prepares the order.

            Customer arrives.
        """

        decisions: list[WorkflowDecision] = []

        for step in workflow.steps:
            combined_text = (
                f"{step.title} "
                f"{step.description}"
            ).strip()

            lower_text = combined_text.lower()

            if not self._looks_like_decision(
                lower_text
            ):
                continue

            options = self._extract_decision_options(
                lower_text
            )

            decision_id = (
                f"{workflow.workflow_id}"
                f"_decision_"
                f"{step.number.replace('.', '_')}"
            )

            decision = WorkflowDecision(
                decision_id=decision_id,
                step_id=step.step_id,
                question=combined_text,
                options=options,
                page_number=step.page_number,
                metadata={
                    "source": "text",
                    "inferred": True,
                },
            )

            decisions.append(decision)

        return decisions

    def _looks_like_decision(
        self,
        text: str,
    ) -> bool:
        """
        Detect explicit decision language.

        This method is deliberately conservative.

        Strong signals:

            ?
            if ...
            whether ...
            check if ...
            check whether ...
            is ... available
            is ... valid
            is ... successful
            are ... available
            does ... exist
            can ... continue
            should ... continue
        """

        # -----------------------------------------------------------
        # Explicit question.
        # -----------------------------------------------------------

        if "?" in text:
            return True

        # -----------------------------------------------------------
        # Explicit conditional language.
        # -----------------------------------------------------------

        if re.search(
            r"\bif\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        if re.search(
            r"\bwhether\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        # -----------------------------------------------------------
        # Explicit check-if / check-whether language.
        # -----------------------------------------------------------

        if re.search(
            r"\bcheck\s+(if|whether)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        # -----------------------------------------------------------
        # Strong decision sentence patterns.
        # -----------------------------------------------------------

        decision_patterns = [
            r"\bis\s+.+\bavailable\b",
            r"\bare\s+.+\bavailable\b",
            r"\bis\s+.+\bvalid\b",
            r"\bare\s+.+\bvalid\b",
            r"\bis\s+.+\bsuccessful\b",
            r"\bare\s+.+\bsuccessful\b",
            r"\bdoes\s+.+\bexist\b",
            r"\bcan\s+.+\bcontinue\b",
            r"\bshould\s+.+\bcontinue\b",
        ]

        for pattern in decision_patterns:
            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                return True

        return False

    @staticmethod
    def _extract_decision_options(
        text: str,
    ) -> list[str]:
        """
        Extract explicit yes/no options when present.

        More complicated options will later come from
        visual workflow analysis.
        """

        options: list[str] = []

        if re.search(
            r"\byes\b",
            text,
            flags=re.IGNORECASE,
        ):
            options.append("yes")

        if re.search(
            r"\bno\b",
            text,
            flags=re.IGNORECASE,
        ):
            options.append("no")

        return options

    # ===============================================================
    # KEYWORD HELPERS
    # ===============================================================

    def _contains_workflow_keyword(
        self,
        text: str,
    ) -> bool:
        """
        Check whether a section title contains a strong
        workflow/process keyword.
        """

        normalized = re.sub(
            r"\s+",
            " ",
            text.lower(),
        ).strip()

        for keyword in self.WORKFLOW_TITLE_KEYWORDS:
            if self._keyword_matches(
                normalized,
                keyword,
            ):
                return True

        return False

    @staticmethod
    def _keyword_matches(
        text: str,
        keyword: str,
    ) -> bool:
        """
        Word-safe keyword matching.
        """

        return bool(
            re.search(
                rf"\b{re.escape(keyword)}\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    # ===============================================================
    # TEXT CLEANING
    # ===============================================================

    @staticmethod
    def _clean_lines(
        text: str,
    ) -> list[str]:
        """
        Normalize region text into clean non-empty lines.
        """

        lines: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            line = re.sub(
                r"\s+",
                " ",
                line,
            )

            lines.append(line)

        return lines