from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# ============================================================================
# TESSERACT
# ============================================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if Path(TESSERACT_PATH).exists():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================================
# OCR MODEL
# ============================================================================

@dataclass
class OCRText:
    """
    One OCR word/text item detected inside the diagram.
    """

    text: str

    x: int
    y: int

    width: int
    height: int

    confidence: float = 0.0

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            self.x + self.width / 2,
            self.y + self.height / 2,
        )

    @property
    def center_x(self) -> float:
        return self.center[0]

    @property
    def center_y(self) -> float:
        return self.center[1]


# ============================================================================
# DIAGRAM ELEMENT
# ============================================================================

@dataclass
class DiagramElement:
    """
    One detected visual element/node in the diagram.
    """

    element_id: str

    element_type: str

    bbox: tuple[int, int, int, int]

    center: tuple[float, float]

    text: str = ""

    confidence: float = 0.0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Geometry compatibility properties
    # ------------------------------------------------------------------

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def width(self) -> int:
        return self.bbox[2]

    @property
    def height(self) -> int:
        return self.bbox[3]

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.center[0]

    @property
    def center_y(self) -> float:
        return self.center[1]

    @property
    def area(self) -> int:
        return self.width * self.height


# ============================================================================
# DIAGRAM ANALYSIS
# ============================================================================

@dataclass
class DiagramAnalysis:
    """
    Complete analysis result for one diagram image.
    """

    image_path: str

    image_width: int

    image_height: int

    ocr_text: list[OCRText] = field(
        default_factory=list
    )

    elements: list[DiagramElement] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Compatibility collections
    # ------------------------------------------------------------------

    @property
    def rectangles(self) -> list[DiagramElement]:
        return [
            element
            for element in self.elements
            if element.element_type == "rectangle"
        ]

    @property
    def rounded_rectangles(self) -> list[DiagramElement]:
        return [
            element
            for element in self.elements
            if element.element_type == "rounded_rectangle"
        ]

    @property
    def diamonds(self) -> list[DiagramElement]:
        return [
            element
            for element in self.elements
            if element.element_type == "diamond"
        ]

    @property
    def ellipses(self) -> list[DiagramElement]:
        return [
            element
            for element in self.elements
            if element.element_type == "ellipse"
        ]

    @property
    def text_regions(self) -> list[DiagramElement]:
        return [
            element
            for element in self.elements
            if element.element_type == "text_region"
        ]

    @property
    def shape_elements(self) -> list[DiagramElement]:
        return [
            element
            for element in self.elements
            if element.element_type != "text_region"
        ]

    @property
    def element_count(self) -> int:
        return len(self.elements)

    @property
    def ocr_count(self) -> int:
        return len(self.ocr_text)


# ============================================================================
# DIAGRAM EXTRACTOR
# ============================================================================

class DiagramExtractor:
    """
    Extracts structural information from workflow diagrams.

    Current pipeline:

        workflow image
              |
              v
            OCR
              |
              v
        contour detection
              |
              v
        shape classification
              |
              v
        shape clustering
              |
              v
        duplicate removal
              |
              v
        local OCR association
              |
              v
        clean diagram elements

    This class intentionally DOES NOT build the workflow graph yet.

    Graph construction comes after node extraction becomes reliable.
    """

    # ------------------------------------------------------------------
    # Shape thresholds
    # ------------------------------------------------------------------

    MIN_SHAPE_AREA = 900

    MIN_SHAPE_WIDTH = 35

    MIN_SHAPE_HEIGHT = 25

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    MIN_OCR_CONFIDENCE = 20.0

    # ------------------------------------------------------------------
    # Giant container filtering
    # ------------------------------------------------------------------

    MAX_CONTAINER_WIDTH_RATIO = 0.72

    MAX_CONTAINER_HEIGHT_RATIO = 0.72

    MAX_CONTAINER_AREA_RATIO = 0.35

    # ------------------------------------------------------------------
    # Shape duplicate filtering
    # ------------------------------------------------------------------

    DUPLICATE_IOU_THRESHOLD = 0.60

    STRONG_DUPLICATE_IOU_THRESHOLD = 0.45

    CONTAINMENT_THRESHOLD = 0.78

    CENTER_DISTANCE_RATIO = 0.35

    # ------------------------------------------------------------------
    # Text region filtering
    # ------------------------------------------------------------------

    MIN_TEXT_REGION_AREA = 30

    MAX_TEXT_REGION_WIDTH_RATIO = 0.45

    MAX_TEXT_REGION_HEIGHT_RATIO = 0.20

    # ------------------------------------------------------------------
    # OCR association
    # ------------------------------------------------------------------

    OCR_PADDING = 8

    MAX_OCR_DISTANCE = 90

    # ------------------------------------------------------------------
    # Noise OCR
    # ------------------------------------------------------------------

    NOISE_WORDS = {
        "",
        "t",
        "ip",
        "y",
        "ed",
        "omer",
        "acoress",
        "picky",
        "ves",
        "|",
        "tp",
    }

    # ------------------------------------------------------------------
    # Workflow vocabulary
    # ------------------------------------------------------------------

    WORKFLOW_WORDS = {
        "order",
        "restaurant",
        "driver",
        "pickup",
        "pick",
        "deliver",
        "delivery",
        "accepted",
        "accept",
        "preparation",
        "prepared",
        "ready",
        "user",
        "branch",
        "service",
        "assigned",
        "arrive",
        "arrived",
        "success",
        "successful",
        "rate",
    }

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def __init__(
        self,
        tesseract_path: str | None = None,
    ):
        if tesseract_path:
            path = Path(tesseract_path)

            if path.exists():
                pytesseract.pytesseract.tesseract_cmd = str(
                    path
                )

    def analyze(
        self,
        image_path: str | Path,
    ) -> DiagramAnalysis:
        """
        Analyze one diagram image.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Diagram image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        image_width, image_height = image.size

        # --------------------------------------------------------------
        # OCR
        # --------------------------------------------------------------

        ocr_text = self._extract_ocr(
            image
        )

        # --------------------------------------------------------------
        # Shape detection
        # --------------------------------------------------------------

        shape_image = self._prepare_shape_image(
            image
        )

        contours = self._find_contours(
            shape_image
        )

        candidates = self._classify_contours(
            contours,
            image_width,
            image_height,
        )

        # --------------------------------------------------------------
        # Remove giant page containers
        # --------------------------------------------------------------

        candidates = self._remove_container_shapes(
            candidates,
            image_width,
            image_height,
        )

        # --------------------------------------------------------------
        # Remove tiny shapes
        # --------------------------------------------------------------

        candidates = self._remove_small_shapes(
            candidates
        )

        # --------------------------------------------------------------
        # NEW:
        # Cluster physically overlapping contours.
        # --------------------------------------------------------------

        candidates = self._cluster_shape_candidates(
            candidates
        )

        # --------------------------------------------------------------
        # Final duplicate cleanup
        # --------------------------------------------------------------

        candidates = self._remove_duplicate_shapes(
            candidates
        )

        # --------------------------------------------------------------
        # Associate OCR locally
        # --------------------------------------------------------------

        elements = self._associate_ocr_with_shapes(
            candidates,
            ocr_text,
        )

        # --------------------------------------------------------------
        # OCR not belonging to a node
        # --------------------------------------------------------------

        text_regions = self._build_text_regions(
            ocr_text,
            elements,
        )

        elements.extend(
            text_regions
        )

        # --------------------------------------------------------------
        # Final cleanup
        # --------------------------------------------------------------

        elements = self._final_cleanup(
            elements
        )

        # --------------------------------------------------------------
        # Stable IDs
        # --------------------------------------------------------------

        elements = self._assign_element_ids(
            elements
        )

        return DiagramAnalysis(
            image_path=str(
                image_path
            ),
            image_width=image_width,
            image_height=image_height,
            ocr_text=ocr_text,
            elements=elements,
        )

    # =========================================================================
    # OCR
    # =========================================================================

    def _extract_ocr(
        self,
        image: Image.Image,
    ) -> list[OCRText]:
        """
        Extract OCR at word level.

        Important:
        We intentionally keep OCR items relatively small here.

        We do NOT merge a whole diagram row into one giant text region.
        """

        scale = 2.5

        width = max(
            1,
            int(image.width * scale),
        )

        height = max(
            1,
            int(image.height * scale),
        )

        resized = image.resize(
            (
                width,
                height,
            ),
            Image.Resampling.LANCZOS,
        )

        gray = ImageOps.grayscale(
            resized
        )

        gray = ImageEnhance.Contrast(
            gray
        ).enhance(
            2.0
        )

        gray = gray.filter(
            ImageFilter.SHARPEN
        )

        array = np.array(
            gray
        )

        array = cv2.GaussianBlur(
            array,
            (3, 3),
            0,
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        array = clahe.apply(
            array
        )

        processed = Image.fromarray(
            array
        )

        config = (
            "--oem 3 "
            "--psm 11"
        )

        data = pytesseract.image_to_data(
            processed,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        results: list[OCRText] = []

        count = len(
            data["text"]
        )

        for index in range(
            count
        ):
            raw_text = data["text"][index]

            if raw_text is None:
                continue

            text = str(
                raw_text
            ).strip()

            if not text:
                continue

            try:
                confidence = float(
                    data["conf"][index]
                )
            except (
                ValueError,
                TypeError,
            ):
                confidence = 0.0

            if (
                confidence
                < self.MIN_OCR_CONFIDENCE
            ):
                continue

            text = self._normalize_text(
                text
            )

            if self._is_noise_text(
                text
            ):
                continue

            x = int(
                data["left"][index]
                / scale
            )

            y = int(
                data["top"][index]
                / scale
            )

            width = int(
                data["width"][index]
                / scale
            )

            height = int(
                data["height"][index]
                / scale
            )

            if width <= 0 or height <= 0:
                continue

            results.append(
                OCRText(
                    text=text,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    confidence=confidence,
                )
            )

        return results

    # =========================================================================
    # OCR HELPERS
    # =========================================================================

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        text = text.strip()

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text

    def _is_noise_text(
        self,
        text: str,
    ) -> bool:
        normalized = (
            text.strip()
            .lower()
        )

        if normalized in self.NOISE_WORDS:
            return True

        if len(normalized) == 1:
            if not normalized.isalnum():
                return True

        if not any(
            character.isalnum()
            for character in normalized
        ):
            return True

        return False

    # =========================================================================
    # SHAPE PREPROCESSING
    # =========================================================================

    def _prepare_shape_image(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """
        Prepare image for contour detection.

        Hough lines are deliberately NOT used.

        Workflow text produces too many line segments and makes
        connector detection unreliable.
        """

        scale = 1.5

        width = max(
            1,
            int(image.width * scale),
        )

        height = max(
            1,
            int(image.height * scale),
        )

        resized = image.resize(
            (
                width,
                height,
            ),
            Image.Resampling.LANCZOS,
        )

        gray = ImageOps.grayscale(
            resized
        )

        array = np.array(
            gray
        )

        threshold = cv2.adaptiveThreshold(
            array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            7,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
        )

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        return threshold

    # =========================================================================
    # CONTOURS
    # =========================================================================

    def _find_contours(
        self,
        image: np.ndarray,
    ) -> list[np.ndarray]:

        contours, _ = cv2.findContours(
            image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return contours

    # =========================================================================
    # SHAPE CLASSIFICATION
    # =========================================================================

    def _classify_contours(
        self,
        contours: list[np.ndarray],
        image_width: int,
        image_height: int,
    ) -> list[DiagramElement]:

        candidates: list[
            DiagramElement
        ] = []

        scale = 1.5

        for contour in contours:

            contour_area = cv2.contourArea(
                contour
            )

            if contour_area <= 0:
                continue

            x, y, width, height = (
                cv2.boundingRect(
                    contour
                )
            )

            # Convert back from 1.5x image.
            x = int(
                x / scale
            )

            y = int(
                y / scale
            )

            width = int(
                width / scale
            )

            height = int(
                height / scale
            )

            if (
                width
                < self.MIN_SHAPE_WIDTH
            ):
                continue

            if (
                height
                < self.MIN_SHAPE_HEIGHT
            ):
                continue

            area = width * height

            if area < self.MIN_SHAPE_AREA:
                continue

            if x < 0 or y < 0:
                continue

            if (
                x + width
                > image_width + 5
            ):
                continue

            if (
                y + height
                > image_height + 5
            ):
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            if perimeter <= 0:
                continue

            approximation = cv2.approxPolyDP(
                contour,
                0.025 * perimeter,
                True,
            )

            vertices = len(
                approximation
            )

            circularity = (
                4
                * np.pi
                * contour_area
                / (
                    perimeter
                    * perimeter
                )
            )

            aspect_ratio = (
                width
                / max(
                    1,
                    height,
                )
            )

            element_type = (
                self._classify_shape_type(
                    vertices=vertices,
                    circularity=circularity,
                    aspect_ratio=aspect_ratio,
                )
            )

            center = (
                x + width / 2,
                y + height / 2,
            )

            confidence = (
                self._shape_confidence(
                    vertices=vertices,
                    circularity=circularity,
                    element_type=element_type,
                )
            )

            candidates.append(
                DiagramElement(
                    element_id="",
                    element_type=element_type,
                    bbox=(
                        x,
                        y,
                        width,
                        height,
                    ),
                    center=center,
                    confidence=confidence,
                    metadata={
                        "vertices": vertices,
                        "circularity": circularity,
                        "aspect_ratio": aspect_ratio,
                        "contour_area": float(
                            contour_area
                        ),
                    },
                )
            )

        return candidates

    def _classify_shape_type(
        self,
        vertices: int,
        circularity: float,
        aspect_ratio: float,
    ) -> str:

        # Four-sided shapes.
        if vertices == 4:

            if (
                0.65
                <= aspect_ratio
                <= 1.45
            ):
                return "diamond"

            return "rectangle"

        # Strongly circular shapes.
        if circularity >= 0.70:
            return "ellipse"

        # Rounded / irregular workflow boxes.
        if 5 <= vertices <= 12:
            return "rounded_rectangle"

        if vertices >= 13:
            return "ellipse"

        return "rectangle"

    def _shape_confidence(
        self,
        vertices: int,
        circularity: float,
        element_type: str,
    ) -> float:

        if element_type == "diamond":

            if vertices == 4:
                return 0.90

            return 0.70

        if element_type == "rectangle":

            if vertices == 4:
                return 0.88

            return 0.70

        if element_type == "ellipse":

            return min(
                0.95,
                0.55
                + circularity * 0.4,
            )

        if (
            element_type
            == "rounded_rectangle"
        ):
            return 0.78

        return 0.50

    # =========================================================================
    # CONTAINER REMOVAL
    # =========================================================================

    def _remove_container_shapes(
        self,
        elements: list[DiagramElement],
        image_width: int,
        image_height: int,
    ) -> list[DiagramElement]:

        image_area = (
            image_width
            * image_height
        )

        result: list[
            DiagramElement
        ] = []

        for element in elements:

            width_ratio = (
                element.width
                / max(
                    1,
                    image_width,
                )
            )

            height_ratio = (
                element.height
                / max(
                    1,
                    image_height,
                )
            )

            area_ratio = (
                element.area
                / max(
                    1,
                    image_area,
                )
            )

            is_container = (
                width_ratio
                >= self.MAX_CONTAINER_WIDTH_RATIO
                or height_ratio
                >= self.MAX_CONTAINER_HEIGHT_RATIO
                or area_ratio
                >= self.MAX_CONTAINER_AREA_RATIO
            )

            if is_container:
                continue

            result.append(
                element
            )

        return result

    # =========================================================================
    # SMALL SHAPES
    # =========================================================================

    def _remove_small_shapes(
        self,
        elements: list[DiagramElement],
    ) -> list[DiagramElement]:

        result = []

        for element in elements:

            if (
                element.width
                < self.MIN_SHAPE_WIDTH
            ):
                continue

            if (
                element.height
                < self.MIN_SHAPE_HEIGHT
            ):
                continue

            if (
                element.area
                < self.MIN_SHAPE_AREA
            ):
                continue

            result.append(
                element
            )

        return result

    # =========================================================================
    # NEW SHAPE CLUSTERING
    # =========================================================================

    def _cluster_shape_candidates(
        self,
        elements: list[DiagramElement],
    ) -> list[DiagramElement]:
        """
        Merge contours that represent the same physical workflow node.

        Example:

            diamond
               +
            rounded rectangle
               +
            inner contour

        should become ONE node.

        We use spatial overlap/containment rather than only
        element_type because OpenCV can classify the same physical
        shape differently depending on the contour.
        """

        if not elements:
            return []

        ordered = sorted(
            elements,
            key=lambda element: (
                element.area,
                element.confidence,
            ),
            reverse=True,
        )

        clusters: list[
            list[DiagramElement]
        ] = []

        for candidate in ordered:

            assigned_cluster = None

            for cluster in clusters:

                representative = (
                    self._choose_cluster_representative(
                        cluster
                    )
                )

                if self._same_physical_node(
                    candidate,
                    representative,
                ):
                    assigned_cluster = cluster
                    break

            if assigned_cluster is None:
                clusters.append(
                    [candidate]
                )
            else:
                assigned_cluster.append(
                    candidate
                )

        result: list[
            DiagramElement
        ] = []

        for cluster in clusters:

            representative = (
                self._choose_cluster_representative(
                    cluster
                )
            )

            merged_bbox = (
                self._merge_cluster_bbox(
                    cluster
                )
            )

            # Keep representative's shape
            # because it normally has the strongest
            # contour classification.
            representative.bbox = (
                merged_bbox
            )

            representative.center = (
                merged_bbox[0]
                + merged_bbox[2] / 2,
                merged_bbox[1]
                + merged_bbox[3] / 2,
            )

            representative.metadata[
                "cluster_size"
            ] = len(cluster)

            representative.metadata[
                "cluster_types"
            ] = [
                item.element_type
                for item in cluster
            ]

            result.append(
                representative
            )

        return result

    def _choose_cluster_representative(
        self,
        cluster: list[DiagramElement],
    ) -> DiagramElement:

        # Prefer diamond because decision diamonds
        # are semantically important.
        diamonds = [
            item
            for item in cluster
            if item.element_type
            == "diamond"
        ]

        if diamonds:
            return max(
                diamonds,
                key=lambda item: (
                    item.confidence,
                    item.area,
                ),
            )

        # Prefer rectangle over weak rounded contours.
        rectangles = [
            item
            for item in cluster
            if item.element_type
            == "rectangle"
        ]

        if rectangles:
            return max(
                rectangles,
                key=lambda item: (
                    item.confidence,
                    item.area,
                ),
            )

        return max(
            cluster,
            key=lambda item: (
                item.confidence,
                item.area,
            ),
        )

    def _same_physical_node(
        self,
        first: DiagramElement,
        second: DiagramElement,
    ) -> bool:

        iou = self._iou(
            first.bbox,
            second.bbox,
        )

        if (
            iou
            >= self.DUPLICATE_IOU_THRESHOLD
        ):
            return True

        containment = max(
            self._containment_ratio(
                first.bbox,
                second.bbox,
            ),
            self._containment_ratio(
                second.bbox,
                first.bbox,
            ),
        )

        if (
            containment
            >= self.CONTAINMENT_THRESHOLD
        ):
            return True

        # Same center and similar size.
        distance = self._center_distance(
            first,
            second,
        )

        average_size = (
            (
                first.width
                + first.height
            )
            / 2
            + (
                second.width
                + second.height
            )
            / 2
        ) / 2

        if (
            distance
            <= average_size
            * self.CENTER_DISTANCE_RATIO
        ):
            size_ratio = (
                min(
                    first.area,
                    second.area,
                )
                / max(
                    first.area,
                    second.area,
                )
            )

            if size_ratio >= 0.45:
                return True

        # Moderate overlap between a diamond and a rounded contour
        # is very often the same workflow node.
        if (
            iou
            >= self.STRONG_DUPLICATE_IOU_THRESHOLD
            and (
                first.element_type
                == "diamond"
                or second.element_type
                == "diamond"
            )
        ):
            return True

        return False

    def _merge_cluster_bbox(
        self,
        cluster: list[DiagramElement],
    ) -> tuple[int, int, int, int]:

        x0 = min(
            item.x
            for item in cluster
        )

        y0 = min(
            item.y
            for item in cluster
        )

        x1 = max(
            item.x1
            for item in cluster
        )

        y1 = max(
            item.y1
            for item in cluster
        )

        return (
            x0,
            y0,
            x1 - x0,
            y1 - y0,
        )

    # =========================================================================
    # FINAL DUPLICATE REMOVAL
    # =========================================================================

    def _remove_duplicate_shapes(
        self,
        elements: list[DiagramElement],
    ) -> list[DiagramElement]:

        if not elements:
            return []

        ordered = sorted(
            elements,
            key=lambda element: (
                element.area,
                element.confidence,
            ),
            reverse=True,
        )

        kept: list[
            DiagramElement
        ] = []

        for candidate in ordered:

            duplicate = False

            for existing in kept:

                iou = self._iou(
                    candidate.bbox,
                    existing.bbox,
                )

                if (
                    iou
                    >= self.DUPLICATE_IOU_THRESHOLD
                ):
                    duplicate = True
                    break

                containment = max(
                    self._containment_ratio(
                        candidate.bbox,
                        existing.bbox,
                    ),
                    self._containment_ratio(
                        existing.bbox,
                        candidate.bbox,
                    ),
                )

                if (
                    containment
                    >= self.CONTAINMENT_THRESHOLD
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(
                    candidate
                )

        return kept

    # =========================================================================
    # OCR ASSOCIATION
    # =========================================================================

    def _associate_ocr_with_shapes(
        self,
        elements: list[DiagramElement],
        ocr_text: list[OCRText],
    ) -> list[DiagramElement]:
        """
        Associate OCR locally with shapes.

        Important difference from the previous approach:

        We do not allow one huge OCR region to swallow
        text belonging to multiple nodes.

        Each OCR item is assigned to at most one shape.
        """

        assignments: dict[
            int,
            list[OCRText]
        ] = {}

        for ocr_index, text_item in enumerate(
            ocr_text
        ):

            best_element_index = None

            best_score = float(
                "-inf"
            )

            for element_index, element in enumerate(
                elements
            ):

                score = (
                    self._ocr_shape_score(
                        text_item,
                        element,
                    )
                )

                if score > best_score:
                    best_score = score
                    best_element_index = (
                        element_index
                    )

            if (
                best_element_index
                is not None
                and best_score
                > 0
            ):
                assignments.setdefault(
                    best_element_index,
                    [],
                ).append(
                    text_item
                )

        for element_index, items in (
            assignments.items()
        ):

            element = elements[
                element_index
            ]

            items.sort(
                key=lambda item: (
                    item.y,
                    item.x,
                )
            )

            # Prevent excessively large OCR associations.
            items = self._filter_local_ocr(
                items,
                element,
            )

            if not items:
                continue

            element.text = self._compose_node_text(
                items
            )

            element.metadata[
                "ocr_items"
            ] = len(items)

            element.metadata[
                "ocr_confidence"
            ] = (
                sum(
                    item.confidence
                    for item in items
                )
                / len(items)
            )

            element.metadata[
                "ocr_boxes"
            ] = [
                {
                    "text": item.text,
                    "x": item.x,
                    "y": item.y,
                    "width": item.width,
                    "height": item.height,
                    "confidence": item.confidence,
                }
                for item in items
            ]

            ocr_confidence = (
                element.metadata[
                    "ocr_confidence"
                ]
                / 100
            )

            element.confidence = min(
                0.99,
                (
                    element.confidence
                    + ocr_confidence
                )
                / 2
                + 0.15,
            )

        return elements

    def _ocr_shape_score(
        self,
        text_item: OCRText,
        element: DiagramElement,
    ) -> float:
        """
        Score how strongly an OCR item belongs to a node.
        """

        center_x = text_item.center_x
        center_y = text_item.center_y

        # --------------------------------------------------------------
        # 1. OCR center inside node
        # --------------------------------------------------------------

        if (
            element.x - self.OCR_PADDING
            <= center_x
            <= element.x1 + self.OCR_PADDING
            and
            element.y - self.OCR_PADDING
            <= center_y
            <= element.y1 + self.OCR_PADDING
        ):
            return 1000.0

        # --------------------------------------------------------------
        # 2. OCR box intersects node
        # --------------------------------------------------------------

        overlap = self._ocr_overlap_ratio(
            text_item,
            element,
        )

        if overlap > 0:
            return (
                500.0
                + overlap * 200
            )

        # --------------------------------------------------------------
        # 3. OCR near node
        # --------------------------------------------------------------

        distance = self._point_to_box_distance(
            center_x,
            center_y,
            element.bbox,
        )

        if (
            distance
            > self.MAX_OCR_DISTANCE
        ):
            return -1.0

        # Prefer smaller distance.
        return (
            200.0
            - distance
        )

    def _filter_local_ocr(
        self,
        items: list[OCRText],
        element: DiagramElement,
    ) -> list[OCRText]:
        """
        Remove OCR items that are clearly outside the node.

        This prevents neighboring node text from getting swallowed.
        """

        result: list[
            OCRText
        ] = []

        max_distance = max(
            35,
            min(
                75,
                int(
                    (
                        element.width
                        + element.height
                    )
                    * 0.35
                ),
            ),
        )

        for item in items:

            inside = (
                element.x - 5
                <= item.center_x
                <= element.x1 + 5
                and
                element.y - 5
                <= item.center_y
                <= element.y1 + 5
            )

            overlap = (
                self._ocr_overlap_ratio(
                    item,
                    element,
                )
            )

            distance = (
                self._point_to_box_distance(
                    item.center_x,
                    item.center_y,
                    element.bbox,
                )
            )

            if inside:
                result.append(
                    item
                )
                continue

            if overlap >= 0.30:
                result.append(
                    item
                )
                continue

            if distance <= max_distance:
                result.append(
                    item
                )

        return result

    def _compose_node_text(
        self,
        items: list[OCRText],
    ) -> str:
        """
        Compose OCR items belonging to one node.

        Groups items by vertical line first.
        """

        if not items:
            return ""

        lines: list[
            list[OCRText]
        ] = []

        for item in items:

            assigned = False

            for line in lines:

                average_y = (
                    sum(
                        x.center_y
                        for x in line
                    )
                    / len(line)
                )

                tolerance = max(
                    12,
                    min(
                        item.height,
                        30,
                    ),
                )

                if (
                    abs(
                        item.center_y
                        - average_y
                    )
                    <= tolerance
                ):
                    line.append(
                        item
                    )
                    assigned = True
                    break

            if not assigned:
                lines.append(
                    [item]
                )

        for line in lines:
            line.sort(
                key=lambda item: item.x
            )

        lines.sort(
            key=lambda line: (
                sum(
                    item.center_y
                    for item in line
                )
                / len(line)
            )
        )

        text_lines = []

        for line in lines:

            line_text = " ".join(
                item.text
                for item in line
            )

            line_text = self._normalize_text(
                line_text
            )

            if line_text:
                text_lines.append(
                    line_text
                )

        return " ".join(
            text_lines
        ).strip()

    # =========================================================================
    # TEXT REGIONS
    # =========================================================================

    def _build_text_regions(
        self,
        ocr_text: list[OCRText],
        elements: list[DiagramElement],
    ) -> list[DiagramElement]:
        """
        Create text regions only for OCR that does not belong to a node.

        We intentionally avoid giant cross-diagram OCR regions.
        """

        regions: list[
            DiagramElement
        ] = []

        for index, item in enumerate(
            ocr_text
        ):

            if self._ocr_belongs_to_any_shape(
                item,
                elements,
            ):
                continue

            area = (
                item.width
                * item.height
            )

            if (
                area
                < self.MIN_TEXT_REGION_AREA
            ):
                continue

            regions.append(
                DiagramElement(
                    element_id="",
                    element_type="text_region",
                    bbox=(
                        item.x,
                        item.y,
                        item.width,
                        item.height,
                    ),
                    center=item.center,
                    text=item.text,
                    confidence=(
                        item.confidence
                        / 100
                    ),
                    metadata={
                        "ocr_index": index,
                        "ocr_confidence": item.confidence,
                    },
                )
            )

        # Merge only nearby text boxes.
        return self._merge_text_regions(
            regions
        )

    def _ocr_belongs_to_any_shape(
        self,
        item: OCRText,
        elements: list[DiagramElement],
    ) -> bool:

        for element in elements:

            if (
                self._ocr_shape_score(
                    item,
                    element,
                )
                >= 500
            ):
                return True

        return False

    def _merge_text_regions(
        self,
        regions: list[DiagramElement],
    ) -> list[DiagramElement]:
        """
        Merge only text boxes that are genuinely adjacent.

        We specifically reject huge cross-diagram merges.
        """

        if not regions:
            return []

        regions = sorted(
            regions,
            key=lambda item: (
                item.y,
                item.x,
            ),
        )

        merged: list[
            DiagramElement
        ] = []

        for region in regions:

            merged_into_existing = False

            for existing in merged:

                if not self._text_regions_are_close(
                    existing,
                    region,
                ):
                    continue

                new_x = min(
                    existing.x,
                    region.x,
                )

                new_y = min(
                    existing.y,
                    region.y,
                )

                new_x1 = max(
                    existing.x1,
                    region.x1,
                )

                new_y1 = max(
                    existing.y1,
                    region.y1,
                )

                new_width = (
                    new_x1
                    - new_x
                )

                new_height = (
                    new_y1
                    - new_y
                )

                existing_area = (
                    new_width
                    * new_height
                )

                # Don't create giant text regions.
                if (
                    new_width
                    > 1000
                ):
                    continue

                if (
                    new_height
                    > 120
                ):
                    continue

                existing.text = (
                    existing.text
                    + " "
                    + region.text
                ).strip()

                existing.bbox = (
                    new_x,
                    new_y,
                    new_width,
                    new_height,
                )

                existing.center = (
                    new_x
                    + new_width / 2,
                    new_y
                    + new_height / 2,
                )

                existing.confidence = (
                    existing.confidence
                    + region.confidence
                ) / 2

                merged_into_existing = True

                break

            if not merged_into_existing:
                merged.append(
                    region
                )

        return merged

    def _text_regions_are_close(
        self,
        first: DiagramElement,
        second: DiagramElement,
    ) -> bool:

        # Same approximate line.
        vertical_distance = abs(
            first.center_y
            - second.center_y
        )

        if (
            vertical_distance
            <= max(
                first.height,
                second.height,
            ) * 1.5
        ):
            horizontal_gap = min(
                abs(
                    second.x
                    - first.x1
                ),
                abs(
                    first.x
                    - second.x1
                ),
            )

            if horizontal_gap <= 45:
                return True

        # Vertically stacked short text.
        horizontal_distance = abs(
            first.center_x
            - second.center_x
        )

        if (
            horizontal_distance
            <= max(
                first.width,
                second.width,
            ) * 0.5
        ):
            vertical_gap = min(
                abs(
                    second.y
                    - first.y1
                ),
                abs(
                    first.y
                    - second.y1
                ),
            )

            if vertical_gap <= 25:
                return True

        return False

    # =========================================================================
    # FINAL CLEANUP
    # =========================================================================

    def _final_cleanup(
        self,
        elements: list[DiagramElement],
    ) -> list[DiagramElement]:

        cleaned: list[
            DiagramElement
        ] = []

        for element in elements:

            # ----------------------------------------------------------
            # Text region cleanup
            # ----------------------------------------------------------

            if (
                element.element_type
                == "text_region"
            ):

                if not element.text.strip():
                    continue

                if self._is_noise_text(
                    element.text
                ):
                    continue

                # Reject giant text regions.
                if (
                    element.width
                    > 1000
                ):
                    continue

                if (
                    element.height
                    > 120
                ):
                    continue

            # ----------------------------------------------------------
            # Shape cleanup
            # ----------------------------------------------------------

            else:

                if (
                    element.width
                    < self.MIN_SHAPE_WIDTH
                ):
                    continue

                if (
                    element.height
                    < self.MIN_SHAPE_HEIGHT
                ):
                    continue

            cleaned.append(
                element
            )

        # Sort spatially.
        cleaned.sort(
            key=lambda element: (
                element.center_y,
                element.center_x,
            )
        )

        return cleaned

    # =========================================================================
    # ID ASSIGNMENT
    # =========================================================================

    def _assign_element_ids(
        self,
        elements: list[DiagramElement],
    ) -> list[DiagramElement]:

        counters: dict[
            str,
            int,
        ] = {}

        for element in elements:

            element_type = (
                element.element_type
            )

            current = counters.get(
                element_type,
                0,
            )

            element.element_id = (
                f"{element_type}_{current}"
            )

            counters[element_type] = (
                current + 1
            )

        return elements

    # =========================================================================
    # GEOMETRY HELPERS
    # =========================================================================

    def _iou(
        self,
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:

        ax, ay, aw, ah = first

        bx, by, bw, bh = second

        ax1 = ax + aw
        ay1 = ay + ah

        bx1 = bx + bw
        by1 = by + bh

        intersection_x0 = max(
            ax,
            bx,
        )

        intersection_y0 = max(
            ay,
            by,
        )

        intersection_x1 = min(
            ax1,
            bx1,
        )

        intersection_y1 = min(
            ay1,
            by1,
        )

        intersection_width = max(
            0,
            intersection_x1
            - intersection_x0,
        )

        intersection_height = max(
            0,
            intersection_y1
            - intersection_y0,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        if (
            intersection_area
            <= 0
        ):
            return 0.0

        first_area = (
            aw * ah
        )

        second_area = (
            bw * bh
        )

        union = (
            first_area
            + second_area
            - intersection_area
        )

        if union <= 0:
            return 0.0

        return (
            intersection_area
            / union
        )

    def _containment_ratio(
        self,
        inner: tuple[int, int, int, int],
        outer: tuple[int, int, int, int],
    ) -> float:

        ix, iy, iw, ih = inner

        ox, oy, ow, oh = outer

        ix1 = ix + iw
        iy1 = iy + ih

        ox1 = ox + ow
        oy1 = oy + oh

        intersection_x0 = max(
            ix,
            ox,
        )

        intersection_y0 = max(
            iy,
            oy,
        )

        intersection_x1 = min(
            ix1,
            ox1,
        )

        intersection_y1 = min(
            iy1,
            oy1,
        )

        intersection_width = max(
            0,
            intersection_x1
            - intersection_x0,
        )

        intersection_height = max(
            0,
            intersection_y1
            - intersection_y0,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        inner_area = (
            iw * ih
        )

        if inner_area <= 0:
            return 0.0

        return (
            intersection_area
            / inner_area
        )

    def _center_distance(
        self,
        first: DiagramElement,
        second: DiagramElement,
    ) -> float:

        return float(
            np.sqrt(
                (
                    first.center_x
                    - second.center_x
                )
                ** 2
                +
                (
                    first.center_y
                    - second.center_y
                )
                ** 2
            )
        )

    def _ocr_overlap_ratio(
        self,
        text_item: OCRText,
        element: DiagramElement,
    ) -> float:

        tx0 = text_item.x
        ty0 = text_item.y
        tx1 = text_item.x1
        ty1 = text_item.y1

        ex0 = element.x
        ey0 = element.y
        ex1 = element.x1
        ey1 = element.y1

        ix0 = max(
            tx0,
            ex0,
        )

        iy0 = max(
            ty0,
            ey0,
        )

        ix1 = min(
            tx1,
            ex1,
        )

        iy1 = min(
            ty1,
            ey1,
        )

        width = max(
            0,
            ix1 - ix0,
        )

        height = max(
            0,
            iy1 - iy0,
        )

        intersection = (
            width * height
        )

        text_area = (
            text_item.width
            * text_item.height
        )

        if text_area <= 0:
            return 0.0

        return (
            intersection
            / text_area
        )

    def _point_to_box_distance(
        self,
        point_x: float,
        point_y: float,
        bbox: tuple[int, int, int, int],
    ) -> float:

        x, y, width, height = bbox

        x1 = x + width
        y1 = y + height

        if point_x < x:
            dx = x - point_x
        elif point_x > x1:
            dx = point_x - x1
        else:
            dx = 0.0

        if point_y < y:
            dy = y - point_y
        elif point_y > y1:
            dy = point_y - y1
        else:
            dy = 0.0

        return float(
            np.sqrt(
                dx * dx
                + dy * dy
            )
        )