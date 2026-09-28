from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Chunk:
    """
    Represents one searchable chunk.
    """

    chunk_id: str

    document_id: str

    text: str

    metadata: Dict[str, str] = field(default_factory=dict)

    token_count: int = 0