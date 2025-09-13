"""Dataset loaders for benchmarking."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union


@dataclass
class QAItem:
    """A question-answer item for benchmarking."""
    question: str
    answer: str
    context: str = ""
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def load_qa_dataset(file_path: Union[str, Path], format: str = "auto") -> list[QAItem]:
    """
    Load a QA dataset from file.

    Args:
        file_path: Path to dataset file
        format: File format ("jsonl", "csv", or "auto" to detect from extension)

    Returns:
        List of QAItem objects

    Raises:
        ValueError: If format is unsupported or file cannot be parsed
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Auto-detect format from extension
    if format == "auto":
        ext = file_path.suffix.lower()
        if ext == ".jsonl":
            format = "jsonl"
        elif ext == ".csv":
            format = "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {ext}")

    if format == "jsonl":
        return _load_jsonl(file_path)
    elif format == "csv":
        return _load_csv(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_jsonl(file_path: Path) -> list[QAItem]:
    """Load QA dataset from JSONL format."""
    items = []

    with open(file_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                item = _parse_qa_item(data, line_num)
                items.append(item)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            except KeyError as e:
                raise ValueError(
                    f"Missing required field on line {line_num}: {e}"
                ) from e

    return items


def _load_csv(file_path: Path) -> list[QAItem]:
    """Load QA dataset from CSV format."""
    items = []

    with open(file_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, 2):  # Start at 2 (header is row 1)
            try:
                item = _parse_qa_item(row, row_num)
                items.append(item)
            except KeyError as e:
                raise ValueError(f"Missing required field on row {row_num}: {e}") from e

    return items


def _parse_qa_item(data: dict[str, Any], source_ref: Union[int, str]) -> QAItem:
    """
    Parse a QA item from dictionary data.

    Expected fields:
    - question (required)
    - answer (required)
    - context (optional)
    - Any other fields are stored in metadata
    """
    # Required fields
    question = data.get("question")
    answer = data.get("answer")

    if not question:
        raise KeyError("'question' field is required")
    if not answer:
        raise KeyError("'answer' field is required")

    # Optional fields
    context = data.get("context", "")

    # Metadata (everything else)
    metadata = {}
    for key, value in data.items():
        if key not in ["question", "answer", "context"]:
            metadata[key] = value

    return QAItem(
        question=question,
        answer=answer,
        context=context,
        metadata=metadata
    )
