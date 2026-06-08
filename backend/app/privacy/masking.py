# backend/app/privacy/masking.py

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskingResult:
    masked_text: str
    masked_fields: list[str]
    mask_count: int


EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# Indian + general phone forms:
# +91 9876543210, 98765 43210, 09876543210
PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"
)

# Indian PAN format: ABCDE1234F
PAN_PATTERN = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    re.IGNORECASE,
)

# Long IDs: account/client/reference IDs etc.
# Avoid masking small quantities/prices by requiring 12+ digits.
LONG_NUMBER_PATTERN = re.compile(
    r"(?<!\d)\d{12,}(?!\d)"
)

# Common statement metadata lines that may contain personal info.
SENSITIVE_LINE_KEYWORDS = (
    "name",
    "email",
    "mobile",
    "phone",
    "pan",
    "address",
    "client id",
    "client code",
    "account",
    "folio",
    "dp id",
    "demat",
    "bo id",
    "user id",
)


def _mask_pattern(
    text: str,
    pattern: re.Pattern[str],
    replacement: str,
    field_name: str,
    masked_fields: list[str],
) -> tuple[str, int]:
    new_text, count = pattern.subn(replacement, text)
    if count > 0 and field_name not in masked_fields:
        masked_fields.append(field_name)
    return new_text, count


def _mask_sensitive_metadata_lines(
    text: str,
    masked_fields: list[str],
) -> tuple[str, int]:
    lines = text.splitlines()
    masked_count = 0
    result_lines: list[str] = []

    for line in lines:
        normalized = " ".join(line.lower().split())

        # Only mask likely metadata lines, not table rows.
        # Example: "Name: Varun Kumar" -> "Name: [MASKED_PERSONAL_METADATA]"
        matched_keyword = next(
            (
                keyword
                for keyword in SENSITIVE_LINE_KEYWORDS
                if keyword in normalized
            ),
            None,
        )

        if matched_keyword and ":" in line and len(line) <= 180:
            key = line.split(":", 1)[0].strip()
            result_lines.append(f"{key}: [MASKED_PERSONAL_METADATA]")
            masked_count += 1
            if "personal_metadata" not in masked_fields:
                masked_fields.append("personal_metadata")
        else:
            result_lines.append(line)

    return "\n".join(result_lines), masked_count


def mask_sensitive_text(text: str | None) -> MaskingResult:
    """Mask obvious personal/sensitive data before sending text to AI.

    This is intentionally conservative and regex-based.
    It should not change investment table values like quantity, price,
    current value, invested value, gain/loss, etc.
    """
    if not text:
        return MaskingResult(masked_text="", masked_fields=[], mask_count=0)

    masked_text = str(text)
    masked_fields: list[str] = []
    total_count = 0

    masked_text, count = _mask_pattern(
        masked_text,
        EMAIL_PATTERN,
        "[MASKED_EMAIL]",
        "email",
        masked_fields,
    )
    total_count += count

    masked_text, count = _mask_pattern(
        masked_text,
        PHONE_PATTERN,
        "[MASKED_PHONE]",
        "phone",
        masked_fields,
    )
    total_count += count

    masked_text, count = _mask_pattern(
        masked_text,
        PAN_PATTERN,
        "[MASKED_PAN]",
        "pan",
        masked_fields,
    )
    total_count += count

    masked_text, count = _mask_pattern(
        masked_text,
        LONG_NUMBER_PATTERN,
        "[MASKED_LONG_ID]",
        "long_numeric_id",
        masked_fields,
    )
    total_count += count

    masked_text, count = _mask_sensitive_metadata_lines(
        masked_text,
        masked_fields,
    )
    total_count += count

    return MaskingResult(
        masked_text=masked_text,
        masked_fields=masked_fields,
        mask_count=total_count,
    )