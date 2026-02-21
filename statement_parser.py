#!/usr/bin/env python3
"""Offline parser for bank statement PDFs (text + scanned).

Features:
- Extracts transactions from text-based PDFs using pdfplumber table/text parsing.
- Falls back to OCR (pytesseract) for scanned pages.
- Normalizes output to: date, description, amount, balance, operation_type.
- Supports bank-specific profile configuration in YAML.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pdfplumber
import pytesseract
import yaml
from PIL import Image

SUPPORTED_PDF_SUFFIXES = {".pdf"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUPPORTED_INPUT_SUFFIXES = SUPPORTED_PDF_SUFFIXES | SUPPORTED_IMAGE_SUFFIXES


@dataclass
class ParserProfile:
    date_regex: str
    amount_regex: str
    balance_regex: str
    debit_keywords: list[str]
    credit_keywords: list[str]
    date_format: str = "%d.%m.%Y"
    decimal_sep: str = ","
    thousands_sep: str = " "
    ocr_lang: str = "ukr+eng+deu"
    amount_strategy: str = "last"
    debit_credit_order: str = "debit_credit"
    transaction_type_regex: str | None = None


DEFAULT_PROFILE = ParserProfile(
    date_regex=r"\b\d{2}[./-]\d{2}[./-]\d{4}\b",
    amount_regex=r"[-+]?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})",
    balance_regex=r"[-+]?\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{2})",
    debit_keywords=["спис", "debit", "withdraw", "оплата", "платіж"],
    credit_keywords=["зарах", "credit", "deposit", "надход", "повернення"],
)


def load_profile(profile_path: Path | None, profile_name: str | None) -> ParserProfile:
    if not profile_path:
        return DEFAULT_PROFILE

    data = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "profiles" not in data:
        raise ValueError("Profile YAML must contain top-level 'profiles' map")

    profiles = data["profiles"]
    selected = profile_name or "default"
    if selected not in profiles:
        raise ValueError(f"Profile '{selected}' not found in {profile_path}")

    merged = {**DEFAULT_PROFILE.__dict__, **profiles[selected]}
    return ParserProfile(**merged)


def normalize_number(value: str, decimal_sep: str, thousands_sep: str) -> float | None:
    if not value:
        return None
    cleaned = value.strip().replace("\u00A0", " ")
    cleaned = cleaned.replace(thousands_sep, "")
    if decimal_sep != ".":
        cleaned = cleaned.replace(decimal_sep, ".")
    cleaned = cleaned.replace(",", ".")
    cleaned = re.sub(r"[^0-9.+-]", "", cleaned)
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def guess_operation_type(description: str, amount: float | None, profile: ParserProfile) -> str:
    text = description.lower()
    if any(k in text for k in profile.credit_keywords):
        return "credit"
    if any(k in text for k in profile.debit_keywords):
        return "debit"
    if amount is not None:
        if amount < 0:
            return "debit"
        if amount > 0:
            return "credit"
    return "unknown"


def extract_transaction_type(line: str, profile: ParserProfile) -> str:
    if not profile.transaction_type_regex:
        return ""
    match = re.search(profile.transaction_type_regex, line, re.IGNORECASE)
    if not match:
        return ""
    if "type" in match.groupdict():
        return match.group("type").strip()
    if match.groups():
        return match.group(1).strip()
    return match.group(0).strip()


def pick_amount(values: list[float], line: str, profile: ParserProfile) -> float | None:
    if not values:
        return None
    if profile.amount_strategy == "first":
        return values[0]
    if profile.amount_strategy == "debit_credit" and len(values) >= 2:
        debit_value, credit_value = values[0], values[1]
        if profile.debit_credit_order == "credit_debit":
            credit_value, debit_value = values[0], values[1]
        if abs(debit_value) > 0:
            return -abs(debit_value)
        if abs(credit_value) > 0:
            return abs(credit_value)
    if any(k in line.lower() for k in profile.debit_keywords):
        return -abs(values[-1])
    if any(k in line.lower() for k in profile.credit_keywords):
        return abs(values[-1])
    return values[-1]


def resolve_amount_balance(
    numeric_amounts: list[float],
    line: str,
    profile: ParserProfile,
) -> tuple[float | None, float | None, set[int]]:
    if not numeric_amounts:
        return None, None, set()

    if profile.amount_strategy == "debit_credit":
        amount_values = numeric_amounts[:2]
        amount = pick_amount(amount_values, line, profile)
        used = {0, 1} if len(numeric_amounts) >= 2 else {0}
        balance = numeric_amounts[2] if len(numeric_amounts) >= 3 else None
        if len(numeric_amounts) >= 3:
            used.add(2)
        return amount, balance, used

    amount = pick_amount(numeric_amounts, line, profile)

    if profile.amount_strategy == "first":
        amount_index = 0
        balance_index = len(numeric_amounts) - 1 if len(numeric_amounts) > 1 else None
    else:
        amount_index = len(numeric_amounts) - 1
        balance_index = len(numeric_amounts) - 2 if len(numeric_amounts) > 1 else None

    used = {amount_index}
    balance = None
    if balance_index is not None and balance_index != amount_index:
        balance = numeric_amounts[balance_index]
        used.add(balance_index)

    return amount, balance, used


def lines_to_candidate_rows(lines: list[str], date_regex: str) -> list[str]:
    grouped: list[str] = []
    current: list[str] = []
    for raw in lines:
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        if re.search(date_regex, line):
            if current:
                grouped.append(" ".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        grouped.append(" ".join(current))
    return grouped


def row_to_transaction(cells: list[str], profile: ParserProfile) -> dict[str, Any] | None:
    line = " | ".join(c for c in cells if c).strip()
    if not line:
        return None

    date_match = re.search(profile.date_regex, line)
    if not date_match:
        return None

    date = date_match.group(0)
    date_start, date_end = date_match.span()
    parsed_matches: list[tuple[re.Match[str], float]] = []
    for match in re.finditer(profile.amount_regex, line):
        start, end = match.span()
        if start < date_end and end > date_start:
            continue
        parsed = normalize_number(match.group(0), profile.decimal_sep, profile.thousands_sep)
        if parsed is not None:
            parsed_matches.append((match, parsed))

    if not parsed_matches:
        return None

    numeric_amounts = [value for _, value in parsed_matches]
    amount, balance, used_indexes = resolve_amount_balance(numeric_amounts, line, profile)

    description = line.replace(date, "", 1)
    for idx in sorted(used_indexes, reverse=True):
        if 0 <= idx < len(parsed_matches):
            description = description.replace(parsed_matches[idx][0].group(0), "", 1)

    description = re.sub(r"\s+", " ", description).strip(" |-")
    transaction_type = extract_transaction_type(description, profile)
    op_type = guess_operation_type(description, amount, profile)

    return {
        "date": date,
        "transaction_type": transaction_type,
        "description": description,
        "amount": amount,
        "balance": balance,
        "operation_type": op_type,
    }


def parse_text_pdf(pdf_path: Path, profile: ParserProfile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            table = page.extract_table(
                {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                }
            )
            if table:
                for raw_row in table:
                    cells = [c.strip() if c else "" for c in raw_row]
                    tx = row_to_transaction(cells, profile)
                    if tx:
                        rows.append(tx)
                continue

            text = page.extract_text() or ""
            for line in lines_to_candidate_rows(text.splitlines(), profile.date_regex):
                tx = row_to_transaction([line], profile)
                if tx:
                    rows.append(tx)

    return rows


def parse_ocr_pdf(pdf_path: Path, profile: ParserProfile, dpi: int = 300) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            rendered = page.to_image(resolution=dpi).original
            pil_image = rendered if isinstance(rendered, Image.Image) else Image.fromarray(rendered)
            text = pytesseract.image_to_string(pil_image, lang=profile.ocr_lang)
            for line in lines_to_candidate_rows(text.splitlines(), profile.date_regex):
                tx = row_to_transaction([line], profile)
                if tx:
                    rows.append(tx)
    return rows


def parse_ocr_image(image_path: Path, profile: ParserProfile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Image.open(image_path) as image:
        text = pytesseract.image_to_string(image, lang=profile.ocr_lang)
    for line in lines_to_candidate_rows(text.splitlines(), profile.date_regex):
        tx = row_to_transaction([line], profile)
        if tx:
            rows.append(tx)
    return rows


def deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (
            row.get("date"),
            row.get("transaction_type"),
            row.get("description"),
            row.get("amount"),
            row.get("balance"),
            row.get("operation_type"),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def write_output(rows: list[dict[str, Any]], out_path: Path) -> None:
    df = pd.DataFrame(
        rows,
        columns=["date", "transaction_type", "description", "amount", "balance", "operation_type"],
    )
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    elif out_path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(out_path, index=False)
    else:
        raise ValueError("Output file extension must be .csv, .xlsx or .xls")


def run_parser(
    input_path: Path,
    output_path: Path,
    force_ocr: bool = False,
    profile_config: Path | None = None,
    profile_name: str | None = None,
) -> int:
    profile = load_profile(profile_config, profile_name)
    input_suffix = input_path.suffix.lower()

    if input_suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError("Input file extension must be .pdf, .jpg, .jpeg or .png")

    if input_suffix in SUPPORTED_IMAGE_SUFFIXES:
        rows = parse_ocr_image(input_path, profile)
    else:
        rows = parse_ocr_pdf(input_path, profile) if force_ocr else parse_text_pdf(input_path, profile)
        if not rows and not force_ocr:
            rows = parse_ocr_pdf(input_path, profile)

    rows = deduplicate(rows)
    if not rows:
        raise ValueError("No transactions found. Try --ocr or adjust profile regexes.")

    write_output(rows, output_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse bank statement PDF/JPG/PNG to CSV/XLSX")
    parser.add_argument("input", type=Path, help="Path to source PDF/JPG/PNG")
    parser.add_argument("output", type=Path, help="Path to destination .csv/.xlsx")
    parser.add_argument("--ocr", action="store_true", help="Force OCR mode (for scanned PDFs)")
    parser.add_argument("--profile-config", type=Path, help="YAML file with bank profiles")
    parser.add_argument("--profile", type=str, help="Profile name from YAML (default: default)")
    args = parser.parse_args()

    parsed_count = run_parser(
        input_path=args.input,
        output_path=args.output,
        force_ocr=args.ocr,
        profile_config=args.profile_config,
        profile_name=args.profile,
    )
    print(f"Parsed {parsed_count} transactions -> {args.output}")


if __name__ == "__main__":
    main()
