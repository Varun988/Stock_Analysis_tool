from io import BytesIO

import pandas as pd
from fastapi import UploadFile
from pypdf import PdfReader


SUPPORTED_TEXT_EXTRACTION_EXTENSIONS = (
    ".pdf",
    ".xml",
    ".txt",
    ".csv",
    ".xlsx",
    ".xls",
)


def _decode_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1")


def _extract_text_from_pdf(file_bytes: bytes, password: str | None = None) -> str:
    reader = PdfReader(BytesIO(file_bytes))

    if reader.is_encrypted:
        if not password:
            raise ValueError("PDF is password protected. Please provide password.")

        decrypt_result = reader.decrypt(password)

        if decrypt_result == 0:
            raise ValueError("Could not decrypt PDF. Please check the password.")

    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(page_text)

    extracted_text = "\n".join(text_parts).strip()

    if not extracted_text:
        raise ValueError(
            "Could not extract readable text from PDF. "
            "Scanned/image PDFs are not supported yet."
        )

    return extracted_text


def _extract_text_from_excel(file_bytes: bytes, file_name: str) -> str:
    file_name_lower = file_name.lower()

    if file_name_lower.endswith(".xls"):
        df = pd.read_excel(BytesIO(file_bytes), engine="xlrd")
    else:
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")

    if df.empty:
        raise ValueError("Uploaded Excel file does not contain any rows.")

    return df.to_csv(index=False)


async def extract_text_from_uploaded_file(
    file: UploadFile,
    password: str | None = None,
) -> dict:
    file_name = file.filename or ""
    file_name_lower = file_name.lower()

    if not file_name_lower.endswith(SUPPORTED_TEXT_EXTRACTION_EXTENSIONS):
        raise ValueError(
            "Unsupported file format. Supported formats: PDF, XML, TXT, CSV, XLSX, XLS."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    if file_name_lower.endswith(".pdf"):
        extracted_text = _extract_text_from_pdf(
            file_bytes=file_bytes,
            password=password,
        )

    elif file_name_lower.endswith((".xml", ".txt", ".csv")):
        extracted_text = _decode_bytes(file_bytes)

    elif file_name_lower.endswith((".xlsx", ".xls")):
        extracted_text = _extract_text_from_excel(
            file_bytes=file_bytes,
            file_name=file_name,
        )

    else:
        raise ValueError("Unsupported file format.")

    return {
        "file_name": file_name,
        "text": extracted_text,
    }