import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import pdfplumber
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL.Image import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_THREAD_LOCK = threading.Lock()
_OCR_INSTANCES: dict[str, Any] = {}
_OCR_ENGINES: dict[str, Callable[[str, list[int], int], str]] = {}


def extract_text(
    file_path: str | Path,
    file_type: str | None = None,
    *,
    first_page: int = 1,
    last_page: int | None = None,
    ocr_engine: str = "cnocr",
    num_workers: int = 4,
    force_ocr: bool = False,
) -> str:
    """Extract text content from file. For PDFs, tries pdfplumber first, then falls back to OCR.

    Args:
        file_path: Path to the file
        file_type: Type of file ('pdf' or 'txt')
        first_page: First page to process (1-based)
        last_page: Last page to process (inclusive), None for all pages
        ocr_engine: OCR engine to use ('cnocr', 'tesseract', or 'paddleocr')
        num_workers: Number of parallel workers
        force_ocr: Whether to skip pdfplumber and use OCR directly

    Returns:
        Extracted text content

    Raises:
        ValueError: If file_type or ocr_engine is not supported
        ImportError: If required OCR package is not installed
    """
    file_path = str(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_type is None:
        file_type = file_path.split(".")[-1]

    ocr_engine = ocr_engine.lower()
    if ocr_engine not in _OCR_ENGINES:
        raise ValueError(f"Unsupported OCR engine: {ocr_engine}")

    file_type = file_type.lower()
    if file_type not in {"pdf", "txt"}:
        raise ValueError(f"Unsupported file type: {file_type}")

    if file_type == "txt":
        return _read_txt(file_path)

    pages = _get_pdf_pages(file_path, first_page, last_page)

    if not force_ocr:
        if text := _extract_with_pdfplumber(file_path, pages):
            return text
        logger.info("PDFplumber extraction failed, trying OCR...")

    text = _OCR_ENGINES[ocr_engine](file_path, pages, num_workers)

    if ocr_engine == "tesseract":
        del os.environ["OMP_THREAD_LIMIT"]

    if not text:
        logger.warning("No text content extracted")

    return text


def _register_ocr_engine(name: str):
    """Register OCR engine and handle parallel processing.

    This decorator registers the OCR function to _OCR_ENGINES and automatically
    handles parallel processing. The decorated function should focus only on
    processing a single page.

    Args:
        name: Name of the OCR engine to register

    Example:
        @register_ocr_engine("cnocr")
        def process_with_cnocr(file_path: str, page_num: int) -> str | None:
            # Process single page logic here
            ...
    """

    def decorator(
        func: Callable[[str, int], str | None],
    ) -> Callable[[str, list[int], int], str]:
        @wraps(func)
        def wrapper(file_path: str, pages: list[int], num_workers: int) -> str:
            logger.info(f"Starting {name} extraction with {num_workers} workers...")
            results = []

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(func, file_path, num): num for num in pages}

                for future in tqdm(
                    as_completed(futures), total=len(pages), unit="page"
                ):
                    page_num = futures[future]
                    try:
                        if page_text := future.result():
                            results.append((page_num, page_text))
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")

            return "\n".join(text for _, text in sorted(results))

        _OCR_ENGINES[name] = wrapper
        return wrapper

    return decorator


# OCR engine implementations
@_register_ocr_engine("cnocr")
def _extract_with_cnocr(file_path: str, page_num: int) -> str | None:
    """Extract text from a single page using CnOCR"""
    try:
        from cnocr import CnOcr
    except ImportError:
        raise ImportError("Please install cnocr with: pip install cnocr[ort-cpu]")

    if "cnocr" not in _OCR_INSTANCES:
        with _THREAD_LOCK:
            if "cnocr" not in _OCR_INSTANCES:
                _OCR_INSTANCES["cnocr"] = CnOcr()

    ocr = _OCR_INSTANCES["cnocr"]
    img = _convert_page_to_image(file_path, page_num)
    if img is None:
        return None

    try:
        res = ocr.ocr(img)
        return "\n".join(line["text"] for line in res).strip()
    except Exception as e:
        logger.warning(f"Error processing page {page_num}: {e}")
    return None


@_register_ocr_engine("tesseract")
def _extract_with_tesseract(file_path: str, page_num: int) -> str | None:
    """Extract text from a single page using Tesseract"""
    try:
        import pytesseract
    except ImportError:
        raise ImportError("Please install pytesseract with: pip install pytesseract")

    os.environ["OMP_THREAD_LIMIT"] = "1"

    img = _convert_page_to_image(file_path, page_num)
    if img is None:
        return None

    try:
        text = pytesseract.image_to_string(img, lang="chi_sim")
        return text.strip()
    except Exception as e:
        logger.warning(f"Error processing page {page_num}: {e}")
        return None


@_register_ocr_engine("paddleocr")
def _extract_with_paddle(file_path: str, page_num: int) -> str | None:
    """Extract text from a single page using PaddleOCR"""
    try:
        import numpy as np
        from paddleocr import PaddleOCR
    except ImportError:
        raise ImportError("Please install paddleocr with: pip install paddleocr")

    if "paddle" not in _OCR_INSTANCES:
        with _THREAD_LOCK:
            if "paddle" not in _OCR_INSTANCES:
                _OCR_INSTANCES["paddle"] = PaddleOCR(
                    use_angle_cls=True,
                    lang="ch",
                    show_log=False,
                    use_mp=True,
                )

    ocr = _OCR_INSTANCES["paddle"]
    img = _convert_page_to_image(file_path, page_num, "RGB")
    if img is None:
        return None

    try:
        with _THREAD_LOCK:
            result = ocr.ocr(np.array(img))
        return "\n".join(line[1][0] for line in result[0]).strip()
    except Exception as e:
        logger.warning(f"Error processing page {page_num}: {e}")
        return None


def _extract_with_pdfplumber(file_path: str, pages: list[int]) -> str:
    """Extract text from PDF using PDFplumber"""
    logger.info("Starting PDFplumber extraction...")
    texts = []

    with pdfplumber.open(file_path, pages=pages) as pdf:
        for page in tqdm(pdf.pages, total=len(pages), unit="page"):
            if text := page.extract_text():
                texts.append(text)

    return "".join(texts)


# Helper functions
def _get_pdf_pages(file_path: str, first_page: int, last_page: int | None) -> list[int]:
    """Get PDF page range"""
    pdf_info = pdfinfo_from_path(file_path)
    total_pages: int = pdf_info["Pages"]
    last_page = min(last_page, total_pages) if last_page is not None else total_pages
    return list(range(first_page, last_page + 1))


def _read_txt(file_path: str) -> str:
    """Read text from TXT file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _convert_page_to_image(
    file_path: str, page_num: int, convert_mode: str = "L"
) -> Image | None:
    """Convert PDF page to image"""
    try:
        images = convert_from_path(file_path, first_page=page_num, last_page=page_num)
        return images[0].convert(convert_mode)
    except Exception as e:
        logger.warning(f"Error converting page {page_num} to image: {e}")
        return None
