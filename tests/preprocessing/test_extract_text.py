from pathlib import Path

import pytest

from kgtools.preprocessing import extract_text

# 示例文件路径
TESTS_ROOT = Path(__file__).parent.parent
TEST_PDF = TESTS_ROOT / "data" / "test.pdf"
TEST_TXT = TESTS_ROOT / "data" / "test.txt"


def test_extract_text_txt():
    """测试从txt文件提取文本"""
    text = extract_text(TEST_TXT, "txt")
    assert len(text) > 0


def test_extract_text_pdf():
    """测试从PDF文件提取文本"""
    text = extract_text(TEST_PDF, "pdf")
    assert len(text) > 0


def test_extract_text_pdf_with_ocr():
    """测试使用OCR提取PDF文本"""
    text = extract_text(TEST_PDF, "pdf", force_ocr=True)
    assert len(text) > 0


def test_invalid_ocr_engine():
    """测试不支持的OCR引擎"""
    with pytest.raises(ValueError) as exc_info:
        extract_text(TEST_PDF, "pdf", ocr_engine="invalid_engine")
    assert "Unsupported OCR engine" in str(exc_info.value)


@pytest.mark.parametrize("ocr_engine", ["cnocr", "tesseract"])
def test_different_ocr_engines(ocr_engine: str):
    """测试不同的OCR引擎"""
    text = extract_text(TEST_PDF, "pdf", ocr_engine=ocr_engine, force_ocr=True)
    assert len(text) > 0


def test_extract_text_not_found():
    """测试文件不存在的情况"""
    with pytest.raises(FileNotFoundError):
        extract_text(Path("not_exists.txt"), "txt")


def test_empty_file(tmp_path: Path):
    """测试空文件"""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    text = extract_text(empty_file, "txt")
    assert text == ""


def test_pdf_page_range():
    """测试PDF页面范围提取"""
    # 只提取第一页
    text1 = extract_text(TEST_PDF, "pdf", first_page=1, last_page=1)
    # 只提取第二页
    text2 = extract_text(TEST_PDF, "pdf", first_page=2, last_page=2)
    # 确保不同页面提取的内容不同
    assert text1 != text2
