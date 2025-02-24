from pathlib import Path

from kgtools.preprocessing import normalize_text

# Test data path
TESTS_ROOT = Path(__file__).parent.parent
TEST_TXT = TESTS_ROOT / "data" / "test.txt"


def test_basic_punctuation():
    """Test punctuation standardization"""
    cases = [
        ("测试,测试;测试！测试测试", "测试，测试，测试。测试测试。"),  # 加长句子
        ("测试测试。。。这也是测试测试！！", "测试测试。这也是测试测试。"),
        ("测试测试，。这也是测试测试?", "测试测试。这也是测试测试。"),
        ("这是一个测试！。这也是测试...", "这是一个测试。这也是测试。"),
        ("第一句测试。，第二句测试!？", "第一句测试。第二句测试。"),
    ]
    for input_text, expected in cases:
        assert (
            normalize_text(input_text, min_paragraph_length=5, min_sentence_length=4)
            == expected
        )


def test_repeated_chars():
    """Test repeated character compression"""
    cases = [
        ("啊啊啊啊啊太好了呀呀呀", "啊啊太好了呀呀。"),  # 加长句子
        ("好好学习天天向上", "好好学习天天向上。"),  # 正好2个不压缩
        ("哈哈哈哈哈哈笑死我了", "哈哈笑死我了。"),  # 多个重复字符
        ("太棒了太棒了！！！！！", "太棒了太棒了。"),  # 重复标点处理
    ]
    for input_text, expected in cases:
        assert (
            normalize_text(input_text, min_paragraph_length=5, min_sentence_length=4)
            == expected
        )


def test_similar_sentences():
    """Test similar sentence deduplication"""
    cases = [
        # 完全相同的句子
        ("今天天气真不错啊。今天天气真不错啊。", "今天天气真不错啊。"),
        # 不相似的句子
        (
            "今天下雨了真糟糕。明天会放晴真不错。",
            "今天下雨了真糟糕。明天会放晴真不错。",
        ),
    ]
    for input_text, expected in cases:
        assert (
            normalize_text(input_text, min_paragraph_length=8, min_sentence_length=5)
            == expected
        )


def test_paragraph_handling():
    """Test paragraph processing"""
    cases = [
        # 正常段落
        (
            "第一段第一句话完整。第一段第二句话结束。\n第二段内容完整的句子。",
            "第一段第一句话完整。第一段第二句话结束。\n第二段内容完整的句子。",
        ),
        # 多个换行符
        (
            "这是第一个完整段落。\n\n\n这是第二个完整段落。",
            "这是第一个完整段落。\n这是第二个完整段落。",
        ),
        # 段落中的空格
        (
            "段落开始内容  中间有空格  结束语。\n这是下一段完整内容。",
            "段落开始内容中间有空格结束语。\n这是下一段完整内容。",
        ),
    ]
    for input_text, expected in cases:
        assert (
            normalize_text(input_text, min_paragraph_length=8, min_sentence_length=5)
            == expected
        )


def test_mixed_content():
    """Test mixed content handling"""
    cases = [
        # 中英文混合
        ("Hello世界123！", "世界。"),
        # 特殊字符
        ("测试#@$%特殊字符", "测试特殊字符。"),
        # 空白字符
        ("有很多   空格  和\t制表符", "有很多空格和制表符。"),
    ]
    for input_text, expected in cases:
        assert (
            normalize_text(input_text, min_sentence_length=2, min_paragraph_length=2)
            == expected
        )


def test_real_text_file():
    """Test with real text file"""
    # 读取测试文件
    with open(TEST_TXT, "r", encoding="utf-8") as f:
        content = f.read()

    # 规范化文本
    normalized = normalize_text(content)

    # 检查结果
    def check_normalized_text(text: str) -> bool:
        # 1. 检查标点符号
        for char in text:
            if char in "!?;:，。！？；：":
                assert char in ["，", "。"], f"Found invalid punctuation: {char}"

        # 2. 检查段落分隔
        paragraphs = text.split("\n")
        for i, para in enumerate(paragraphs):
            # 检查每个段落是否以。结尾
            if i < len(paragraphs) - 1:  # 除最后一段外
                assert para.endswith("。"), f"Paragraph {i} doesn't end with 。"

            # 3. 检查段落内部没有空白字符
            assert not any(
                c.isspace() for c in para
            ), f"Found whitespace in paragraph {i}"

            # 4. 检查段落内没有换行符
            assert "\n" not in para, f"Found newline in paragraph {i}"

        return True

    assert check_normalized_text(normalized)

    # 额外检查
    assert len(normalized) > 0, "Normalized text should not be empty"
    assert "。\n" in normalized, "Paragraphs should be separated by 。\\n"
    assert not normalized.endswith("\n"), "Text should not end with newline"
    assert normalized.endswith("。"), "Text should end with 。"


def test_edge_cases():
    """Test edge cases"""
    assert normalize_text("") == ""  # 空字符串
    assert normalize_text(" ") == ""  # 只有空格
    assert normalize_text("\n\n\n") == ""  # 只有换行
    assert normalize_text("。。。") == ""  # 只有标点
    assert normalize_text("a b c") == ""  # 只有英文
    assert normalize_text("123") == ""  # 只有数字
    assert normalize_text("测") == ""  # 单个字符
