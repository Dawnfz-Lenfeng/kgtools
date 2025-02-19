from kgtools.preprocessing.normalize_text import (
    compress_chars,
    normalize_text,
    remove_redundant_text,
    standardize_punctuation,
)


def test_standardize_punctuation():
    """测试标点符号规范化"""
    cases = [
        ("测试，，，测试。", "测试，测试。"),
        ("测试。。。测试。", "测试。测试。"),
        ("测试，。测试。", "测试。测试。"),
        ("测试！。测试。", "测试。测试。"),
        ("测试。，测试。", "测试。测试。"),
    ]
    for input_text, expected in cases:
        assert standardize_punctuation(input_text) == expected


def test_compress_chars():
    """测试压缩重复字符"""
    cases = [
        ("试试试试试测试。", "试测试。"),  # 超过阈值压缩
        ("好好学习天天向上。", "好好学习天天向上。"),  # 未超过阈值不压缩
        ("啊啊啊测试测试。", "啊测试测试。"),  # 刚好达到阈值压缩
    ]
    for input_text, expected in cases:
        assert compress_chars(input_text, char_threshold=2) == expected


def test_remove_redundant_text():
    """测试去除重复内容"""
    # 测试相似句子
    text = "这是一个比较长的测试句子。这是一个很相似的长句子。这是完全不同的长句子。"
    result = remove_redundant_text(text)
    assert "这是一个比较长的测试句子。" in result
    assert "这是完全不同的长句子。" in result

    # 测试完全重复的句子
    text = "这是一个重复的长句子测试。这是一个重复的长句子测试。这是不同的长句子。"
    result = remove_redundant_text(text)
    assert result.count("这是一个重复的长句子测试") == 1


def test_clean_text_handles_empty_input():
    """测试空输入处理"""
    assert normalize_text("") == ""
    assert normalize_text(" ") == ""
    assert normalize_text("\n") == ""


def test_clean_text_removes_short_content():
    """测试删除过短内容"""
    text = "短。这是一个比较完整的长句子测试。太短。"
    result = normalize_text(text)
    assert "这是一个比较完整的长句子测试。" in result
    assert "短。" not in result
    assert "太短。" not in result


def test_clean_text_integration():
    """集成测试"""
    cases = [
        ("Hello世界123！！！", ""),  # 太短会被清理
        (
            "这是一个完整的中文句子用来测试效果Hello123！！！",
            "这是一个完整的中文句子用来测试效果。",
        ),
        ("试试试试试", ""),  # 太短会被清理
        (
            """
        Hello世界123！！！,,,测试测试。。。
        这是一个重复的长句子测试。这是一个重复的长句子测试。
        这是另一个完整的中文句子用来测试！！！
        """,
            "这是另一个完整的中文句子用来测试。",
        ),
    ]

    for input_text, expected in cases:
        assert normalize_text(input_text) == expected
