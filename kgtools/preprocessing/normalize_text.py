import difflib
import re

# 字符集
HANZI = r"\u4e00-\u9fa5"
COMMAS = ",;，；"
STOPS = ".!?。！？"
PUNCTS = COMMAS + STOPS

# 正则
RE_PUNCT = re.compile(f"，。|。，|[{COMMAS}]+|[{STOPS}]+")
RE_COMMA = re.compile(f"[{COMMAS}]+")
RE_STOP = re.compile(f"[{STOPS}]+")
RE_NONZH = re.compile(f"[^{HANZI}{PUNCTS}\n]")
RE_PARA = re.compile(f"(?<=[{STOPS}])\n")
RE_SENT = re.compile(r"(?<=[。\.])\s*")
RE_HEAD = re.compile(f"^[{PUNCTS}]+")
RE_MULTI_PUNCT = re.compile(r"[，。]+")


def normalize_text(
    text: str | None,
    *,
    min_paragraph_length: int = 15,
    min_sentence_length: int = 5,
    char_threshold: int = 2,
    sentence_threshold: float = 0.95,
) -> str:
    """Normalize Chinese text by cleaning and standardizing content.

    Args:
        text: The input text to normalize. Can be None.
        min_paragraph_length: Minimum length for a paragraph to be kept (default: 15)
        min_sentence_length: Minimum length for a sentence to be kept (default: 5)
        char_threshold: Maximum number of consecutive repeated characters to keep (default: 2)
        sentence_threshold: Similarity threshold for sentence deduplication (default: 0.95)

    Returns:
        str: The normalized text with the following characteristics:
            - Only contains Chinese characters and punctuation marks
            - Only uses "，" for commas and "。" for periods
            - Paragraphs are separated by "。\\n"
            - No whitespace within paragraphs
            - Empty string if input is None or results in no valid content

    Notes:
        - Sentences shorter than min_sentence_length will be removed
        - Paragraphs shorter than min_paragraph_length will be removed
        - Similar sentences (similarity >= sentence_threshold) will be deduplicated
        - Consecutive repeated characters beyond char_threshold will be compressed
    """
    if not text:
        return ""

    text = RE_NONZH.sub("", text)
    text = _standardize_punctuation(text)
    text = _compress_chars(text, char_threshold)
    text = _remove_redundant_text(
        text, sentence_threshold, min_paragraph_length, min_sentence_length
    )
    text = _clean_consecutive_puncts(text)
    if text == "。":
        return ""

    return text


def _clean_consecutive_puncts(text: str) -> str:
    """清理连续的句号和逗号"""

    def replace_puncts(match: re.Match) -> str:
        puncts = match.group(0)
        if "。" in puncts:
            return "。"

        return "，"

    return RE_MULTI_PUNCT.sub(replace_puncts, text)


def _standardize_punctuation(text: str) -> str:
    """统一标点符号格式"""
    text = RE_COMMA.sub("，", text)
    text = RE_STOP.sub("。", text)
    # 处理连续标点
    return _clean_consecutive_puncts(text)


def _remove_redundant_text(
    text: str,
    sentence_threshold: float,
    min_paragraph_length: int,
    min_sentence_length: int,
) -> str:
    """移除文本中的重复内容"""
    cleaned_paras = []
    for para in RE_PARA.split(text):
        para = para.replace("\n", "")
        if not para:
            continue

        cleaned = _deduplicate_sentences(para, sentence_threshold, min_sentence_length)
        if len(cleaned) >= min_paragraph_length:
            cleaned_paras.append(cleaned)

    return "\n".join(cleaned_paras) + "。"


def _compress_chars(text: str, char_threshold: int) -> str:
    """压缩连续重复的字符"""
    if not text:
        return ""

    chars = []
    i = 0
    while i < len(text):
        curr_char = text[i]
        j = i + 1
        while j < len(text) and text[j] == curr_char:
            j += 1

        repeat_count = j - i
        if repeat_count > char_threshold:
            chars.extend([curr_char] * char_threshold)
        else:
            chars.extend([curr_char] * repeat_count)
        i = j

    return "".join(chars)


def _deduplicate_sentences(
    paragraph: str, similarity_threshold: float, min_sentence_length: int
) -> str:
    """去除段落中的相似句子"""
    sentences = [s for s in RE_SENT.split(paragraph) if len(s) >= min_sentence_length]
    if not sentences:
        return ""

    sent_features = {}
    for sent in sentences:
        sent_features[sent] = set(sent)

    unique_sentences = []
    used = set()

    for i, sent1 in enumerate(sentences):
        if sent1 in used:
            continue

        chars1 = sent_features[sent1]
        unique_sentences.append(sent1)
        used.add(sent1)

        for sent2 in sentences[i + 1 :]:
            if sent2 in used:
                continue

            chars2 = sent_features[sent2]
            overlap = len(chars1 & chars2) / len(chars1 | chars2)

            if overlap >= similarity_threshold:
                if (
                    difflib.SequenceMatcher(None, sent1, sent2).ratio()
                    >= similarity_threshold
                ):
                    used.add(sent2)

    return RE_HEAD.sub("", "".join(unique_sentences))
