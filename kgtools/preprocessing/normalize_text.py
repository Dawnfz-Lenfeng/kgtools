import difflib
import re

# 字符集
HANZI = r"\u4e00-\u9fa5"
COMMAS = ",;，；"
STOPS = ".!?。！？"
PUNCTS = COMMAS + STOPS

# 长度阈值
MIN_PARA = 15
MIN_SENT = 5

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
    char_threshold: int = 2,
    sentence_threshold: float = 0.9,
) -> str:
    """清理文本：移除非中文字符，统一标点符号，去除重复内容

    :param text: 待清理的文本
    :param char_threshold: 连续重复字符的最小长度
    :param sentence_threshold: 句子重复度的阈值
    """
    if not text:
        return ""

    text = RE_NONZH.sub("", text)
    text = standardize_punctuation(text)
    text = remove_redundant_text(text, char_threshold, sentence_threshold)
    text = clean_consecutive_puncts(text)
    if text == "。":
        return ""

    return text


def clean_consecutive_puncts(text: str) -> str:
    """清理连续的句号和逗号"""

    def replace_puncts(match: re.Match) -> str:
        puncts = match.group(0)
        if "。" in puncts:
            return "。"

        return "，"

    return RE_MULTI_PUNCT.sub(replace_puncts, text)


def standardize_punctuation(text: str) -> str:
    """统一标点符号格式"""
    text = RE_COMMA.sub("，", text)
    text = RE_STOP.sub("。", text)
    # 处理连续标点
    return clean_consecutive_puncts(text)


def remove_redundant_text(
    text: str, char_threshold: int = 2, sentence_threshold: float = 0.9
) -> str:
    """移除文本中的重复内容"""
    text = compress_chars(text, char_threshold)

    cleaned_paras = []
    for para in RE_PARA.split(text):
        para = para.replace("\n", "")
        if not para:
            continue

        cleaned = deduplicate_sentences(para, sentence_threshold)
        if len(cleaned) >= MIN_PARA:
            cleaned_paras.append(cleaned)

    return "\n".join(cleaned_paras) + "。"


def compress_chars(text: str, char_threshold: int = 2) -> str:
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
            chars.append(curr_char)
        else:
            chars.extend([curr_char] * repeat_count)
        i = j

    return "".join(chars)


def deduplicate_sentences(paragraph: str, similarity_threshold: float = 0.9) -> str:
    """去除段落中的相似句子"""
    sentences = [s for s in RE_SENT.split(paragraph) if len(s) >= MIN_SENT]
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
