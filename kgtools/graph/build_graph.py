import jieba
import numpy as np
from mittens import GloVe
from scipy.sparse import csr_matrix
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def build_graph(
    docs: list[str],
    keywords: list[str],
    *,
    embedding_size: int = 200,
    context_size: int = 10,
    glove_epochs: int = 100,
    glasso_epochs: int = 100,
    tolerance: float = 1e-3,
    learning_rate: float = 0.05,
    lambda_glasso: float = 0.1,
    min_weight: float = 0.1,
):
    """构建知识图谱的关系矩阵

    Args:
        docs: 文档列表
        keywords: 关键词列表
        embedding_size: GloVe 词向量维度
        context_size: 上下文窗口大小
        glove_epochs: GloVe 训练轮次
        learning_rate: GloVe 学习率
        lambda_glasso: Graphical Lasso 正则化参数
        min_weight: 最小关系权重阈值

    Returns:
        sparse.csr_matrix: 关键词之间的关系矩阵
    """
    glove = GloVe(
        n=embedding_size,
        max_iter=glove_epochs,
        learning_rate=learning_rate,
    )
    glasso = GraphicalLasso(
        alpha=lambda_glasso,
        assume_centered=True,
        max_iter=glasso_epochs,
        tol=tolerance,
    )

    # 预处理文档
    tokenized_docs = _preprocess_docs(docs, keywords)

    # 构建共现矩阵
    cooccur_matrix = _build_cooccurrence_matrix(
        tokenized_docs,
        keywords,
        context_size,
    )

    # 训练词向量
    embeddings = glove.fit(cooccur_matrix)

    # 构建关系矩阵
    relation_matrix = _build_relation_matrix(
        embeddings,
        glasso,
        min_weight,
    )

    return relation_matrix


def _preprocess_docs(docs: list[str], keywords: list[str]) -> list[list[str]]:
    """预处理文档"""
    for word in keywords:
        jieba.add_word(word)
        jieba.suggest_freq(word, tune=True)

    tokenized = []
    for doc in tqdm(docs, desc="Preprocessing documents"):
        tokens = [w for w in jieba.lcut(doc) if w.strip() and not _is_punct(w)]
        tokenized.append(tokens)

    return tokenized


def _is_punct(word: str) -> bool:
    """检查是否为标点符号"""
    return word in {"，", "。", "\n"}


def _build_cooccurrence_matrix(
    tokenized_docs: list[list[str]],
    keywords: list[str],
    context_size: int,
) -> np.ndarray:
    """构建共现矩阵"""
    word2id = {word: idx for idx, word in enumerate(keywords)}
    vocab_size = len(keywords)
    cooccur = np.zeros((vocab_size, vocab_size), dtype=np.float32)

    for doc in tqdm(tokenized_docs, desc="Building co-occurrence matrix"):
        doc_len = len(doc)
        for i, word in enumerate(doc):
            if word not in word2id:
                continue

            window_start = max(0, i - context_size)
            window_end = min(doc_len, i + context_size + 1)

            for j in range(window_start, window_end):
                if i == j:
                    continue

                context_word = doc[j]
                if context_word not in word2id:
                    continue

                distance = abs(i - j)
                weight = 1.0 / distance
                cooccur[word2id[word], word2id[context_word]] += weight

    return cooccur


def _build_relation_matrix(
    embeddings: np.ndarray,
    glasso: GraphicalLasso,
    min_weight: float,
):
    """构建关系矩阵"""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings.T)

    glasso.fit(embeddings_scaled)
    precision_matrix = glasso.precision_

    relation_matrix = np.abs(precision_matrix)
    np.fill_diagonal(relation_matrix, 0)

    if relation_matrix.max() > 0:
        relation_matrix = relation_matrix / relation_matrix.max()

    relation_matrix[relation_matrix < min_weight] = 0

    return csr_matrix(relation_matrix)
