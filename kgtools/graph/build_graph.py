import numpy as np
import jieba
from collections import defaultdict
from tqdm import tqdm
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def build_graph(
    documents: list[str],
    keywords: list[str],
    min_weight: float = 0.1,
    normalize: bool = True,
    embedding_size: int = 200,
    context_size: int = 10,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 0.05,
) -> np.ndarray:
    """
    Build relationship matrix between keywords using GloVe embeddings

    Args:
        documents: 待分析的文档列表
        keywords: 需要分析关系的关键词列表
        min_weight: 保留关系的最小权重阈值
        normalize: 是否归一化到0-1范围
        embedding_size: 词向量维度
        context_size: 上下文窗口半径
        epochs: 训练轮次
        batch_size: 训练批次大小
        learning_rate: 学习率

    Returns:
        关键词关系矩阵 (n_keywords x n_keywords)
    """
    # 初始化模型组件
    word2id = {}
    id2word = []
    cooccur_matrix = defaultdict(float)

    # 阶段1：预处理和共现矩阵构建
    def preprocess():
        nonlocal word2id, id2word

        # 加载自定义词典
        for word in keywords:
            jieba.add_word(word)
            jieba.suggest_freq(word, tune=True)

        # 分词和统计
        word_counts = defaultdict(int)
        tokenized_docs = []
        for doc in tqdm(documents, desc="文档预处理"):
            words = [w for w in jieba.lcut(doc) if w in keywords]
            for word in words:
                word_counts[word] += 1
            tokenized_docs.append(words)

        # 构建词汇表
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        id2word = [w[0] for w in sorted_words]
        word2id = {w: i for i, w in enumerate(id2word)}

        # 构建共现矩阵
        for doc in tqdm(tokenized_docs, desc="构建共现矩阵"):
            for pos, word in enumerate(doc):
                if word not in word2id:
                    continue
                start = max(0, pos - context_size)
                end = min(len(doc), pos + context_size + 1)
                for ctx_pos in range(start, end):
                    if ctx_pos == pos:
                        continue
                    ctx_word = doc[ctx_pos]
                    if ctx_word in word2id:
                        distance = abs(pos - ctx_pos)
                        decay = 1.0 / distance
                        cooccur_matrix[(word2id[word], word2id[ctx_word])] += decay

    # 阶段2：GloVe模型训练
    def train_glove():
        # 准备数据
        i_indices, j_indices, counts = [], [], []
        for (i, j), count in cooccur_matrix.items():
            i_indices.append(i)
            j_indices.append(j)
            counts.append(count)

        # 构建计算图
        tf.reset_default_graph()

        # 输入占位符
        focal_input = tf.placeholder(tf.int32, shape=[None])
        context_input = tf.placeholder(tf.int32, shape=[None])
        cooccur_count = tf.placeholder(tf.float32, shape=[None])

        # 嵌入层
        vocab_size = len(id2word)
        with tf.variable_scope("embeddings"):
            focal_emb = tf.get_variable(
                "focal_emb",
                [vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(-1, 1),
            )
            context_emb = tf.get_variable(
                "context_emb",
                [vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(-1, 1),
            )
            focal_bias = tf.get_variable(
                "focal_bias",
                [vocab_size],
                initializer=tf.random_uniform_initializer(-1, 1),
            )
            context_bias = tf.get_variable(
                "context_bias",
                [vocab_size],
                initializer=tf.random_uniform_initializer(-1, 1),
            )

        # 前向计算
        focal_vec = tf.nn.embedding_lookup(focal_emb, focal_input)
        context_vec = tf.nn.embedding_lookup(context_emb, context_input)
        focal_b = tf.nn.embedding_lookup(focal_bias, focal_input)
        context_b = tf.nn.embedding_lookup(context_bias, context_input)

        # 损失函数
        log_counts = tf.log(cooccur_count + 1e-8)
        prediction = tf.reduce_sum(focal_vec * context_vec, 1) + focal_b + context_b
        loss = tf.reduce_mean(tf.square(prediction - log_counts))
        optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

        # 训练循环
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(epochs):
                indices = np.random.permutation(len(i_indices))
                total_loss = 0
                for start in tqdm(
                    range(0, len(i_indices), batch_size),
                    desc=f"Epoch {epoch+1}/{epochs}",
                ):
                    end = min(start + batch_size, len(i_indices))
                    batch_idx = indices[start:end]

                    feed_dict = {
                        focal_input: np.array(i_indices)[batch_idx],
                        context_input: np.array(j_indices)[batch_idx],
                        cooccur_count: np.array(counts)[batch_idx],
                    }
                    _, batch_loss = sess.run([optimizer, loss], feed_dict=feed_dict)
                    total_loss += batch_loss

                print(
                    f"Epoch {epoch+1} Loss: {total_loss/(len(i_indices)//batch_size + 1):.4f}"
                )

            # 合并嵌入向量
            embeddings = sess.run(focal_emb + context_emb)

        return embeddings

    # 阶段3：构建关系矩阵
    def build_matrix(embeddings):
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.where(norms == 0, 1e-8, norms)

        # 计算相似度矩阵
        sim_matrix = np.dot(normalized, normalized.T)
        np.fill_diagonal(sim_matrix, 0)

        # 归一化和阈值处理
        if normalize:
            sim_matrix = (sim_matrix - sim_matrix.min()) / (
                sim_matrix.max() - sim_matrix.min()
            )
        sim_matrix[sim_matrix < min_weight] = 0

        return sim_matrix

    # 执行流程
    preprocess()
    embeddings = train_glove()
    relation_matrix = build_matrix(embeddings)

    # 对齐关键词顺序
    keyword_indices = [word2id[kw] for kw in keywords if kw in word2id]
    filtered_matrix = relation_matrix[np.ix_(keyword_indices, keyword_indices)]

    return filtered_matrix
